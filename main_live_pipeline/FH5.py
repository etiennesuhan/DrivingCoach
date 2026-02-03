import datetime
import socket
import sqlite3
import struct
import time
import ast
import sys
import numpy as np
import pandas as pd
import pathlib
import ollama
import re
import tiktoken

from typing import Iterable
from pathlib import Path
from pandas import DataFrame
from itertools import product


class Listener():
    def __init__(self, DB_FILE):
        self.UDP_IP = "0.0.0.0"
        self.UDP_PORT = 5300
        self.DB_FILE = DB_FILE
        self.MAX_PACKETS_PER_SECOND = 1
        self.MIN_SAVE_INTERVAL = 1.0 / self.MAX_PACKETS_PER_SECOND
        self.FORZA_FIELDS = self.load_tuples('main_live_pipeline/forza_fields.txt')
        self.SCHEMAS = struct.calcsize("<" + "".join(fmt for _, fmt in self.FORZA_FIELDS))
        self.BASE_STRUCT_FORMAT = "<" + "".join(fmt for _, fmt in self.FORZA_FIELDS)
        
        
    def run(self):
        conn = self.init_db(self.DB_FILE)
        sock = self.create_socket(self.UDP_IP, self.UDP_PORT)
        print(f"Lausche auf UDP {self.UDP_IP}:{self.UDP_PORT} ...")
        last_saved = 0.0

        try:
            while True:
                data, _ = sock.recvfrom(512)
                parsed_result = self.parse_telemetry(data)
                schema_name, parsed = parsed_result
                if parsed.get("engine_max_rpm", 0) == 0:
                    continue
                now = time.monotonic()
                if now - last_saved < self.MIN_SAVE_INTERVAL:
                    continue
                last_saved = now

                timestamp_utc = (datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z")
                self.insert_sample(conn, timestamp_utc, data, schema_name, parsed)
        except KeyboardInterrupt:
            print("\nBeende Listener...")
        finally:
            conn.close()
            sock.close()

    def load_tuples(self, path: str):
        return ast.literal_eval("[" + Path(path).read_text(encoding="utf-8").strip().strip().rstrip(",") + "]")

    def create_socket(self, ip: str, port: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((ip, port))
        return sock


    def ensure_columns(self, conn: sqlite3.Connection, table: str, columns: Iterable[tuple[str, str]]) -> None:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, sql_type in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
        conn.commit()


    def init_db(self, db_path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path)

        db_types = {"f": "REAL", "i": "INTEGER", "I": "INTEGER", "H": "INTEGER", "B": "INTEGER"}
        column_defs = [
            "id INTEGER PRIMARY KEY AUTOINCREMENT",
            "timestamp_utc TEXT NOT NULL",
            "packet_length INTEGER NOT NULL",
            "packet_schema TEXT NOT NULL",
            "raw BLOB NOT NULL",
        ]
        for name, fmt in self.FORZA_FIELDS:
            column_defs.append(f"{name} {db_types[fmt]} NOT NULL")

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS telemetry_samples (
                {", ".join(column_defs)}
            );
            """
        )

        self.ensure_columns(
            conn,
            "telemetry_samples",
            [
                ("timestamp_utc", "TEXT"),
                ("packet_length", "INTEGER"),
                ("packet_schema", "TEXT"),
                ("raw", "BLOB")
            ]
            + [(name, db_types[fmt]) for name, fmt in self.FORZA_FIELDS]
        )
        return conn


    def parse_telemetry(self, data: bytes) -> tuple[str, dict] | None:
        for schema_name, fields, size in self.SCHEMAS:
            if len(data) != size:
                continue
            fmt = "<" + "".join(fmt for _, fmt in fields)
            values = struct.unpack_from(fmt, data, 0)
            return schema_name, {name: value for (name, _), value in zip(fields, values)}
        return None


    def insert_sample(
        self, conn: sqlite3.Connection, timestamp_utc: str, raw: bytes, schema_name: str, parsed: dict
    ) -> None:
        columns = ["timestamp_utc", "packet_length", "packet_schema", "raw"] + [name for name, _ in self.FORZA_FIELDS]
        placeholders = ", ".join(["?"] * len(columns))
        values = [
            timestamp_utc,
            len(raw),
            schema_name,
            sqlite3.Binary(raw)]
        
        for name, _ in self.FORZA_FIELDS:
            values.append(parsed.get(name, 0))

        conn.execute(f"INSERT INTO telemetry_samples ({', '.join(columns)}) VALUES ({placeholders})",values,)
        conn.commit()


class Preprocessing():
    def __init__(self, CURRENT_LINE, OPTIMAL_LINE):
        self.DATABASE_OPTIMAL = f"sqlite:///../{OPTIMAL_LINE}"
        self.DATABASE_CURRENT = f"sqlite:///../{CURRENT_LINE}"
        self.OUTPUT = '../data/output.txt'
        self.MD = '../data/output.md'
        self.PROMPTS = '../prompts/prompts.txt'
        self.TRACK_2_SEGMENT_POINTS = [(610, 2485), (635, 2780), (525, 2720), (880, 2790)]
        self.QUERY = """SELECT timestamp_utc AS timestamp, acceleration_x, acceleration_y, acceleration_z, yaw, position_x, position_y, position_z, speed, lap_number
        FROM telemetry_samples
        WHERE distance_traveled != 0
        ORDER BY id"""
    
    def run(self):
        df_fast = self.preprocess(pd.read_sql(self.QUERY, self.DATABASE_OPTIMAL))
        df_slow = self.preprocess(pd.read_sql(self.QUERY, self.DATABASE_CURRENT))
        df_slow["segment"] = self.segment_labels(self.TRACK_2_SEGMENT_POINTS, df_slow)
        df_slow = self.match_lines_by_euclid(df_slow, df_fast)
        df_slow = df_slow.rename(
            columns={
                'timestamp': 'timestamp in s',
                'acceleration_x': 'acceleration_x in m/s²',
                'acceleration_y': 'acceleration_y in m/s²',
                'acceleration_z': 'acceleration_z in m/s²',
                'yaw': 'yaw in degrees',
                'position_x': 'position_x in m',
                'position_y': 'position_y in m',
                'position_z': 'position_z in m',
                'speed': 'speed in km/h',
            }
        )

        # Write as markdown
        markdown = df_slow.to_markdown()
        with open(self.MD, 'w', encoding="utf-8") as f:
            f.write(markdown)
        
        # Write output.txt
        # records = df_slow.to_dict(orient="records")
        # text = str(records)
        # for char in '[]{}()':
        #     text = text.replace(char, '')
            
        # text = text.replace('timestamp', '\ntimestamp')

        # with open(self.OUTPUT, "w", encoding="utf-8") as f:
        #     f.write(text)
    
    
    def preprocess(self, df):
        df['timestamp'] = pd.to_datetime(df["timestamp"], utc=True).astype("int64") // 10**6
        df['timestamp'] = (df['timestamp'] - min(df['timestamp'])) / 1000

        df['position_x'] = df['position_x'].astype(int)
        df['position_y'] = df['position_y'].astype(int)
        df['position_z'] = df['position_z'].astype(int)

        df['acceleration_x'] = df['acceleration_x'].astype(int)
        df['acceleration_y'] = df['acceleration_y'].astype(int)
        df['acceleration_z'] = df['acceleration_z'].astype(int)
        df['yaw'] = df['yaw'].round(decimals=2)
        df['speed'] = (df['speed'] * 3.6).astype(int)
        
        return df
    

    def get_index_nearest_position(self, point: tuple, positions_x: np.ndarray, positions_z: np.ndarray) -> int:
        smallest_distance = np.inf
        index = 0
        for i in range(positions_x.shape[0]):
            distance = np.sqrt(((point[0] - positions_x[i])**2 + (point[1] - positions_z[i])**2))
            if distance < smallest_distance:
                smallest_distance = distance
                index = i
        return index


    def segment_labels(self, points: list, track: DataFrame):
        labels_all: list[int] = []
        for lap in track["lap_number"].unique():
            lap_df = track[track["lap_number"] == lap]
            n = len(lap_df)
            idxs = [
                int(self.get_index_nearest_position(
                    p,
                    lap_df["position_x"].to_numpy(),
                    lap_df["position_z"].to_numpy()
                ))
                for p in points
            ]
            idxs = sorted({max(0, min(n, i)) for i in idxs})
            lap_labels = np.empty(n, dtype=int)
            start = 0
            seg = 0
            for cut in idxs:
                if cut <= start:
                    continue
                lap_labels[start:cut] = seg
                seg += 1
                start = cut
            lap_labels[start:] = seg
            labels_all.extend(lap_labels.tolist())
        return labels_all
    

    def match_lines_by_euclid(self, df_slow, df_fast):
        original_dtypes = df_slow.dtypes.copy()
        # Convert all columns to object dtype to allow storing tuples
        for col in df_slow.columns:
            df_slow[col] = df_slow[col].astype(object)
        
        #for line in df_slow:
        for _, line in df_slow.iterrows():
            # find line in optimal_df with minimal euclidian in x and z, with distance in y < 2
            optimal_line = self.find_optimal_line(df_fast, line)

            # write all optimal values into df_slow with the new values being second in a tuple, e.g. "timestamp": 0.0 -> "timestamp": [0.0,0.0]
            for column in df_slow.columns:
                if(column != "segment" and column != "lap_number"):
                    optimal_val = optimal_line[column]
                    slow_val = df_slow.at[line.name, column]
                    
                    # Convert based on original dtype
                    if pd.api.types.is_integer_dtype(original_dtypes[column]):
                        slow_val = int(slow_val)
                        optimal_val = int(optimal_val)
                    elif pd.api.types.is_float_dtype(original_dtypes[column]):
                        slow_val = float(slow_val)
                        optimal_val = float(optimal_val)
                    
                    df_slow.at[line.name, column] = (slow_val, optimal_val)
            # cut off all lines before found line in optimal df without 
            df_fast = df_fast[df_fast.index > optimal_line.name]
        return df_slow


    def find_optimal_line(self, df_fast, line):
        min_distance = sys.maxsize
        optimal_line = None

        for _, fast_line in df_fast.iterrows():
            # Skip certain segments on the first lap
            if(line['lap_number'] == '0' and line['segment'] == 0):
                if(fast_line['lap_number'] != line['lap_number'] and fast_line['segment'] != line['segment']):
                    continue
            
            # Now minimmize euclidian distance in x and z while holding y-constraint
            if abs(fast_line['position_y'] - line['position_y']) < 2:
                distance = ((fast_line['position_x'] - line['position_x']) ** 2 + 
                            (fast_line['position_z'] - line['position_z']) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    optimal_line = fast_line

        return optimal_line


class Model():
    def __init__(self):
        self.OUTPUT_MD = '../data/output.md'
        self.PROMPTS = '../prompts/prompts.txt'
        self.NON_PERSISTENT_PROMPTS = '../prompts/non_persistent_prompts.txt'
        self.SYSTEM_PROMPT = """
        Role:
        Experienced racing driver coach.

        Task:
        Analyze driving data and give short, precise feedback.
        You receive a table with attributes.
        So that you can perform your analysis, in addition to the player’s driving data you receive an optimally driven reference lap for comparison.
        Each attribute in the table has two values: first value = driver, second value = reference.

        Notes:
        Take into account the units of the respective attributes. They are specified in the table.
        Keep in mind that the user neither knows the optimal driving line nor values that refer to the x, y, and z axes.
        They only know throttle, brake, steering, speed, position on the track, and timestamp.
        The user cannot do much with concrete timestamps, but rather thinks in sections of the track.

        Examples:
        'You drove this segment without errors; however, in the preceding segment you were too slow and could no longer reach a sufficient top speed.',
        'Your line was too wide on the outside, which led to a longer path and less grip. Try turning in earlier to find a better racing line.',
        'You have to take the corner at 110 instead of 160 km/h in order not to crash.',
        'You drove this section almost perfectly, keep it up!'

        Negative examples:
        'The strongly negative acceleration_x (-34 vs. reference 0) together with a high Z value (-77 vs. -20) indicates excessive braking and understeer, causing the speed to drop to 96 km/h instead of the optimal 228 km/h.',
        'From about 9 s onward you reach lateral accelerations of up to +20 m/s², while the reference lap only allows 0–5 m/s² – this creates strong under-/oversteer. Later (from 21 s onward) you brake too late, causing the speed to drop abruptly from 198 → 96 km/h and the vehicle to deviate from the ideal line at yaw values around –0.5 rad.' 
        'In the first minutes, eexcessive negative acceleration_x and incorrectly dosed yaw rotation dominate, which leads to repeated slowing; from about 68 s onward the acceleration is chosen positively and the speed can increase again. The goal is to reduce braking so that the vehicle can take the corner with a similar speed as in the reference value (e.g. 173 km/h at 70.9 s).'
        """
        self.USER_PROMPT = ""

        self.print_token_length()
        # --- Read markdown file ---
        with open(self.OUTPUT_MD, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]

        # --- Split header and rows ---
        header = lines[0]
        separator = lines[1]
        rows = lines[2:]

        # Delete existing non-persistent prompts log and create new empty file
        pathlib.Path(self.NON_PERSISTENT_PROMPTS).unlink(missing_ok=True)
        pathlib.Path(self.NON_PERSISTENT_PROMPTS).parent.mkdir(parents=True, exist_ok=True)  # sicherstellen, dass Ordner existiert
        pathlib.Path(self.NON_PERSISTENT_PROMPTS).touch()  # erstellt leere Datei

        # --- Process per lap / segment ---
        for lap, segment in product([0], [0, 1, 2, 3, 4]):
            print("Lap:" + str(lap) + ", Segment:" + str(segment))
            segment_rows = [r for r in rows if self.get_segment_md(r) == segment and self.get_lap_md(r) == lap]

            if not segment_rows:
                continue
            
            md_block = "\n".join([header, separator, *segment_rows])
            print(md_block)

            segment_info = self.get_segment_information(segment_rows)

            # --- LLM call ---
            resp = ollama.chat(
                think=True,
                model="glm-4.7-flash:q4_K_M",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.USER_PROMPT + f"\n```markdown\n{md_block}\n```" + f"\n\nSegment Summary:\n{segment_info}"},
                ],
            )

            # Logging
            data_tokens = self.count_tokens(md_block, model_name="gpt-4")
            print(f"Token-Anzahl User {data_tokens}")
            timestamps = [self.get_timestamp_md(r) for r in segment_rows]
            self.log_response(timestamps, lap, segment, self.SYSTEM_PROMPT, self.USER_PROMPT, resp, md_block, segment_info)
    
    
    def count_tokens(self, prompt: str, model_name: str = "gpt-4") -> int:
        """
        Zählt die Tokens für einen Prompt für das angegebene Modell.
        """
        encoding = tiktoken.encoding_for_model(model_name)
        tokens = encoding.encode(prompt)
        return len(tokens)
    

    def print_token_length(self):
        text = []
        with open('../data/output.md', "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line != '\'':
                    text.append(line + '\n')

        sys_tokens = self.count_tokens(self.SYSTEM_PROMPT, model_name="gpt-4")
        print(f"Token-Anzahl System: {sys_tokens}")
        usr_tokens = self.count_tokens(self.USER_PROMPT, model_name="gpt-4")
        print(f"Token-Anzahl User {usr_tokens}")
    
    
    def log_response(self, timestamps, lap, segment, system_prompt, user_prompt, resp, md, segment_info):
        # -- Logging ---
        if(segment == 0):
            self.log_round(system_prompt, user_prompt)
        self.log_segment_response(timestamps, lap, segment, resp, md, segment_info)


    def log_round(self, system_prompt, user_prompt):
        # --- Logging ---
        pathlib.Path("prompts").mkdir(parents=True, exist_ok=True)
        with open(self.PROMPTS, "a", encoding="utf-8") as file:
            file.writelines([
                "XXXX" * 80 + "\n",
                "System Prompt: " + system_prompt + "\n",
                "User Prompt: " + user_prompt + "\n",
            ])
        with open(self.NON_PERSISTENT_PROMPTS, "a", encoding="utf-8") as file:
            file.writelines([
                "XXXX" * 80 + "\n",
                "System Prompt: " + system_prompt + "\n",
                "User Prompt: " + user_prompt + "\n",
            ])


    def log_segment_response(self, timestamps, lap, segment, resp, md, segment_info):
        min_timestamp = min(timestamps)
        max_timestamp = max(timestamps)

        pathlib.Path("prompts").mkdir(parents=True, exist_ok=True)
        with open(self.PROMPTS, "a", encoding="utf-8") as file:
            file.writelines([
                "__" * 80 + "\n",
                f"Lap: {lap}, Segment: {segment}, Sequence: {min_timestamp} - {max_timestamp}\n",
                f"{segment_info}\n",
                "Response: " + resp["message"]["content"] + "\n\n",
            ])
        with open(self.NON_PERSISTENT_PROMPTS, "a", encoding="utf-8") as file:
            file.writelines([
                "__" * 10 + "\n",
                f"Lap: {lap}, Segment: {segment}, Sequence: {min_timestamp} - {max_timestamp}\n",
                f"{md}\n",
                f"{segment_info}\n",
                "Response: " + resp["message"]["content"] + "\n\n",
            ])


    def get_segment_md(self, md_row: str) -> int:
        return int(md_row.split("|")[-2].strip())


    def get_lap_md(self, md_row: str) -> int:
        return int(md_row.split("|")[-3].strip())


    def get_timestamp_md(self, md_row: str) -> float:
        ts_cell = md_row.split("|")[2].strip()
        return float(ts_cell.strip("()").split(",")[0])
    
    
    def get_segment_information(self, segment_rows)-> str:
        num = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"
        pair_re = re.compile(rf"\(\s*({num})\s*,\s*({num})\s*\)")

        # lists for user / optimal (reference)
        timestamps_user = []
        timestamps_opt = []
        speeds_user = []
        speeds_opt = []
        yaws_user = []
        yaws_opt = []

        for row in segment_rows:
            cells = [c.strip() for c in row.split("|")]

            # timestamp usually in column index 2
            if len(cells) > 2:
                m = pair_re.search(cells[2])
                if m:
                    timestamps_user.append(float(m.group(1)))
                    timestamps_opt.append(float(m.group(2)))

            # yaw usually in column index 6
            if len(cells) > 6:
                m = pair_re.search(cells[6])
                if m:
                    yaws_user.append(float(m.group(1)))
                    yaws_opt.append(float(m.group(2)))

            # speed usually in column index 10
            if len(cells) > 10:
                m = pair_re.search(cells[10])
                if m:
                    speeds_user.append(float(m.group(1)))
                    speeds_opt.append(float(m.group(2)))

        # --- Timestamp / duration ---
        if timestamps_user:
            start_user = min(timestamps_user)
            end_user = max(timestamps_user)
            duration_user = end_user - start_user
        else:
            start_user = end_user = duration_user = 0.0

        if timestamps_opt:
            start_opt = min(timestamps_opt)
            end_opt = max(timestamps_opt)
            duration_opt = end_opt - start_opt
        else:
            start_opt = end_opt = duration_opt = 0.0

        # --- Speed stats ---
        max_speed_user = max(speeds_user) if speeds_user else 0.0
        min_speed_user = min(speeds_user) if speeds_user else 0.0
        max_speed_opt = max(speeds_opt) if speeds_opt else 0.0
        min_speed_opt = min(speeds_opt) if speeds_opt else 0.0

        # --- Yaw stats (use absolute values for extremes) ---
        max_yaw_user = max((abs(v) for v in yaws_user), default=0.0)
        max_yaw_opt = max((abs(v) for v in yaws_opt), default=0.0)

        # --- Build summary ---
        ret = (
            f"Time — user: {duration_user:.3f}s, "
            f"ref: {duration_opt:.3f}s (Δ {duration_user - duration_opt:+.3f}s). "
        )

        if speeds_user or speeds_opt:
            ret += (
                f"Max speed — user: {max_speed_user:.1f} km/h, ref: {max_speed_opt:.1f} km/h; "
                f"Min speed — user: {min_speed_user:.1f} km/h, ref: {min_speed_opt:.1f} km/h. "
            )
        else:
            ret += "No speed data. "

        if yaws_user or yaws_opt:
            ret += f"Max yaw (abs) — user: {max_yaw_user:.3f}°, ref: {max_yaw_opt:.3f}°."
        else:
            ret += "No yaw data."

        print(ret)
        return ret

