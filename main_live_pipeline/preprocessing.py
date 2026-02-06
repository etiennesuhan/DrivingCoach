import sys
import numpy as np
import pandas as pd
from pathlib import Path
from pandas import DataFrame
import sqlite3


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


class Preprocessing():
    def __init__(self, CURRENT_LINE, OPTIMAL_LINE):
        CURRENT_LINE = "data/" + CURRENT_LINE
        OPTIMAL_LINE = "data/" + OPTIMAL_LINE
        current_path = resolve_repo_path(CURRENT_LINE)
        optimal_path = resolve_repo_path(OPTIMAL_LINE)
        self.DATABASE_OPTIMAL = optimal_path.as_posix()
        self.DATABASE_CURRENT = current_path.as_posix()
        self.OUTPUT = (REPO_ROOT / "data/output.txt").as_posix()
        self.MD = (REPO_ROOT / "data/output.md").as_posix()
        self.PROMPTS = (REPO_ROOT / "prompts/prompts.txt").as_posix()
        self.TRACK_2_SEGMENT_POINTS = [(610, 2485), (635, 2780), (525, 2720), (880, 2790)]
        self.QUERY = """SELECT timestamp_utc AS timestamp, acceleration_x, acceleration_y, acceleration_z, yaw, position_x, position_y, position_z, speed, lap_number
        FROM telemetry_samples
        WHERE distance_traveled != 0
        ORDER BY id"""
    
    def run(self):
        conn_fast = sqlite3.connect(self.DATABASE_OPTIMAL)
        df_fast = self.preprocess(pd.read_sql(self.QUERY, conn_fast))
        
        conn_slow = sqlite3.connect(self.DATABASE_CURRENT)
        df_slow = self.preprocess(pd.read_sql(self.QUERY, conn_slow))
        
        df_slow = df_slow.iloc[::int(10), :]
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
        df_formatted = self.clean_decimals(df_slow)
        markdown = df_formatted.to_markdown()
        with open(self.MD, 'w', encoding="utf-8") as f:
            f.write(markdown)
        return markdown
    
    
    def preprocess(self, df):
        df['timestamp'] = pd.to_datetime(df["timestamp"], utc=True).astype("int64") // 10**6
        df['timestamp'] = (df['timestamp'] - min(df['timestamp'])) / 1000

        df['timestamp'] = df['timestamp'].round(decimals=3)
        
        df['position_x'] = df['position_x'].astype(int)
        df['position_y'] = df['position_y'].astype(int)
        df['position_z'] = df['position_z'].astype(int)

        df['acceleration_x'] = df['acceleration_x'].astype(int)
        df['acceleration_y'] = df['acceleration_y'].astype(int)
        df['acceleration_z'] = df['acceleration_z'].astype(int)
        df['yaw'] = df['yaw'].round(decimals=2)
        df['speed'] = (df['speed'] * 3.6).astype(int)
        
        return df
    
    def clean_decimals(self, df):
        df = df.copy()
    
        # Debug: Check input
        print("Before cleaning:")
        print(df["timestamp in s"].head())
        print(f"Type of first value: {type(df['timestamp in s'].iloc[0])}")
        
        df["timestamp in s"] = df["timestamp in s"].apply(
            lambda x: (round(x[0], 1), round(x[1], 1))
        )
        
        # Debug: Check output
        print("\nAfter cleaning:")
        print(df["timestamp in s"].head())
        print(f"Type of first value: {type(df['timestamp in s'].iloc[0])}")
        
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
        
        df_fast_working = df_fast.copy()
        
        #for line in df_slow:
        for idx, line in df_slow.iterrows():
            # find line in optimal_df with minimal euclidian in x and z, with distance in y < 2
            optimal_line = self.find_optimal_line(df_fast_working, line)

            # write all optimal values into df_slow with the new values being second in a tuple, e.g. "timestamp": 0.0 -> "timestamp": [0.0,0.0]
            for column in df_slow.columns:
                if column not in ["segment", "lap_number"]:
                    optimal_val = optimal_line[column]
                    slow_val = line[column]
                    
                    # Convert based on original dtype
                    if pd.api.types.is_integer_dtype(original_dtypes[column]):
                        slow_val = int(slow_val)
                        optimal_val = int(optimal_val)
                    elif pd.api.types.is_float_dtype(original_dtypes[column]):
                        slow_val = round(float(slow_val), 3)
                        optimal_val = round(float(optimal_val), 3)
                    
                    df_slow.at[idx, column] = (slow_val, optimal_val)
                    
            # only advance cutting of when we found a match
            if optimal_line is not None:
                df_fast_working = df_fast_working[df_fast_working.index > optimal_line.name]
        return df_slow


    def find_optimal_line(self, df_fast, line):
        min_distance = sys.maxsize
        optimal_line = None

        for _, fast_line in df_fast.iterrows():
            # Now minimmize euclidian distance in x and z while holding y-constraint
            if abs(fast_line['position_y'] - line['position_y']) < 2:
                distance = ((fast_line['position_x'] - line['position_x']) ** 2 + 
                            (fast_line['position_z'] - line['position_z']) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    optimal_line = fast_line
        if optimal_line is not None:
            return optimal_line

        # Fallback: ignore the y-constraint if nothing matched
        for _, fast_line in df_fast.iterrows():
            if(line['lap_number'] == '0' and line['segment'] == 0):
                if(fast_line['lap_number'] != line['lap_number'] and fast_line['segment'] != line['segment']):
                    continue
            distance = ((fast_line['position_x'] - line['position_x']) ** 2 + 
                        (fast_line['position_z'] - line['position_z']) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                optimal_line = fast_line
        return optimal_line
    
p = Preprocessing("test.db", "track2_good.db")

p.run()