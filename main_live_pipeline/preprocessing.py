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
    def __init__(self, database, lap_id, optimal_lap, segments):
        current_path = resolve_repo_path(database)
        self.database = current_path.as_posix()
        self.OUTPUT = (REPO_ROOT / "data/output.txt").as_posix()
        self.MD = (REPO_ROOT / "data/output.md").as_posix()
        self.PROMPTS = (REPO_ROOT / "prompts/prompts.txt").as_posix()
        self.lap_id = lap_id
        self.optimal_lap = optimal_lap
        self.segments = segments
    
    def run(self):
        conn = sqlite3.connect(self.database)
        df_slow = self.preprocess(pd.read_sql(self.query(self.lap_id), conn))
        df_slow = df_slow.iloc[::int(5), :]
        df_slow["segment"] = self.segment_labels(df_slow)
        if self.optimal_lap is not None:
            df_fast_raw = pd.read_sql(self.query(self.optimal_lap), conn)
            if not df_fast_raw.empty:
                df_fast = self.preprocess(df_fast_raw)
                df_slow = self.match_lines_by_euclid(df_slow, df_fast)
            else:
                self.optimal_lap = None
        if self.optimal_lap is None:
            cols = [c for c in df_slow.columns if c not in ("segment", "lap_number")]
            for c in cols:
                vals = df_slow[c].tolist()
                df_slow[c] = list(zip(vals, vals))
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
        df_slow["timestamp in s"] = df_slow["timestamp in s"].apply(lambda x: (round(x[0], 1), round(x[1], 1)))
        markdown = df_slow.to_markdown()
        with open(self.MD, 'w', encoding="utf-8") as f:
            f.write(markdown)
        return markdown
    
    def query(self, lap_id):
        return f"""SELECT timestamp_utc AS timestamp, acceleration_x, acceleration_y, acceleration_z, yaw, position_x, position_y, position_z, speed, lap_number
        FROM telemetry_samples
        WHERE lap_id = {lap_id}
        ORDER BY id"""

    def preprocess(self, df: DataFrame):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['timestamp'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
        
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


    def segment_labels(self, track: DataFrame):
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
                for p in self.segments
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
    
    def match_lines_by_euclid(
            self, df_slow: pd.DataFrame,
            df_fast: pd.DataFrame,
            exclude_cols=("segment", "lap_number"),
            keep_fast_index_col="fast_match_idx",
            chunk_size=204
        ):

        df_slow = df_slow.copy()
        df_fast = df_fast.reset_index(drop=True)

        slow_xz = df_slow[["position_x", "position_z"]].to_numpy(dtype=float)
        fast_xz = df_fast[["position_x", "position_z"]].to_numpy(dtype=float)

        idx = self.find_optimal_line(slow_xz, fast_xz, chunk_size=chunk_size)
        matched_fast = df_fast.iloc[idx].reset_index(drop=True)

        df_slow[keep_fast_index_col] = idx
        cols = [c for c in df_slow.columns if c not in exclude_cols and c != keep_fast_index_col]
        for c in cols:
            slow_vals = df_slow[c].to_numpy()
            fast_vals = matched_fast[c].to_numpy()
            df_slow[c] = list(zip(slow_vals.tolist(), fast_vals.tolist()))


        return df_slow
    
    def find_optimal_line(self, slow_xz: np.ndarray, fast_xz: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
        idx_out = np.empty((slow_xz.shape[0],), dtype=int)

        for start in range(0, slow_xz.shape[0], chunk_size):
            end = min(start + chunk_size, slow_xz.shape[0])
            chunk = slow_xz[start:end]
            d2 = ((chunk[:, None, 0] - fast_xz[None, :, 0]) ** 2 +
                (chunk[:, None, 1] - fast_xz[None, :, 1]) ** 2)
            idx_out[start:end] = d2.argmin(axis=1)

        return idx_out
