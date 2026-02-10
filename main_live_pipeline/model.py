import pathlib
import ollama
import re
import threading

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent


class Model():
    _log_lock = threading.Lock()
    _non_persistent_reset = False

    def __init__(self):
        # self.MODEL_NAME="glm-4.7-flash:q4_K_M",
        # self.MODEL_NAME="nemotron-3-nano:30b",
        self.MODEL_NAME = "llama3.1"
        self.OUTPUT_MD = (REPO_ROOT / "data/output.md").as_posix()
        self.PROMPTS = (REPO_ROOT / "prompts/prompts.txt").as_posix()
        self.NON_PERSISTENT_PROMPTS = (REPO_ROOT / "prompts/non_persistent_prompts.txt").as_posix()
        self.SYSTEM_PROMPT = """
        Role:
        Experiences racing driver coach.

        Task:
        Transform the tabular telemetry data into clear,
        readable coaching feedback describing how this track segment was driven compared to an optimally driven reference lap. Each attribute contains two values: First value, driver; second value reference.
        Explain how the segment unfolded in driving terms and identify errors if present.

        At the end, output a json object with the following structure, where status signals wether the message has to be read out (has only to be read out when significant improvements are necessary; status can be True or False) and the message states the improvement text:
        {
            'status': ...,
            'message': ...,
        }
        Only output the json!

        Notes:
        - Consider the units, but never mention them explicitly.
        - Translate telemetry only into: throttle, braking, steering, speed, track position.
        - Do not mention axes, telemetry terms, or timestamps.
        - Think in track sections, not time.
        - Focus on cause and effect.
        - Do not speculate beyond the data.

        Examples:
        Good throttle use, speed carried well.
        Clean line, no time lost here.
        Brake earlier to stabilize the car.
        Turn in earlier for better exit.
        Strong exit, close to optimal.
        Over-slowed entry cost exit speed.

        Negative examples:
        - The strongly negative acceleration_x (-34 vs. reference 0) together with a high Z value…
        - From about 9 s onward you reach lateral accelerations of up to +20 m/s²…
        - Excessive negative acceleration_x and incorrectly dosed yaw rotation dominate…
        """
        self.USER_PROMPT = ""
        self._reset_non_persistent_once()
    
    
    def _reset_non_persistent_once(self):
        if Model._non_persistent_reset:
            return
        pathlib.Path(self.NON_PERSISTENT_PROMPTS).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.NON_PERSISTENT_PROMPTS).unlink(missing_ok=True)
        pathlib.Path(self.NON_PERSISTENT_PROMPTS).touch()
        Model._non_persistent_reset = True

    def warmup_model(self):
        ollama.chat(
            model=self.MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
        )
        
        
    def run(self, lap, segment, md_text: str | None = None) -> str | None:
        # --- Read markdown file ---
        if md_text is None:
            with open(self.OUTPUT_MD, "r", encoding="utf-8") as f:
                lines = [line.rstrip("\n") for line in f if line.strip()]
        else:
            lines = [line.rstrip("\n") for line in md_text.splitlines() if line.strip()]

        if len(lines) < 3:
            print("No markdown data available; skipping model run.")
            return

        # --- Split header and rows ---
        header = lines[0]
        separator = lines[1]
        rows = lines[2:]
        segment_rows = [r for r in rows if self.get_segment_md(r) == segment and self.get_lap_md(r) == lap]
        if not segment_rows:
            return
        
        md_block = "\n".join([header, separator, *segment_rows])
        segment_info = self.get_segment_information(segment_rows)

        # --- LLM call ---
        resp = ollama.chat(
            # think=True,
            model=self.MODEL_NAME,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.USER_PROMPT + f"\n```markdown\n{md_block}\n```" + f"\n\nSegment Summary:\n{segment_info}"},
            ],
        )
        timestamps = [self.get_timestamp_md(r) for r in segment_rows]
        self.log_response(timestamps, lap, segment, self.SYSTEM_PROMPT, self.USER_PROMPT, resp, md_block, segment_info)
        try:
            return resp["message"]["content"]
        except Exception:
            return None
    
    
    def log_response(self, timestamps, lap, segment, system_prompt, user_prompt, resp, md, segment_info):
        with self._log_lock:
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
