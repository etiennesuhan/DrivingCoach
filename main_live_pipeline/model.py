import pathlib
import ollama
import re
import threading
import json
import ast

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent


class Model():
    _log_lock = threading.Lock()
    _non_persistent_reset = False

    def __init__(self):
        self.MODEL_NAME = "llama3.1"
        self.OUTPUT_MD = (REPO_ROOT / "data/output.md").as_posix()
        self.PROMPTS = (REPO_ROOT / "prompts/prompts.txt").as_posix()
        self.NON_PERSISTENT_PROMPTS = (REPO_ROOT / "prompts/non_persistent_prompts.txt").as_posix()
        self.SYSTEM_PROMPT = """
        Rolle: Erfahrener Rennfahrer-Coach.

        Aufgabe:
        Wandle tabellarische Telemetrie in extrem kurzes Coaching um.
        Jedes Attribut hat zwei Werte: erster Wert Fahrer, zweiter Wert Referenz.

        Selbst-augmentiert (intern, NICHT ausgeben):
        1) Bestimme intern die wichtigste, sicher belegte Verbesserung (Fahrer vs. Referenz).
        2) Formuliere daraus genau eine klare Handlungsanweisung.

        Ausgabe:
        Gib exakt ein JSON-Objekt aus:
        {
        "status": true|false,
        "message": "..."
        }

        Regeln:
        - Gib nur JSON aus, keinen weiteren Text.
        - status nur true/false (JSON-Boolean).
        - Keine Markdown-Backticks, keine Erklärung außerhalb des JSON.
        - message immer Deutsch (de-DE), Du-Form, Imperativ.
        - Nur Verbesserungen (keine Analyse, kein Vergleichstext).
        - Keine Telemetrie-Feldnamen, keine Zeitstempel.

        LÄNGE (SEHR WICHTIG):
        - message maximal 10 Wörter (Wörter = durch Leerzeichen getrennt).
        - Genau 1 Satz. Keine zweite Anweisung.
        - Keine Füllwörter (z. B. "versuche", "einfach", "wirklich").

        SCHREIBWEISE (SEHR WICHTIG FÜR TTS):
        - Verwende echte Umlaute (ä, ö, ü, Ä, Ö, Ü) und ß, wenn passend.
        - Schreibe Umlaute niemals als zwei Buchstaben.
        - Ersetze ß nicht durch Doppel-s.
        - Prüfe vor der Ausgabe: Wenn die message nicht natürliches Deutsch ist, korrigiere sie intern.

        WORTWAHL:
        - VERBOTEN: Trigger, Stick, Controller, progressiv, "ruhiger", "flüssiger", "Bogen".
        - Erlaubt (Beispiele, Stil genau so kurz halten):
        * "Früher bremsen."
        * "Bremse früher lösen."
        * "Sanfter bremsen."
        * "Später ans Gas."
        * "Gas sanfter aufbauen."
        * "Früher einlenken."
        * "Später einlenken."
        * "Weniger nachlenken."
        * "Weiter innen fahren." (nur wenn wirklich belegbar)

        STATUS:
        - status=true nur, wenn mindestens eine klare Verbesserung nötig ist.
        - Wenn status=false: message = "Passt."
        """

        self.USER_PROMPT = (
        "Antworte ausschließlich auf Deutsch (de-DE). "
        "Antwortformat strikt: {\"status\": true|false, \"message\": \"...\"}. "
        "Wichtig: Verwende echte Umlaute und ß, keine Umschrift mit zwei Buchstaben."
        )
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
        header_cells = self._parse_header_cells(header)
        idx_map = {name: idx for idx, name in enumerate(header_cells)}
        segment_rows = [
            r for r in rows
            if self.get_segment_md(r, idx_map=idx_map) == segment
            and self.get_lap_md(r, idx_map=idx_map) == lap
        ]
        if not segment_rows:
            return
        
        md_block = "\n".join([header, separator, *segment_rows])
        segment_info = self.get_segment_information(segment_rows, idx_map=idx_map)

        # --- LLM call ---
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_PROMPT + f"\n```markdown\n{md_block}\n```" + f"\n\nSegment Summary:\n{segment_info}"},
        ]
        resp = self._chat_model(messages)
        raw_content = self._safe_get_response_content(resp)
        normalized_content = self._normalize_response_content(raw_content)
        if normalized_content is not None:
            self._safe_set_response_content(resp, normalized_content)
        timestamps = [self.get_timestamp_md(r, idx_map=idx_map) for r in segment_rows]
        self.log_response(timestamps, lap, segment, self.SYSTEM_PROMPT, self.USER_PROMPT, resp, md_block, segment_info)
        return normalized_content or raw_content

    def _chat_model(self, messages: list[dict]) -> dict:
        # Prefer strict JSON mode. If unavailable, fallback to normal chat call.
        try:
            return ollama.chat(
                model=self.MODEL_NAME,
                format="json",
                messages=messages,
            )
        except Exception:
            return ollama.chat(
                model=self.MODEL_NAME,
                messages=messages,
            )

    def _safe_get_response_content(self, resp: dict | None) -> str | None:
        try:
            return resp["message"]["content"]
        except Exception:
            return None

    def _safe_set_response_content(self, resp: dict | None, content: str) -> None:
        try:
            if isinstance(resp, dict) and isinstance(resp.get("message"), dict):
                resp["message"]["content"] = content
        except Exception:
            pass

    def _normalize_response_content(self, response_text: str | None) -> str | None:
        payload = self._parse_response_payload(response_text)
        if payload is None:
            return response_text
        return json.dumps(payload, ensure_ascii=False)

    def _parse_response_payload(self, response_text: str | None) -> dict | None:
        if not response_text or not response_text.strip():
            return None
        text = response_text.strip()
        candidates = [text]
        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        candidates.extend(chunk.strip() for chunk in fenced if chunk and chunk.strip())
        brace_candidate = self._extract_first_brace_block(text)
        if brace_candidate:
            candidates.append(brace_candidate)

        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(candidate)
                except Exception:
                    continue
                if not isinstance(parsed, dict):
                    continue
                status = self._normalize_status(parsed.get("status"))
                message = parsed.get("message")
                message = "" if message is None else str(message).strip()
                if status is None:
                    status = False
                return {"status": bool(status), "message": message}
        return None

    def _normalize_status(self, value) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("true", "1", "yes", "y", "ja", "wahr"):
                return True
            if normalized in ("false", "0", "no", "n", "nein", "falsch"):
                return False
        return None

    def _extract_first_brace_block(self, text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1].strip()
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


    def _parse_header_cells(self, header: str) -> list[str]:
        return [c.strip() for c in header.split("|")[1:-1]]


    def _get_cell(self, md_row: str, idx_map: dict[str, int], name: str) -> str | None:
        idx = idx_map.get(name)
        if idx is None:
            return None
        cells = [c.strip() for c in md_row.split("|")[1:-1]]
        if idx >= len(cells):
            return None
        return cells[idx]


    def get_segment_md(self, md_row: str, idx_map: dict[str, int] | None = None) -> int:
        if idx_map is not None:
            cell = self._get_cell(md_row, idx_map, "segment")
            return int(cell) if cell is not None else -1
        return int(md_row.split("|")[-2].strip())


    def get_lap_md(self, md_row: str, idx_map: dict[str, int] | None = None) -> int:
        if idx_map is not None:
            cell = self._get_cell(md_row, idx_map, "lap_number")
            return int(cell) if cell is not None else -1
        return int(md_row.split("|")[-3].strip())


    def get_timestamp_md(self, md_row: str, idx_map: dict[str, int] | None = None) -> float:
        if idx_map is not None:
            ts_cell = self._get_cell(md_row, idx_map, "timestamp in s") or ""
        else:
            ts_cell = md_row.split("|")[2].strip()
        return float(ts_cell.strip("()").split(",")[0])
    
    
    def get_segment_information(self, segment_rows, idx_map: dict[str, int] | None = None) -> str:
        num = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"
        pair_re = re.compile(rf"\(\s*({num})\s*,\s*({num})\s*\)")
        timestamps_user = []
        timestamps_opt = []
        speeds_user = []
        speeds_opt = []
        yaws_user = []
        yaws_opt = []

        for row in segment_rows:
            if idx_map is not None:
                ts_cell = self._get_cell(row, idx_map, "timestamp in s") or ""
                yaw_cell = self._get_cell(row, idx_map, "yaw in degrees") or ""
                speed_cell = self._get_cell(row, idx_map, "speed in km/h") or ""
            else:
                cells = [c.strip() for c in row.split("|")]
                ts_cell = cells[2] if len(cells) > 2 else ""
                yaw_cell = cells[6] if len(cells) > 6 else ""
                speed_cell = cells[10] if len(cells) > 10 else ""

            # timestamp usually in column index 2
            m = pair_re.search(ts_cell)
            if m:
                timestamps_user.append(float(m.group(1)))
                timestamps_opt.append(float(m.group(2)))

            # yaw usually in column index 6
            m = pair_re.search(yaw_cell)
            if m:
                yaws_user.append(float(m.group(1)))
                yaws_opt.append(float(m.group(2)))

            # speed usually in column index 10
            m = pair_re.search(speed_cell)
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

        return ret

