import datetime
import socket
import sqlite3
import struct
import time
from typing import Iterable


UDP_IP = "0.0.0.0"  # lauscht auf allen Netzwerkinterfaces
UDP_PORT = 5300     # muss mit dem in FH5 eingestellten Port uebereinstimmen
DB_FILE = "telemetry.db"

# Zwei bekannte Dash-Formate:
# - fh5_dash_324: aktuelles FH5-Format (324 Byte, inkl. 3 zusaetzlicher Floats nach num_cylinders)
# - fh4_dash_312: aelteres FH4/FH5-Format (312 Byte)
FORZA_FIELDS_DASH_324: list[tuple[str, str]] = [
    ("is_race_on", "i"),
    ("timestamp_ms", "I"),
    ("engine_max_rpm", "f"),
    ("engine_idle_rpm", "f"),
    ("current_engine_rpm", "f"),
    ("acceleration_x", "f"),
    ("acceleration_y", "f"),
    ("acceleration_z", "f"),
    ("velocity_x", "f"),
    ("velocity_y", "f"),
    ("velocity_z", "f"),
    ("angular_velocity_x", "f"),
    ("angular_velocity_y", "f"),
    ("angular_velocity_z", "f"),
    ("yaw", "f"),
    ("pitch", "f"),
    ("roll", "f"),
    ("normalized_suspension_travel_fl", "f"),
    ("normalized_suspension_travel_fr", "f"),
    ("normalized_suspension_travel_rl", "f"),
    ("normalized_suspension_travel_rr", "f"),
    ("tire_slip_ratio_fl", "f"),
    ("tire_slip_ratio_fr", "f"),
    ("tire_slip_ratio_rl", "f"),
    ("tire_slip_ratio_rr", "f"),
    ("wheel_rotation_speed_fl", "f"),
    ("wheel_rotation_speed_fr", "f"),
    ("wheel_rotation_speed_rl", "f"),
    ("wheel_rotation_speed_rr", "f"),
    ("wheel_on_rumble_strip_fl", "f"),
    ("wheel_on_rumble_strip_fr", "f"),
    ("wheel_on_rumble_strip_rl", "f"),
    ("wheel_on_rumble_strip_rr", "f"),
    ("wheel_in_puddle_depth_fl", "f"),
    ("wheel_in_puddle_depth_fr", "f"),
    ("wheel_in_puddle_depth_rl", "f"),
    ("wheel_in_puddle_depth_rr", "f"),
    ("surface_rumble_fl", "f"),
    ("surface_rumble_fr", "f"),
    ("surface_rumble_rl", "f"),
    ("surface_rumble_rr", "f"),
    ("tire_slip_angle_fl", "f"),
    ("tire_slip_angle_fr", "f"),
    ("tire_slip_angle_rl", "f"),
    ("tire_slip_angle_rr", "f"),
    ("tire_combined_slip_fl", "f"),
    ("tire_combined_slip_fr", "f"),
    ("tire_combined_slip_rl", "f"),
    ("tire_combined_slip_rr", "f"),
    ("suspension_travel_meters_fl", "f"),
    ("suspension_travel_meters_fr", "f"),
    ("suspension_travel_meters_rl", "f"),
    ("suspension_travel_meters_rr", "f"),
    ("car_ordinal", "i"),
    ("car_class", "i"),
    ("car_performance_index", "i"),
    ("drivetrain_type", "i"),
    ("num_cylinders", "i"),
    ("extra_unknown_1", "f"),  # FH5 sendet hier 3 zusaetzliche Floats
    ("extra_unknown_2", "f"),
    ("extra_unknown_3", "f"),
    ("position_x", "f"),
    ("position_y", "f"),
    ("position_z", "f"),
    ("speed", "f"),
    ("power", "f"),
    ("torque", "f"),
    ("tire_temp_fl", "f"),
    ("tire_temp_fr", "f"),
    ("tire_temp_rl", "f"),
    ("tire_temp_rr", "f"),
    ("boost", "f"),
    ("fuel", "f"),
    ("distance_traveled", "f"),
    ("best_lap", "f"),
    ("last_lap", "f"),
    ("current_lap", "f"),
    ("current_race_time", "f"),
    ("lap_number", "H"),
    ("race_position", "H"),
    ("accel", "B"),
    ("brake", "B"),
    ("clutch", "B"),
    ("handbrake", "B"),
    ("gear", "B"),
    ("steer", "B"),
    ("normalized_driving_line", "B"),
    ("normalized_ai_brake_diff", "B"),
]

# Aelteres 312-Byte-Schema ohne die drei Extra-Floats.
FORZA_FIELDS_DASH_312 = [f for f in FORZA_FIELDS_DASH_324 if not f[0].startswith("extra_unknown_")]

SCHEMAS = [
    ("fh5_dash_324", FORZA_FIELDS_DASH_324, struct.calcsize("<" + "".join(fmt for _, fmt in FORZA_FIELDS_DASH_324))),
    ("fh4_dash_312", FORZA_FIELDS_DASH_312, struct.calcsize("<" + "".join(fmt for _, fmt in FORZA_FIELDS_DASH_312))),
]

# Superset fuer die Tabelle: wir nehmen das groesste (324-Byte) Schema.
BASE_FIELDS = FORZA_FIELDS_DASH_324
BASE_STRUCT_FORMAT = "<" + "".join(fmt for _, fmt in BASE_FIELDS)
EXPECTED_PACKET_SIZE = SCHEMAS[0][2]


def create_socket(ip: str, port: int) -> socket.socket:
    """Erzeuge und binde einen UDP-Socket fuer die Forza-Telemetrie."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    return sock


def ensure_columns(conn: sqlite3.Connection, table: str, columns: Iterable[tuple[str, str]]) -> None:
    """Stellt sicher, dass alle benoetigten Spalten existieren (fuer Upgrades alter DBs)."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, sql_type in columns:
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
    conn.commit()


def init_db(db_path: str) -> sqlite3.Connection:
    """Erzeuge/aktualisiere eine SQLite-Datenbank mit allen Forza-Feldern und Raw-Paket."""
    conn = sqlite3.connect(db_path)

    # Tabellenschema dynamisch aus der groessten Feldliste erzeugen.
    db_types = {"f": "REAL", "i": "INTEGER", "I": "INTEGER", "H": "INTEGER", "B": "INTEGER"}
    column_defs = [
        "id INTEGER PRIMARY KEY AUTOINCREMENT",
        "timestamp_utc TEXT NOT NULL",
        "packet_length INTEGER NOT NULL",
        "packet_schema TEXT NOT NULL",
        "raw BLOB NOT NULL",
    ]
    for name, fmt in BASE_FIELDS:
        column_defs.append(f"{name} {db_types[fmt]} NOT NULL")

    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS telemetry_samples (
            {", ".join(column_defs)}
        );
        """
    )

    # Falls bereits eine alte Tabelle existiert, fehlende Spalten sukzessive anlegen.
    ensure_columns(
        conn,
        "telemetry_samples",
        [
            ("timestamp_utc", "TEXT"),
            ("packet_length", "INTEGER"),
            ("packet_schema", "TEXT"),
            ("raw", "BLOB"),
        ]
        + [(name, db_types[fmt]) for name, fmt in BASE_FIELDS],
    )

    return conn


def parse_telemetry(data: bytes) -> tuple[str, dict] | None:
    """
    Liest alle bekannten Felder aus dem Forza-Telemetriepaket.
    Parst nur, wenn die Laenge exakt zum Schema passt (sonst None).
    """
    for schema_name, fields, size in SCHEMAS:
        if len(data) != size:
            continue
        fmt = "<" + "".join(fmt for _, fmt in fields)
        values = struct.unpack_from(fmt, data, 0)
        return schema_name, {name: value for (name, _), value in zip(fields, values)}
    return None


def insert_sample(
    conn: sqlite3.Connection, timestamp_utc: str, raw: bytes, schema_name: str, parsed: dict
) -> None:
    """Speichert einen kompletten Telemetrie-Datensatz inklusive Raw-BLOB."""
    columns = ["timestamp_utc", "packet_length", "packet_schema", "raw"] + [name for name, _ in BASE_FIELDS]
    placeholders = ", ".join(["?"] * len(columns))
    values = [
        timestamp_utc,
        len(raw),
        schema_name,
        sqlite3.Binary(raw),
    ]
    # Fuer Felder, die in einem kleineren Schema fehlen, Null/0 als Platzhalter.
    for name, _ in BASE_FIELDS:
        values.append(parsed.get(name, 0))

    conn.execute(
        f"INSERT INTO telemetry_samples ({', '.join(columns)}) VALUES ({placeholders})",
        values,
    )
    conn.commit()


def main() -> None:
    print(f"Erwartete Paketlaenge fh5_dash_324: {EXPECTED_PACKET_SIZE} Bytes")
    conn = init_db(DB_FILE)
    sock = create_socket(UDP_IP, UDP_PORT)
    print(f"Lausche auf UDP {UDP_IP}:{UDP_PORT} ...")
    print("Stelle sicher, dass in Forza Horizon 5 'Data Out / Telemetrie'")
    print("aktiviert ist und auf diese IP/Port zeigt.\n")

    last_print = 0.0

    try:
        while True:
            data, _ = sock.recvfrom(512)
            parsed_result = parse_telemetry(data)
            if not parsed_result:
                # Unbekannte Laenge: Hinweis ausgeben und Paket ueberspringen.
                print(f"Ignoriere Paket mit {len(data)} Bytes (unbekannte Laenge).")
                continue
            schema_name, parsed = parsed_result
            if parsed.get("engine_max_rpm", 0) == 0:
                # Ignoriere fruehe/ungenutzte Pakete ohne RPM-Daten.
                continue

            timestamp_utc = (
                datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
            )
            insert_sample(conn, timestamp_utc, data, schema_name, parsed)

            # Alle 0,5 s eine Kurzinfo fuer Sichtkontrolle.
            now = time.monotonic()
            if now - last_print >= 0.5:
                last_print = now
                speed_kmh = parsed["speed"] * 3.6  # Speed ist m/s im Forza-Paket.
                rpm = parsed["current_engine_rpm"]
                print(
                    f"{timestamp_utc} | {speed_kmh:6.1f} km/h | {rpm:6.0f} RPM | len={len(data)} | schema={schema_name}"
                )
    except KeyboardInterrupt:
        print("\nBeende Listener...")
    finally:
        conn.close()
        sock.close()


if __name__ == "__main__":
    main()
