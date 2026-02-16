import argparse

from listener import Listener
from model import Model

CURRENT_LINE = "data/test.db"


def main() -> None:
    parser = argparse.ArgumentParser(description="FH5 Live Pipeline")
    parser.add_argument("--db", default=CURRENT_LINE, help="Pfad zur SQLite-DB")
    parser.add_argument("--debug", action="store_true", help="Debug-Ausgaben aktivieren")
    parser.add_argument("--api", action="store_true", help="REST-API aktivieren")
    parser.add_argument("--api-host", default="127.0.0.1", help="API Host")
    parser.add_argument("--api-port", type=int, default=8000, help="API Port")
    args = parser.parse_args()

    print(f"Lade Modell: {Model().MODEL_NAME}")
    Model().warmup_model()

    listener = Listener(args.db, debug=args.debug)
    if args.api:
        try:
            from api import start_api
        except Exception as exc:
            raise RuntimeError(f"API konnte nicht gestartet werden: {exc}") from exc
        start_api(listener, host=args.api_host, port=args.api_port)

    listener.run()


if __name__ == "__main__":
    main()
