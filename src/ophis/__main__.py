import argparse
from pathlib import Path

DEFAULT_DATASET = Path("data/dataset/DDR")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ophis", description="Diabetic retinopathy toolkit")
    sub = parser.add_subparsers(dest="command")

    gui = sub.add_parser("gui", help="Launch the desktop viewer (default)")
    gui.add_argument("--data", type=Path, default=DEFAULT_DATASET, help="DDR dataset root")

    args = parser.parse_args(argv)

    if args.command in (None, "gui"):
        from ophis.gui.app import run

        return run(getattr(args, "data", DEFAULT_DATASET))

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
