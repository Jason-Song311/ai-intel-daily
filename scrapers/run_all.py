"""Run every source, tolerate per-source failure, emit one dated bundle."""

import argparse
import datetime as dt
import json
import pathlib
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fetch_civitai  # noqa: E402
import fetch_hn  # noqa: E402
import fetch_ph  # noqa: E402
from common import write_json  # noqa: E402

SOURCES = ("hn", "ph", "civitai")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE.parent / "data"))
    parser.add_argument("--only", default=",".join(SOURCES))
    parser.add_argument("--date", default=None)
    args = parser.parse_args()

    day = args.date or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    wanted = [name.strip() for name in args.only.split(",") if name.strip()]
    out_dir = pathlib.Path(args.out)
    bundle = {"date": day, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "sources_ok": [], "sources_error": {}, "notes": {}, "data": {}}

    if "hn" in wanted:
        try:
            bundle["data"]["show_hn"] = fetch_hn.fetch()
            bundle["sources_ok"].append("hn")
        except Exception as exc:  # noqa: BLE001
            bundle["sources_error"]["hn"] = str(exc)
            bundle["notes"].setdefault("hn", []).append(traceback.format_exc()[-1200:])

    if "ph" in wanted:
        try:
            items, notes = fetch_ph.fetch()
            bundle["data"]["product_hunt"] = items
            bundle["notes"]["ph"] = notes
            bundle["sources_ok"].append("ph")
        except Exception as exc:  # noqa: BLE001
            bundle["sources_error"]["ph"] = str(exc)
            bundle["notes"].setdefault("ph", []).append(traceback.format_exc()[-1200:])

    if "civitai" in wanted:
        try:
            items, notes = fetch_civitai.fetch(
                keep=15,
                download=6,
                image_dir=out_dir.parent / "civitai" / "images" / day,
            )
            bundle["data"]["civitai_flux"] = items
            bundle["notes"]["civitai"] = notes
            if items:
                bundle["sources_ok"].append("civitai")
            else:
                bundle["sources_error"]["civitai"] = "api responded but no items matched"
        except Exception as exc:  # noqa: BLE001
            bundle["sources_error"]["civitai"] = str(exc)
            bundle["notes"].setdefault("civitai", []).append(traceback.format_exc()[-1200:])

    path = write_json(out_dir / f"{day}.json", bundle)
    latest = out_dir / "latest.json"
    latest.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved: {path} ({path.stat().st_size} bytes)")
    print(f"sources ok: {bundle['sources_ok']}")
    print(f"sources error: {bundle['sources_error']}")
    for name, notes in bundle["notes"].items():
        for note in notes:
            print(f"[{name}] {str(note)[:200]}")
    for key, rows in bundle["data"].items():
        print(f"  {key}: {len(rows)} items")


if __name__ == "__main__":
    main()
