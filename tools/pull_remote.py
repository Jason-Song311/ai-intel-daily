"""Pull the daily bundle from the collector repo down to the local machine.

raw.githubusercontent.com and cdn.jsdelivr.net are unreachable from the
mainland network, but gh-proxy.com works, so mirror the raw file through it.
"""

import argparse
import json
import pathlib
import sys
import urllib.request

MIRRORS = (
    "https://gh-proxy.com/https://raw.githubusercontent.com/{repo}/{branch}/{path}",
    "https://ghproxy.net/https://raw.githubusercontent.com/{repo}/{branch}/{path}",
)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0.0.0 Safari/537.36"


def fetch(url, timeout=40):
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Jason-Song311/ai-intel-daily")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--path", default="data/latest.json")
    parser.add_argument("--out", default="work/data/latest.json")
    args = parser.parse_args()

    errors = []
    for template in MIRRORS:
        url = template.format(repo=args.repo, branch=args.branch, path=args.path)
        try:
            payload = fetch(url)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url} :: {exc}")
            continue
        target = pathlib.Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        data = json.loads(payload.decode("utf-8"))
        print(f"OK {target} ({len(payload)} bytes) via {url}")
        print("sources ok:", data.get("sources_ok"), "| error:", list(data.get("sources_error") or {}))
        for key, rows in (data.get("data") or {}).items():
            print(f"  {key}: {len(rows)} items")
        return
    print("all mirrors failed:")
    for line in errors:
        print(" ", line)
    sys.exit(1)


if __name__ == "__main__":
    main()
