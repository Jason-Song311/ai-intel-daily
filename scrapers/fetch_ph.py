"""Product Hunt 'Today' leaderboard scraper.

producthunt.com is unreachable from the mainland network, so this runs on the
GitHub Actions runner. It parses the Next.js payload embedded in the
server-rendered HTML, which means no API token is required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import (  # noqa: E402
    ai_hits,
    extract_json_objects,
    http_get,
    platform_smell,
    plain_text,
    today_pt,
)

MARKERS = ('"__typename":"Post"', '"__typename":"Product"')
TEMPLATES = (
    "https://www.producthunt.com/leaderboard/daily/{y}/{m}/{d}",
    "https://www.producthunt.com/",
)


def _collect(html, store):
    for marker in MARKERS:
        for obj in extract_json_objects(html, marker):
            slug = obj.get("slug")
            name = obj.get("name")
            if not isinstance(slug, str) or not isinstance(name, str) or not name:
                continue
            record = {
                "slug": slug,
                "name": name,
                "tagline": plain_text(obj.get("tagline")),
                "url": f"https://www.producthunt.com/posts/{slug}",
                "website": obj.get("website") or obj.get("url"),
                "votes": obj.get("votesCount") or 0,
                "comments": obj.get("commentsCount") or 0,
                "featured_at": obj.get("featuredAt") or obj.get("createdAt"),
                "topics": [
                    t.get("name") for t in (obj.get("topics") or []) if isinstance(t, dict)
                ],
            }
            current = store.get(slug)
            if current is None or record["votes"] >= current["votes"]:
                store[slug] = record
    return store


def fetch(max_items=60):
    day = today_pt()
    store = {}
    notes = []
    for template in TEMPLATES:
        url = template.format(y=day.year, m=f"{day.month:02d}", d=f"{day.day:02d}")
        try:
            html = http_get(url, timeout=45)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"FAIL {url} :: {exc}")
            continue
        before = len(store)
        _collect(html, store)
        notes.append(f"OK {url} (parsed {len(store) - before} new, {len(html)} bytes)")
    items = []
    for record in store.values():
        blob = f"{record['name']} {record['tagline']} {' '.join(record['topics'])}"
        record["ai_keywords"] = ai_hits(blob)
        record["platform_smell"] = platform_smell(blob)
        items.append(record)
    items.sort(key=lambda x: (x["votes"], x["comments"]), reverse=True)
    return items[:max_items], notes


if __name__ == "__main__":
    rows, status = fetch()
    print("\n".join(status))
    print(f"product hunt items: {len(rows)}")
    for row in rows[:15]:
        print(f"{row['votes']:>5} votes | {row['name']} | {row['tagline'][:70]}")
