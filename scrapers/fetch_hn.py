"""Hacker News 'Show HN' scraper (Algolia API, reachable from anywhere)."""

import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import ai_hits, http_get_json, platform_smell  # noqa: E402

API = "https://hn.algolia.com/api/v1/search"


def _domain(url):
    if not url or "//" not in url:
        return ""
    return url.split("/")[2].replace("www.", "")


def fetch(hours=48, min_points=3, limit=120):
    since = int(time.time()) - hours * 3600
    params = {
        "tags": "show_hn",
        "numericFilters": f"created_at_i>{since},points>={min_points}",
        "hitsPerPage": str(limit),
    }
    data = http_get_json(f"{API}?{urllib.parse.urlencode(params)}", timeout=30)
    items = []
    for hit in data.get("hits", []):
        title = hit.get("title") or ""
        story = (hit.get("story_text") or "").replace("<p>", " ").replace("</p>", " ")
        story = story[:800]
        url = hit.get("url")
        blob = f"{title} {story} {url or ''}"
        keywords = ai_hits(blob)
        items.append({
            "id": hit.get("objectID"),
            "title": title,
            "url": url,
            "hn_url": f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
            "domain": _domain(url),
            "points": hit.get("points") or 0,
            "comments": hit.get("num_comments") or 0,
            "created_at": hit.get("created_at"),
            "story_text": story,
            "ai_keywords": keywords,
            "platform_smell": platform_smell(f"{title} {story}"),
            "ai_score": len(keywords),
        })
    items.sort(key=lambda x: (x["ai_score"] > 0, x["points"], x["comments"]), reverse=True)
    return items


def report(items):
    lines = []
    for item in items[:20]:
        lines.append(
            f"{item['points']:>4} pts | {item['comments']:>3} cmt | {item['title']} | {item['url']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    rows = fetch()
    print(f"show_hn items: {len(rows)}")
    print(report(rows))
