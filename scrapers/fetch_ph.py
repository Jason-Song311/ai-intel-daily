"""Product Hunt 'Today' leaderboard scraper.

producthunt.com blocks datacenter IPs, so the HTML path often answers 403 from
GitHub runners. Three strategies are tried in order:
  1. Official GraphQL API when PH_DEV_TOKEN / PRODUCT_HUNT_TOKEN is set
  2. Server-rendered HTML with full browser headers
  3. The public RSS feed
"""

import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import (  # noqa: E402
    UA,
    ai_hits,
    extract_json_objects,
    http_get,
    platform_smell,
    plain_text,
    today_pt,
)

GRAPHQL = "https://api.producthunt.com/v2/api/graphql"
MARKERS = ('"__typename":"Post"', '"__typename":"Product"')
TEMPLATES = (
    "https://www.producthunt.com/leaderboard/daily/{y}/{m}/{d}",
    "https://www.producthunt.com/",
    "https://www.producthunt.com/feed",
)
BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

QUERY = """
query Today($after: DateTime) {
  posts(order: VOTES, postedAfter: $after, first: 50) {
    edges {
      node {
        name
        tagline
        slug
        url
        website
        votesCount
        commentsCount
        featuredAt
        createdAt
        topics { edges { node { name } } }
      }
    }
  }
}
"""


def _topic_names(node):
    topics = node.get("topics")
    if isinstance(topics, dict):
        return [edge["node"]["name"] for edge in (topics.get("edges") or [])
                if isinstance(edge, dict) and edge.get("node")]
    if isinstance(topics, list):
        return [t.get("name") for t in topics if isinstance(t, dict) and t.get("name")]
    return []


def _record(node):
    return {
        "slug": node.get("slug"),
        "name": node.get("name"),
        "tagline": plain_text(node.get("tagline")),
        "url": node.get("url") or f"https://www.producthunt.com/posts/{node.get('slug')}",
        "website": node.get("website"),
        "votes": node.get("votesCount") or 0,
        "comments": node.get("commentsCount") or 0,
        "featured_at": node.get("featuredAt") or node.get("createdAt"),
        "topics": _topic_names(node),
    }


def _fetch_api(token, notes):
    day = today_pt()
    after = f"{day.isoformat()}T07:00:00Z"
    payload = json.dumps({"query": QUERY, "variables": {"after": after}}).encode("utf-8")
    request = urllib.request.Request(
        GRAPHQL, data=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                 "User-Agent": UA, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        data = json.load(response)
    if data.get("errors"):
        raise RuntimeError(str(data["errors"])[:300])
    edges = (((data.get("data") or {}).get("posts") or {}).get("edges")) or []
    notes.append(f"OK graphql api ({len(edges)} posts since {after})")
    return [_record(edge["node"]) for edge in edges if edge.get("node", {}).get("slug")]


def _collect_html(html, store):
    for marker in MARKERS:
        for obj in extract_json_objects(html, marker):
            slug = obj.get("slug")
            name = obj.get("name")
            if not isinstance(slug, str) or not isinstance(name, str) or not name:
                continue
            record = _record(obj)
            current = store.get(slug)
            if current is None or record["votes"] >= current["votes"]:
                store[slug] = record
    return store


def _collect_feed(xml, store):
    import re
    for item in re.findall(r"<item>(.*?)</item>", xml, re.S):
        title = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", item, re.S)
        link = re.search(r"<link>(.*?)</link>", item, re.S)
        if not title or not link:
            continue
        url = link.group(1).strip()
        slug = url.rstrip("/").split("/")[-1]
        store.setdefault(slug, {
            "slug": slug, "name": plain_text(title.group(1)), "tagline": "",
            "url": url, "website": None, "votes": 0, "comments": 0,
            "featured_at": None, "topics": [],
        })
    return store


def fetch(max_items=60):
    day = today_pt()
    notes = []
    store = {}

    token = os.environ.get("PH_DEV_TOKEN") or os.environ.get("PRODUCT_HUNT_TOKEN")
    if token:
        try:
            for record in _fetch_api(token, notes):
                store[record["slug"]] = record
        except Exception as exc:  # noqa: BLE001
            notes.append(f"FAIL graphql api :: {exc}")
    else:
        notes.append("no PH_DEV_TOKEN, falling back to html scraping")

    if len(store) < 10:
        for template in TEMPLATES:
            url = template.format(y=day.year, m=f"{day.month:02d}", d=f"{day.day:02d}")
            try:
                body = http_get(url, timeout=45, headers=BROWSER_HEADERS)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"FAIL {url} :: {exc}")
                continue
            before = len(store)
            if url.endswith("/feed"):
                _collect_feed(body, store)
            else:
                _collect_html(body, store)
            notes.append(f"OK {url} (+{len(store) - before}, {len(body)} bytes)")

    items = []
    for record in store.values():
        blob = f"{record['name']} {record['tagline']} {' '.join(record.get('topics') or [])}"
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
