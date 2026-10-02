"""Shared helpers for the daily intel scrapers."""

import datetime as dt
import json
import pathlib
import re
import time
import urllib.request

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

AI_KEYWORDS = (
    "ai ", " ai", "gpt", "llm", "agent", "model", "neural", "diffusion",
    "prompt", "copilot", "chatbot", "assistant", "voice", "speech", "video",
    "image", "vision", "ocr", "rag", "embedding", "inference", "openai",
    "anthropic", "claude", "gemini", "llama", "whisper", "flux", "stable",
    "transcrib", "generat", "summar", "automat",
)

PLATFORM_SMELL = (
    "platform", "suite", "all-in-one", "all in one", "everything", "end-to-end",
    "workspace", "ecosystem", "one-stop", "operating system for", "infrastructure",
)


def http_get(url, timeout=30, headers=None, retries=2):
    hdrs = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}
    if headers:
        hdrs.update(headers)
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001 - surfaced below
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries + 1} tries: {url} :: {last}")


def http_get_json(url, timeout=30, retries=2):
    return json.loads(http_get(url, timeout=timeout, retries=retries))


def http_get_bytes(url, timeout=60, retries=1):
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed: {url} :: {last}")


def write_json(path, payload):
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def today_pt():
    """Product Hunt resets at midnight Pacific time."""
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=7)).date()


def ai_hits(text):
    blob = f" {(text or '').lower()} "
    return sorted({kw.strip() for kw in AI_KEYWORDS if kw in blob})


def platform_smell(text):
    blob = (text or "").lower()
    return sorted({kw for kw in PLATFORM_SMELL if kw in blob})


def extract_json_objects(text, marker, max_items=600):
    """Pull embedded JSON objects out of a Next.js / Apollo HTML payload."""
    results = []
    idx = 0
    while len(results) < max_items:
        hit = text.find(marker, idx)
        if hit < 0:
            break
        start = text.rfind("{", 0, hit)
        if start < 0:
            idx = hit + len(marker)
            continue
        depth = 0
        in_str = False
        escaped = False
        pos = start
        while pos < len(text):
            char = text[pos]
            if in_str:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_str = False
            elif char == '"':
                in_str = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        chunk = text[start:pos + 1]
        try:
            results.append(json.loads(chunk))
        except Exception:  # noqa: BLE001 - partial cache entries are expected
            pass
        idx = pos + 1
    return results


def plain_text(value):
    """Strip HTML tags that sometimes wrap taglines."""
    return re.sub(r"<[^>]+>", "", value or "").strip()
