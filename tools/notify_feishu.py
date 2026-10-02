"""Push a markdown report to a Feishu custom-bot webhook.

The webhook is read from (in order): --webhook, FEISHU_WEBHOOK env var,
or work/config.json {"feishu_webhook": "..."}.

Note: if the bot has keyword security enabled, the group has to contain the
keyword, otherwise Feishu rejects the message with code 19024.
"""

import argparse
import base64
import hashlib
import hmac
import json
import os
import pathlib
import sys
import time
import urllib.request


def resolve_webhook(explicit):
    if explicit:
        return explicit
    if os.environ.get("FEISHU_WEBHOOK"):
        return os.environ["FEISHU_WEBHOOK"]
    config = pathlib.Path(__file__).resolve().parents[2] / "config.json"
    if config.exists():
        data = json.loads(config.read_text(encoding="utf-8"))
        if data.get("feishu_webhook"):
            return data["feishu_webhook"]
    return None


def clip(text, limit=7000):
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n...(内容过长，完整版见本地文件)"


def sign(secret, timestamp):
    string_to_sign = timestamp + chr(10) + secret
    digest = hmac.new(string_to_sign.encode("utf-8"), b"", hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def send(webhook, text, secret=None):
    payload = {"msg_type": "text", "content": {"text": clip(text)}}
    if secret:
        timestamp = str(int(time.time()))
        payload["timestamp"] = timestamp
        payload["sign"] = sign(secret, timestamp)
    request = urllib.request.Request(
        webhook,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.load(response)
    return body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="markdown file to send")
    parser.add_argument("--webhook", default=None)
    parser.add_argument("--secret", default=os.environ.get("FEISHU_SECRET"))
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    webhook = resolve_webhook(args.webhook)
    if not webhook:
        print("no webhook configured: pass --webhook, set FEISHU_WEBHOOK, or add work/config.json")
        sys.exit(1)

    source = pathlib.Path(args.file)
    text = source.read_text(encoding="utf-8")
    if args.title:
        text = f"{args.title}\n\n{text}"
    result = send(webhook, text, args.secret)
    code = result.get("code", result.get("StatusCode"))
    print("feishu response:", json.dumps(result, ensure_ascii=False)[:400])
    sys.exit(0 if code in (0, None) else 1)


if __name__ == "__main__":
    main()
