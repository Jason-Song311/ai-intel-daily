"""把 markdown 日报推送到飞书自定义机器人。

Webhook 来源（按优先级）：--webhook / 环境变量 FEISHU_WEBHOOK / work/config.json 的 feishu_webhook。
默认用卡片格式（--card），飞书里排版好看；加 --text 则退化成纯文本。

注意：如果机器人在飞书后台开了「自定义关键词」，消息里必须包含那个关键词，
否则飞书会返回 code 19024。
"""

import argparse
import base64
import hashlib
import hmac
import json
import os
import pathlib
import re
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
    return text[:limit] + "\n\n...(内容过长，剩下的看本地日报文件)"


def sign(secret, timestamp):
    string_to_sign = timestamp + chr(10) + secret
    digest = hmac.new(string_to_sign.encode("utf-8"), b"", hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def markdown_to_lark(text):
    """把 markdown 转成飞书卡片能渲染的 lark_md。

    飞书卡片不认标题和表格，所以要降级：
      # / ## / ###  ->  **加粗** 一行
      表格行        ->  用 ｜ 拼成一行，丢掉分隔行
      - / * 列表    ->  • 开头
    """
    lines = []
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if re.match(r"^\|[\s:\-|]+\|$", stripped):
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip().replace("**", "") for c in stripped.strip("|").split("|")]
            stripped = " ｜ ".join(c for c in cells if c)
            lines.append(stripped)
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            lines.append(f"**{heading.group(2).strip()}**")
            continue
        if re.match(r"^[-*]\s+", stripped):
            lines.append("• " + re.sub(r"^[-*]\s+", "", stripped))
            continue
        lines.append(line)
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def build_payload(text, title, secret=None, card=True):
    if card:
        payload = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {"tag": "plain_text", "content": title or "AI 日报"},
                    "template": "blue",
                },
                "elements": [
                    {"tag": "div", "text": {"tag": "lark_md",
                                            "content": clip(markdown_to_lark(text))}},
                ],
            },
        }
    else:
        body = f"{title}\n\n{text}" if title else text
        payload = {"msg_type": "text", "content": {"text": clip(body)}}
    if secret:
        timestamp = str(int(time.time()))
        payload["timestamp"] = timestamp
        payload["sign"] = sign(secret, timestamp)
    return payload


def send(webhook, payload):
    request = urllib.request.Request(
        webhook,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="要发送的 markdown 文件")
    parser.add_argument("--text", dest="raw_text", help="直接发一段文本")
    parser.add_argument("--webhook", default=None)
    parser.add_argument("--secret", default=os.environ.get("FEISHU_SECRET"))
    parser.add_argument("--title", default=None)
    parser.add_argument("--plain", action="store_true", help="退化成纯文本而不是卡片")
    args = parser.parse_args()

    webhook = resolve_webhook(args.webhook)
    if not webhook:
        print("no webhook configured: pass --webhook, set FEISHU_WEBHOOK, or add work/config.json")
        sys.exit(1)
    if not args.file and not args.raw_text:
        print("pass --file or --text")
        sys.exit(1)

    text = args.raw_text or pathlib.Path(args.file).read_text(encoding="utf-8")
    payload = build_payload(text, args.title, args.secret, card=not args.plain)
    result = send(webhook, payload)
    code = result.get("code", result.get("StatusCode"))
    print("feishu response:", json.dumps(result, ensure_ascii=False)[:400])
    sys.exit(0 if code in (0, None) else 1)


if __name__ == "__main__":
    main()