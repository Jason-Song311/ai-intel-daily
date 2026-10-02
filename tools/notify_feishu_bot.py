"""Send a markdown report through a Feishu self-built app bot.

Use this when the group has no custom-bot option. Credentials come from
--app-id/--app-secret, FEISHU_APP_ID/FEISHU_APP_SECRET, or work/config.json
keys "feishu_app_id" / "feishu_app_secret".

Run with --list-chats first: the bot prints every group it was added to,
then pass the wanted chat_id with --chat-id.
"""

import argparse
import json
import os
import pathlib
import sys
import urllib.request

BASE = "https://open.feishu.cn/open-apis"


def load_config():
    config = pathlib.Path(__file__).resolve().parents[2] / "config.json"
    if config.exists():
        return json.loads(config.read_text(encoding="utf-8"))
    return {}


def post(url, payload, token=None, method="POST"):
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def tenant_token(app_id, app_secret):
    result = post(f"{BASE}/auth/v3/tenant_access_token/internal",
                  {"app_id": app_id, "app_secret": app_secret})
    if result.get("code") != 0:
        raise RuntimeError(f"token failed: {json.dumps(result, ensure_ascii=False)}")
    return result["tenant_access_token"]


def list_chats(token):
    result = post(f"{BASE}/im/v1/chats?page_size=50", None, token, method="GET")
    if result.get("code") != 0:
        raise RuntimeError(f"chat list failed: {json.dumps(result, ensure_ascii=False)}")
    return (result.get("data") or {}).get("items") or []


def send(token, chat_id, text):
    payload = {"receive_id": chat_id, "msg_type": "text",
               "content": json.dumps({"text": text}, ensure_ascii=False)}
    return post(f"{BASE}/im/v1/messages?receive_id_type=chat_id", payload, token)


def clip(text, limit=7000):
    return text if len(text) <= limit else text[:limit] + "\n\n...(内容过长，完整版见本地文件)"


def main():
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--app-id", default=os.environ.get("FEISHU_APP_ID") or config.get("feishu_app_id"))
    parser.add_argument("--app-secret", default=os.environ.get("FEISHU_APP_SECRET") or config.get("feishu_app_secret"))
    parser.add_argument("--chat-id", default=config.get("feishu_chat_id"))
    parser.add_argument("--list-chats", action="store_true")
    parser.add_argument("--text")
    args = parser.parse_args()

    if not args.app_id or not args.app_secret:
        print("missing app credentials")
        sys.exit(1)

    token = tenant_token(args.app_id, args.app_secret)

    if args.list_chats:
        chats = list_chats(token)
        if not chats:
            print("机器人还没有被拉进任何群")
        for chat in chats:
            print(f"  chat_id={chat.get('chat_id')}  name={chat.get('name')}  members={chat.get('member_count')}")
        sys.exit(0)

    if not args.file and not args.text:
        print("pass --file or --text")
        sys.exit(1)
    text = args.text or pathlib.Path(args.file).read_text(encoding="utf-8")

    chat_id = args.chat_id
    if not chat_id:
        chats = list_chats(token)
        if len(chats) == 1:
            chat_id = chats[0]["chat_id"]
            print(f"using the only group: {chats[0].get('name')}")
        else:
            print("请用 --chat-id 指定群，或先用 --list-chats 查看")
            for chat in chats:
                print(f"  chat_id={chat.get('chat_id')}  name={chat.get('name')}")
            sys.exit(1)

    result = send(token, chat_id, clip(text))
    print("feishu response:", json.dumps(result, ensure_ascii=False)[:400])
    sys.exit(0 if result.get("code") == 0 else 1)


if __name__ == "__main__":
    main()
