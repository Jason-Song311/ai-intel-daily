"""SDXL 家族（Krea 2 / Illustrious / Pony / NoobAI / 通用 SDXL）工作流生成器。

和 flux_workflow.py、flux2_workflow.py 共用同一套节点发射器：
每个节点都带中文标题和中文注释，导出两种格式：
  ui_format  -> 直接拖进 ComfyUI 窗口就能用的画布格式
  api_format -> /prompt 接口用的格式，方便自动化跑批验证
"""

import argparse
import json
import pathlib

DEFAULT_PROMPT = (
    "masterpiece, best quality, a serene mountain lake at sunrise, "
    "soft golden light, cinematic composition, highly detailed"
)
DEFAULT_MODEL = "krea2Turbo_v10.safetensors"


def node(node_id, node_type, title, note, pos, size, fields, inputs=None, outputs=None):
    return {
        "id": node_id, "type": node_type, "title": title, "note": note,
        "pos": pos, "size": size, "fields": fields,
        "inputs": inputs or [], "outputs": outputs or [],
    }


def build(prompt, negative="", width=1024, height=1024, steps=28, cfg=6.0, seed=0,
          model=DEFAULT_MODEL, sampler="euler_ancestral", scheduler="karras",
          prefix="PACK"):
    """搭出 SDXL 家族的 7 个节点：加载底模 -> 双提示词 -> 画布 -> 采样 -> 解码 -> 保存。"""
    nodes = [
        node(1, "CheckpointLoaderSimple", "1. 加载底模（换模型的唯一入口）",
             "先把这个底模文件放进 ComfyUI/models/checkpoints/，然后在这里选中它。"
             "换风格只要换这里的文件，其他节点不用动。",
             [40, 80], [460, 110], {"ckpt_name": model},
             outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")]),

        node(2, "CLIPTextEncode", "2. 正向提示词（你想要的）",
             "写你想看到的内容。SDXL 家族按逗号分段，越靠前的词权重越大。"
             "推荐顺序：画质词 -> 主体 -> 动作 -> 环境 -> 光线 -> 镜头 -> 风格。",
             [560, 80], [460, 200], {"text": prompt},
             inputs=[("clip", "CLIP", 1, 1)],
             outputs=[("CONDITIONING", "CONDITIONING")]),

        node(3, "CLIPTextEncode", "3. 负向提示词（你不想要的）",
             "写你要排除的东西，SDXL 家族很吃负向词。"
             "常用的有：worst quality, low quality, blurry, extra fingers, watermark, text。",
             [560, 320], [460, 180], {"text": negative},
             inputs=[("clip", "CLIP", 1, 1)],
             outputs=[("CONDITIONING", "CONDITIONING")]),

        node(4, "EmptyLatentImage", "4. 画布尺寸（决定出图比例）",
             "SDXL 原生尺寸是 1024x1024。竖图 832x1216，横图 1216x832。"
             "其他比例会让画面出现重复或畸变。",
             [560, 540], [460, 130], {"width": width, "height": height, "batch_size": 1},
             outputs=[("LATENT", "LATENT")]),

        node(5, "KSampler", "5. 采样器（最核心的一个节点）",
             "步数：20-30 够用，Turbo 系模型 6-8 步。"
             "CFG：怎么写实/动漫模型用 5-7；Turbo 模型用 1-2。"
             "seed：同一个种子 + 同一段提示词 = 同一张图，改了种子才换构图。",
             [1080, 80], [460, 320],
             {"seed": seed, "control_after_generate": "randomize", "steps": steps,
              "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0},
             inputs=[("model", "MODEL", 1, 0), ("positive", "CONDITIONING", 2, 0),
                     ("negative", "CONDITIONING", 3, 0), ("latent_image", "LATENT", 4, 0)],
             outputs=[("LATENT", "LATENT")]),

        node(6, "VAEDecode", "6. 解码成图片",
             "把潜空间里的数据还原成能看的图片。底模自带的 VAE 一般就够用，"
             "画面发灰发暗时，再换成同系列的 VAE。",
             [1080, 440], [460, 100], {},
             inputs=[("samples", "LATENT", 5, 0), ("vae", "VAE", 1, 2)],
             outputs=[("IMAGE", "IMAGE")]),

        node(7, "SaveImage", "7. 保存图片",
             "文件名前缀可以自己改，出的图默认存在 ComfyUI/output/ 里。",
             [1560, 440], [420, 100], {"filename_prefix": prefix},
             inputs=[("images", "IMAGE", 6, 0)]),
    ]
    return nodes, wire(nodes)


def wire(nodes):
    """把每个节点声明的输入，解析成带编号的连线，所有工作流共用。"""
    links = []
    for target in nodes:
        target["input_links"] = {}
        for slot, (input_name, kind, origin_id, origin_slot) in enumerate(target.get("inputs", [])):
            links.append({
                "id": len(links) + 1, "kind": kind, "origin": origin_id,
                "origin_slot": origin_slot, "target": target["id"],
                "target_slot": slot, "target_name": input_name,
            })
            target["input_links"][input_name] = len(links)
    return links


def ui_format(nodes, links):
    """ComfyUI 画布格式：用户直接拖进窗口的那份。"""
    ui_nodes = []
    for order, item in enumerate(nodes):
        outgoing = {}
        for link in links:
            if link["origin"] == item["id"]:
                outgoing.setdefault(link["origin_slot"], []).append(link["id"])
        ui_nodes.append({
            "id": item["id"], "type": item["type"], "title": item["title"],
            "pos": item["pos"], "size": item["size"], "flags": {}, "order": order, "mode": 0,
            "inputs": [
                {"name": name, "type": kind, "link": item["input_links"].get(name)}
                for name, kind, _, _ in item["inputs"]
            ],
            "outputs": [
                {"name": name, "type": kind, "links": outgoing.get(slot) or None,
                 "slot_index": slot}
                for slot, (name, kind) in enumerate(item["outputs"])
            ],
            "properties": {"Node name for S&R": item["type"]},
            "widgets_values": list(item["fields"].values()),
        })

    notes = [
        {
            "id": 1000 + item["id"], "type": "Note",
            "pos": [item["pos"][0], item["pos"][1] - 130],
            "size": [max(380, item["size"][0]), 112], "flags": {}, "order": 0, "mode": 0,
            "outputs": [], "properties": {}, "color": "#432", "bgcolor": "#653",
            "widgets_values": [f"{item['title']}\n{item['note']}"],
        }
        for item in nodes
    ]

    return {
        "last_node_id": max(item["id"] for item in nodes),
        "last_link_id": len(links),
        "nodes": notes + ui_nodes,
        "links": [
            [link["id"], link["origin"], link["origin_slot"], link["target"],
             link["target_slot"], link["kind"]]
            for link in links
        ],
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 1.0, "offset": [0, 0]}},
        "version": 0.4,
    }


def api_format(nodes, links):
    """/prompt 接口格式，用来做自动化验证和批量出图。"""
    api = {}
    for item in nodes:
        inputs = dict(item["fields"])
        for link in links:
            if link["target"] == item["id"]:
                inputs[link["target_name"]] = [str(link["origin"]), link["origin_slot"]]
        api[str(item["id"])] = {
            "class_type": item["type"],
            "inputs": inputs,
            "_meta": {"title": item["title"]},
        }
    return api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative", default="worst quality, low quality, blurry")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--cfg", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="sdxl-ui.json")
    args = parser.parse_args()
    nodes, links = build(prompt=args.prompt, negative=args.negative, width=args.width,
                         height=args.height, steps=args.steps, cfg=args.cfg,
                         seed=args.seed, model=args.model)
    payload = ui_format(nodes, links)
    pathlib.Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()