"""Build a beginner-friendly FLUX.1 text-to-image workflow.

One graph definition, two outputs:
  * UI format  - drag and drop into the ComfyUI window (what customers get)
  * API format - POSTed to /prompt for automated validation and batch runs

Node widget order matches ComfyUI's own /object_info contract, so the generated
files load without "value not in list" errors.
"""

import argparse
import json
import pathlib
from collections import OrderedDict

MODEL_FP8 = "F.1基础算法模型-_F.1-dev-fp8.safetensors"
CLIP_T5 = "t5xxl_fp8_e4m3fn.safetensors"
CLIP_L = "clip_l.safetensors"
VAE_AE = "ae.safetensors"

DEFAULT_PROMPT = (
    "a cozy bookstore on a rainy evening, warm light spilling onto wet cobblestones, "
    "cinematic photography, shallow depth of field, 35mm"
)


def build(prompt, negative="", width=1024, height=1024, steps=20, guidance=3.5,
          seed=0, prefix="FLUX"):
    """Return the node list. Each node declares fields, inputs and outputs once."""

    def node(node_id, node_type, title, note, pos, size, fields, inputs=None, outputs=None):
        return {
            "id": node_id, "type": node_type, "title": title, "note": note,
            "pos": pos, "size": size, "fields": OrderedDict(fields),
            "inputs": inputs or [], "outputs": outputs or [],
        }

    nodes = [
        node(1, "UNETLoader", "1 主模型 UNETLoader",
             "选 FLUX.1 dev 的 fp8 主模型。它已经被压缩过一次了，weight_dtype 保持 default，"
             "再压一次会明显掉画质。",
             [30, 150], [340, 82],
             [("unet_name", MODEL_FP8), ("weight_dtype", "default")],
             outputs=[("MODEL", "MODEL")]),

        node(2, "DualCLIPLoader", "2 文本编码器 DualCLIPLoader",
             "FLUX 要两个文本编码器：clip_name1 填 t5xxl，clip_name2 填 clip_l，type 必须选 flux。"
             "这里选错，出图会变成糊色块。",
             [30, 330], [340, 106],
             [("clip_name1", CLIP_T5), ("clip_name2", CLIP_L), ("type", "flux")],
             outputs=[("CLIP", "CLIP")]),

        node(3, "VAELoader", "3 VAE 解码器 VAELoader",
             "FLUX 用独立的 ae.safetensors，不要用 SD1.5 / SDXL 的 VAE。",
             [30, 500], [340, 58],
             [("vae_name", VAE_AE)],
             outputs=[("VAE", "VAE")]),

        node(4, "CLIPTextEncode", "4 正向提示词",
             "FLUX 对自然语言敏感，按「主体 + 环境 + 光线 + 镜头」写成英文句子即可，"
             "不需要堆一长串标签。",
             [30, 630], [420, 200],
             [("text", prompt)],
             inputs=[("clip", "CLIP", 2, 0)],
             outputs=[("CONDITIONING", "CONDITIONING")]),

        node(5, "CLIPTextEncode", "5 负向提示词（默认留空）",
             "FLUX dev 的 CFG 固定为 1，负向词几乎不生效，留空最安全，填了反而容易干扰画面。",
             [500, 630], [420, 200],
             [("text", negative)],
             inputs=[("clip", "CLIP", 2, 0)],
             outputs=[("CONDITIONING", "CONDITIONING")]),

        node(6, "FluxGuidance", "6 引导强度 FluxGuidance",
             "FLUX 专属参数，常用 2.5-4.5。数字越大越贴合提示词，过大画面会发死发灰。",
             [970, 630], [300, 58],
             [("guidance", guidance)],
             inputs=[("conditioning", "CONDITIONING", 4, 0)],
             outputs=[("CONDITIONING", "CONDITIONING")]),

        node(7, "EmptySD3LatentImage", "7 画布尺寸",
             "FLUX 走 SD3 风格的空白潜空间。1024x1024 最稳，非正方形也尽量保持百万像素级别。",
             [970, 730], [300, 106],
             [("width", width), ("height", height), ("batch_size", 1)],
             outputs=[("LATENT", "LATENT")]),

        node(8, "KSampler", "8 采样器 KSampler",
             "FLUX dev 标准参数：steps 20、cfg 1.0、sampler euler、scheduler simple。"
             "seed 固定就能复现同一张图，换 seed 等于换构图。",
             [1330, 150], [330, 262],
             [("seed", seed), ("steps", steps), ("cfg", 1.0), ("sampler_name", "euler"),
              ("scheduler", "simple"), ("denoise", 1.0)],
             inputs=[("model", "MODEL", 1, 0), ("positive", "CONDITIONING", 6, 0),
                     ("negative", "CONDITIONING", 5, 0), ("latent_image", "LATENT", 7, 0)],
             outputs=[("LATENT", "LATENT")]),

        node(9, "VAEDecode", "9 解码成图片",
             "把潜空间数据还原成可见图片。这一步报错，基本就是 VAE 选错了。",
             [1330, 470], [300, 46],
             [],
             inputs=[("samples", "LATENT", 8, 0), ("vae", "VAE", 3, 0)],
             outputs=[("IMAGE", "IMAGE")]),

        node(10, "SaveImage", "10 保存结果",
             "改 filename_prefix 就能决定图片存进 output 下的哪个子文件夹。",
             [1330, 570], [330, 58],
             [("filename_prefix", prefix)],
             inputs=[("images", "IMAGE", 9, 0)]),
    ]

    return nodes, wire(nodes)


def wire(nodes):
    """Resolve every declared input into a numbered link, shared by all graphs."""
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
    """ComfyUI canvas format: what a user drags into the window."""
    ui_nodes = []
    for order, node in enumerate(nodes):
        outgoing = {}
        for link in links:
            if link["origin"] == node["id"]:
                outgoing.setdefault(link["origin_slot"], []).append(link["id"])
        ui_nodes.append({
            "id": node["id"], "type": node["type"], "title": node["title"],
            "pos": node["pos"], "size": node["size"], "flags": {}, "order": order, "mode": 0,
            "inputs": [
                {"name": name, "type": kind, "link": node["input_links"].get(name)}
                for name, kind, _, _ in node["inputs"]
            ],
            "outputs": [
                {"name": name, "type": kind, "links": outgoing.get(slot) or None,
                 "slot_index": slot}
                for slot, (name, kind) in enumerate(node["outputs"])
            ],
            "properties": {"Node name for S&R": node["type"]},
            "widgets_values": list(node["fields"].values()),
        })

    notes = [
        {
            "id": 1000 + node["id"], "type": "Note",
            "pos": [node["pos"][0], node["pos"][1] - 120],
            "size": [max(340, node["size"][0]), 104], "flags": {}, "order": 0, "mode": 0,
            "outputs": [], "properties": {}, "color": "#432", "bgcolor": "#653",
            "widgets_values": [f"{node['title']}\n{node['note']}"],
        }
        for node in nodes
    ]

    return {
        "last_node_id": max(node["id"] for node in nodes),
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
    """/prompt payload format used for automated validation and batch runs."""
    api = {}
    for node in nodes:
        inputs = dict(node["fields"])
        for link in links:
            if link["target"] == node["id"]:
                inputs[link["target_name"]] = [str(link["origin"]), link["origin_slot"]]
        api[str(node["id"])] = {
            "class_type": node["type"],
            "inputs": inputs,
            "_meta": {"title": node["title"]},
        }
    return api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix", default="FLUX")
    parser.add_argument("--ui", default="work/packs/workflow-ui.json")
    parser.add_argument("--api", dest="api_path", default="work/packs/workflow-api.json")
    args = parser.parse_args()

    nodes, links = build(args.prompt, args.negative, args.width, args.height,
                         args.steps, args.guidance, args.seed, args.prefix)
    ui_path = pathlib.Path(args.ui)
    api_path = pathlib.Path(args.api_path)
    ui_path.parent.mkdir(parents=True, exist_ok=True)
    ui_path.write_text(json.dumps(ui_format(nodes, links), ensure_ascii=False, indent=2),
                       encoding="utf-8")
    api_path.write_text(json.dumps(api_format(nodes, links), ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"ui : {ui_path}")
    print(f"api: {api_path}")
    print(f"nodes={len(nodes)} links={len(links)}")


if __name__ == "__main__":
    main()
