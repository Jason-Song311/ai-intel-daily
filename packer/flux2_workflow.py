"""Flux.2 Klein text-to-image workflow - the light path for 8-12GB GPUs.

Same emitters as flux_workflow.py, different graph. Model, text encoder and VAE
pairing follow ComfyUI's own Flux.2 Klein blueprint.
"""

import argparse
import pathlib
import sys
from collections import OrderedDict

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from flux_workflow import api_format, ui_format, wire  # noqa: E402

UNET = "flux-2-klein-4b-fp8.safetensors"
CLIP = "qwen_3_4b.safetensors"
CLIP_TYPE = "flux2"
VAE = "flux2-vae.safetensors"

DEFAULT_PROMPT = (
    "a cozy bookstore on a rainy evening, warm light spilling onto wet cobblestones, "
    "cinematic photography, shallow depth of field, 35mm"
)


def build(prompt, negative="", width=1024, height=1024, steps=20, cfg=5.0,
          seed=0, prefix="FLUX2"):
    def node(node_id, node_type, title, note, pos, size, fields, inputs=None, outputs=None):
        return {
            "id": node_id, "type": node_type, "title": title, "note": note,
            "pos": pos, "size": size, "fields": OrderedDict(fields),
            "inputs": inputs or [], "outputs": outputs or [],
        }

    nodes = [
        node(1, "UNETLoader", "1 主模型（Flux.2 Klein 4B）",
             "只要 3.8GB，11GB 显存的显卡也能轻松跑。这是本机的推荐主力模型。",
             [30, 150], [360, 82],
             [("unet_name", UNET), ("weight_dtype", "default")],
             outputs=[("MODEL", "MODEL")]),

        node(2, "CLIPLoader", "2 文本编码器（qwen_3_4b）",
             "Flux.2 换用了 Qwen3-4B 当文本编码器，type 必须选 flux2，选错出不了图。",
             [30, 330], [360, 106],
             [("clip_name", CLIP), ("type", CLIP_TYPE), ("device", "default")],
             outputs=[("CLIP", "CLIP")]),

        node(3, "VAELoader", "3 VAE 解码器（flux2-vae）",
             "Flux.2 专用 VAE，和 Flux.1 的 ae.safetensors 不通用。",
             [30, 520], [360, 58],
             [("vae_name", VAE)],
             outputs=[("VAE", "VAE")]),

        node(4, "CLIPTextEncode", "4 正向提示词",
             "用英文写清楚：主体 + 环境 + 光线 + 镜头。Flux.2 理解自然语言，不需要堆标签。",
             [30, 650], [440, 190],
             [("text", prompt)],
             inputs=[("clip", "CLIP", 2, 0)],
             outputs=[("CONDITIONING", "CONDITIONING")]),

        node(5, "CLIPTextEncode", "5 负向提示词（默认留空）",
             "Flux.2 的 CFG 可以大于 1，负向词是生效的。不确定就先留空，出问题再填。",
             [520, 650], [440, 190],
             [("text", negative)],
             inputs=[("clip", "CLIP", 2, 0)],
             outputs=[("CONDITIONING", "CONDITIONING")]),

        node(6, "EmptyFlux2LatentImage", "6 画布尺寸",
             "Flux.2 使用自己的空白潜空间节点，不要用 SD3 或 SDXL 的。",
             [1010, 650], [300, 106],
             [("width", width), ("height", height), ("batch_size", 1)],
             outputs=[("LATENT", "LATENT")]),

        node(7, "Flux2Scheduler", "7 步数调度 Flux2Scheduler",
             "Klein 官方推荐 20 步。步数直接决定生成时长。",
             [1010, 800], [300, 106],
             [("steps", steps), ("width", width), ("height", height)],
             outputs=[("SIGMAS", "SIGMAS")]),

        node(8, "KSamplerSelect", "8 采样算法",
             "保持 euler 即可，Flux 系列标准和最稳的选择。",
             [1010, 950], [300, 58],
             [("sampler_name", "euler")],
             outputs=[("SAMPLER", "SAMPLER")]),

        node(9, "CFGGuider", "9 引导强度",
             "Klein 官方蓝图用 5。数字越大越贴提示词，太大画面会变硬。",
             [1360, 150], [320, 106],
             [("cfg", cfg)],
             inputs=[("model", "MODEL", 1, 0), ("positive", "CONDITIONING", 4, 0),
                     ("negative", "CONDITIONING", 5, 0)],
             outputs=[("GUIDER", "GUIDER")]),

        node(10, "RandomNoise", "10 随机种子",
             "换这个数字就是换构图，固定它可以复现同一张图。",
             [1360, 320], [320, 58],
             [("noise_seed", seed)],
             outputs=[("NOISE", "NOISE")]),

        node(11, "SamplerCustomAdvanced", "11 采样器（总控）",
             "把噪声、引导、采样算法、调度表、画布串起来执行采样，本机就是靠这个节点出图的。",
             [1360, 450], [320, 130],
             [],
             inputs=[("noise", "NOISE", 10, 0), ("guider", "GUIDER", 9, 0),
                     ("sampler", "SAMPLER", 8, 0), ("sigmas", "SIGMAS", 7, 0),
                     ("latent_image", "LATENT", 6, 0)],
             outputs=[("LATENT", "LATENT")]),

        node(12, "VAEDecode", "12 解码成图片",
             "把潜空间还原成可见图片，这一步报错基本都是 VAE 不对。",
             [1360, 640], [320, 46],
             [],
             inputs=[("samples", "LATENT", 11, 0), ("vae", "VAE", 3, 0)],
             outputs=[("IMAGE", "IMAGE")]),

        node(13, "SaveImage", "13 保存结果",
             "改 filename_prefix 决定图片存进 output 下哪个子文件夹。",
             [1360, 740], [320, 58],
             [("filename_prefix", prefix)],
             inputs=[("images", "IMAGE", 12, 0)]),
    ]
    return nodes, wire(nodes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix", default="FLUX2")
    parser.add_argument("--ui", default="work/packs/flux2-ui.json")
    parser.add_argument("--api", dest="api_path", default="work/packs/flux2-api.json")
    args = parser.parse_args()

    nodes, links = build(args.prompt, args.negative, args.width, args.height,
                         args.steps, args.cfg, args.seed, args.prefix)
    ui_path = pathlib.Path(args.ui)
    api_path = pathlib.Path(args.api_path)
    ui_path.parent.mkdir(parents=True, exist_ok=True)
    ui_path.write_text(json_dumps(ui_format(nodes, links)), encoding="utf-8")
    api_path.write_text(json_dumps(api_format(nodes, links)), encoding="utf-8")
    print(f"ui : {ui_path}")
    print(f"api: {api_path}")
    print(f"nodes={len(nodes)} links={len(links)}")


def json_dumps(payload):
    import json
    return json.dumps(payload, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
