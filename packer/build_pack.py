"""Turn one day of Civitai FLUX findings into category folders ready to sell.

Layout produced under --out:
    <category>/01-示例图/            reference renders
    <category>/02-提示词与参数.md     prompt, negative prompt, sampler table
    <category>/03-工作流.json         importable ComfyUI workflow
    <category>/04-使用说明.md         留白给中文讲解
    <category>/05-录屏脚本.md         3-5 分钟视频脚本骨架
    README.md                        pack index
"""

import argparse
import json
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import flux2_workflow  # noqa: E402
import flux_workflow  # noqa: E402

CATEGORIES = (
    ("写实人像", ("portrait", "woman", "man ", "face", "model ", "skin", "photo of")),
    ("电商产品图", ("product", "packshot", "bottle", "perfume", "cosmetic", "commercial",
                "studio shot", "advertisement")),
    ("建筑室内", ("interior", "room", "architecture", "building", "house", "kitchen",
              "living room", "cafe", "storefront")),
    ("动漫插画", ("anime", "manga", "illustration", "chibi", "waifu", "cartoon", "comic")),
    ("海报排版", ("poster", "typography", "logo", "text ", "title", "cover", "flyer")),
    ("风景氛围", ("landscape", "mountain", "forest", "sunset", "sky", "cinematic",
              "wallpaper", "sci-fi", "cyberpunk")),
)
FALLBACK = "其他风格"


def classify(item):
    meta = item.get("meta") or {}
    blob = " ".join([
        str(meta.get("prompt") or ""), str(item.get("model_name") or ""),
        str(item.get("base_model") or ""),
    ]).lower()
    scores = []
    for name, keywords in CATEGORIES:
        hits = sum(1 for keyword in keywords if keyword in blob)
        scores.append((hits, name))
    scores.sort(reverse=True)
    return scores[0][1] if scores and scores[0][0] > 0 else FALLBACK


def workflow_for(item, engine, out_path):
    meta = item.get("meta") or {}
    prompt = meta.get("prompt") or ""
    seed = meta.get("seed")
    steps = meta.get("steps") if isinstance(meta.get("steps"), int) else 20
    width = meta.get("width") if isinstance(meta.get("width"), int) else 1024
    height = meta.get("height") if isinstance(meta.get("height"), int) else 1024
    negative = meta.get("negative_prompt") or ""
    if engine == "flux2":
        nodes, links = flux2_workflow.build(
            prompt=prompt, negative=negative, width=width, height=height,
            steps=max(8, min(steps, 40)), seed=seed or 0, prefix="FLUX2")
        payload = flux2_workflow.api_format(nodes, links)
        ui_payload = flux2_workflow.ui_format(nodes, links)
    else:
        nodes, links = flux_workflow.build(
            prompt=prompt, negative=negative, width=width, height=height,
            steps=max(8, min(steps, 40)), seed=seed or 0, prefix="FLUX")
        payload = flux_workflow.api_format(nodes, links)
        ui_payload = flux_workflow.ui_format(nodes, links)
    out_path.write_text(json.dumps(ui_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def prompt_doc(item, engine):
    meta = item.get("meta") or {}
    licence = item.get("license") or {}
    rows = [
        ("正向提示词", meta.get("prompt")),
        ("负向提示词", meta.get("negative_prompt") or "（无）"),
        ("采样器", meta.get("sampler")),
        ("步数", meta.get("steps")),
        ("CFG", meta.get("cfg_scale")),
        ("种子", meta.get("seed")),
        ("尺寸", f"{meta.get('width')}x{meta.get('height')}"),
    ]
    lines = [f"# 提示词与参数（{engine}）", ""]
    lines.append("| 项目 | 原帖数值 |")
    lines.append("| --- | --- |")
    for label, value in rows:
        lines.append(f"| {label} | {str(value).replace('|', '/')} |")
    lines.append("")
    lines.append("## 英文原文")
    lines.append("")
    lines.append("```")
    lines.append(str(meta.get("prompt") or ""))
    lines.append("```")
    lines.append("")
    lines.append("## 中文讲解（待补）")
    lines.append("")
    lines.append("<!-- 逐段拆解提示词：主体 / 环境 / 光线 / 镜头 / 风格，并说明每一段改动会带来什么变化 -->")
    lines.append("")
    lines.append("## 原作者与授权")
    lines.append("")
    lines.append(f"- 作者：{item.get('username') or '未知'}")
    lines.append(f"- 模型：{licence.get('model') or item.get('model_name')}")
    lines.append(f"- 商用许可：{licence.get('allowCommercialUse')}")
    lines.append(f"- 允许二创：{licence.get('allowDerivatives')}")
    lines.append(f"- 是否需署名：{'否' if licence.get('allowNoCredit') else '是'}")
    lines.append(f"- 作品链接：{item.get('civitai_url')}")
    return "\n".join(lines) + "\n"


def video_script(item, category):
    return f"""# 3-5 分钟录屏脚本 · {category}

## 0:00-0:30 开场
先说清这套包解决什么问题：不会翻墙、看不懂英文，也能直接出这个风格的图。

## 0:30-1:30 装环境
打开绘世启动器 → 选择本工作流需要的模型 → 说明显存要求。

## 1:30-2:30 导入工作流
把 03-工作流.json 拖进 ComfyUI 窗口 → 指出每个高亮节点的作用。

## 2:30-3:30 出图
直接点运行，展示出图结果；再改一次种子，展示同提示词换构图。

## 3:30-4:30 改提示词
演示把「主体」换成自己的产品/人物，其他部分不动。

## 4:30-5:00 收尾
提示常见报错：模型没放对目录、VAE 选错、显存不足。

原始作品：{item.get('civitai_url')}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="work/data/latest.json")
    parser.add_argument("--images", default=None, help="folder holding downloaded civitai images")
    parser.add_argument("--out", default="work/packs/latest")
    parser.add_argument("--engine", default="flux2", choices=("flux2", "flux1"))
    parser.add_argument("--limit-per-category", type=int, default=4)
    args = parser.parse_args()

    bundle = json.loads(pathlib.Path(args.data).read_text(encoding="utf-8"))
    items = (bundle.get("data") or {}).get("civitai_flux") or []
    if not items:
        print("no civitai items in the bundle")
        sys.exit(1)

    grouped = {}
    for item in items:
        grouped.setdefault(classify(item), []).append(item)

    out_root = pathlib.Path(args.out)
    index = [f"# FLUX 傻瓜包 · {bundle.get('date')}", ""]
    for category, rows in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        rows = rows[:args.limit_per_category]
        folder = out_root / category
        (folder / "01-示例图").mkdir(parents=True, exist_ok=True)
        lead = rows[0]
        (folder / "02-提示词与参数.md").write_text(prompt_doc(lead, args.engine), encoding="utf-8")
        workflow_for(lead, args.engine, folder / "03-工作流.json")
        (folder / "04-使用说明.md").write_text(
            f"# 使用说明（{category}）\n\n<!-- 待补：模型放置路径、显存要求、逐节点讲解 -->\n",
            encoding="utf-8")
        (folder / "05-录屏脚本.md").write_text(video_script(lead, category), encoding="utf-8")
        copied = 0
        if args.images:
            source_dir = pathlib.Path(args.images)
            for row in rows:
                name = row.get("local_file")
                if name and (source_dir / name).exists():
                    shutil.copy2(source_dir / name, folder / "01-示例图" / name)
                    copied += 1
        index.append(f"- **{category}**：{len(rows)} 个风格，示例图 {copied} 张，参考作品 {lead.get('civitai_url')}")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"pack built at {out_root}")
    for category, rows in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        print(f"  {category}: {len(rows)} items")


if __name__ == "__main__":
    main()
