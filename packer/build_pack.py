"""把某一天抓到的 Civitai 热门素材，按【底模】分文件夹，打成可以直接卖的"傻瓜包"。

产出结构（--out 目录下）：
    <底模>/01-示例图/            参考图（能联网时自动下载，否则写一份图片来源.txt）
    <底模>/02-提示词与参数.md     主案例逐段中文拆解 + 同底模其他素材全量清单
    <底模>/03-工作流.json         拖进 ComfyUI 就能用，每个节点带中文注释
    <底模>/04-使用说明.md         模型放哪、显存要求、每个节点干什么、常见报错
    <底模>/05-录屏脚本.md         3-5 分钟录屏逐段台词
    README.md                    整包索引
"""

import argparse
import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import flux2_workflow  # noqa: E402
import flux_workflow  # noqa: E402
import sdxl_workflow  # noqa: E402

# 底模分类：(文件夹名, 匹配关键词, 引擎, 友好名)
BASE_MODELS = (
    ("Flux2", ("flux.2", "flux2", "flux-2"), "flux2", "FLUX.2"),
    ("Flux1", ("flux.1", "flux1", "flux-1", "flux d", "flux s"), "flux1", "FLUX.1"),
    ("Qwen-Image", ("qwen",), "sdxl", "Qwen-Image"),
    ("Krea2", ("krea",), "sdxl", "Krea 2"),
    ("Illustrious", ("illustrious",), "sdxl", "Illustrious"),
    ("Pony", ("pony",), "sdxl", "Pony"),
    ("NoobAI", ("noob",), "sdxl", "NoobAI"),
)
FALLBACK = ("其他底模", "通用 SDXL")
PLACEHOLDER_MODEL = "PUT_YOUR_MODEL_HERE.safetensors"

# 提示词分段用的关键词表，顺序就是判断优先级
SEGMENT_RULES = (
    ("画质词", ("masterpiece", "best quality", "high quality", "highres", "high resolution",
              "absurdres", "ultra detailed", "extremely detailed", "8k", "4k", "sharp focus",
              "very aesthetic", "aesthetic", "perfect anatomy", "detailed")),
    ("镜头与构图", ("shot", "close-up", "closeup", "wide", "angle", "lens", "bokeh",
                "depth of field", "dof", "dslr", "macro", "telephoto", "fisheye", "panorama",
                "from above", "from below", "eye level", "full body", "upper body",
                "cowboy shot", "portrait", "centered", "symmetrical")),
    ("光线", ("light", "lighting", "glow", "glowing", "shadow", "shadows", "backlight",
            "rim light", "sunlight", "sunbeam", "neon", "golden hour", "dramatic",
            "volumetric", "bloom", "candle", "moonlight", "luminous")),
    ("风格", ("style", "art", "painting", "watercolor", "anime", "manga", "illustration",
            "realistic", "photorealistic", "cinematic", "film", "3d", "render", "concept art",
            "sketch", "photo", "photograph", "cel shading", "pastel", "gothic", "grim")),
    ("环境背景", ("background", "foreground", "scenery", "landscape", "sky", "city", "street",
              "forest", "ocean", "sea", "mountain", "room", "interior", "building", "garden",
              "night", "day", "sunset", "sunrise", "clouds", "rain", "snow", "field", "beach",
              "desert", "library", "cafe", "kitchen", "ruins", "canyon")),
)

PARAM_HELP = {
    "steps": "步数。20-30 是写实/动漫模型的常规区间；名字里带 Turbo / Lightning / 4step 的模型只要 4-8 步。步数越高细节越多，但收益递减。",
    "cfg_scale": "提示词服从度。写实和动漫模型一般 5-7；Turbo 类模型必须降到 1-2，否则画面会糊成一团。数值越高越“听话”但越僵硬。",
    "sampler": "采样算法。euler_ancestral 通用；dpmpp_2m + karras 细节更稳；dpmpp_sde 有颗粒感。",
    "seed": "随机种子。同种子 + 同提示词 = 完全同一张图；想微调构图就固定种子，想换构图就换种子。",
    "size": "画布尺寸。SDXL 家族原生 1024x1024，竖图 832x1216，横图 1216x832。偏离原生比例容易出现重复肢体。",
    "negative_prompt": "负向提示词。写你不想要的东西，SDXL 家族对负向词很敏感，噪点和畸形大多靠它压住。",
    "clip_skip": "跳过 CLIP 最后几层。数值越负越偏向“画风”而不是“字面意思”，动漫模型常用 -2。",
}


def pick_base_model(item):
    blob = " ".join([
        str(item.get("base_model") or ""),
        str(item.get("model_name") or ""),
    ]).lower()
    for folder, keywords, engine, friendly in BASE_MODELS:
        if any(k in blob for k in keywords):
            return folder, engine, friendly
    return FALLBACK[0], "sdxl", FALLBACK[1]


def clean_model_name(name, fallback):
    if not name:
        return fallback
    name = str(name).strip()
    if not name.lower().endswith((".safetensors", ".ckpt", ".pt")):
        name += ".safetensors"
    return name


def group_model_name(rows):
    counter = collections.Counter(
        str(r.get("model_name")).strip() for r in rows if r.get("model_name")
    )
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def guess_params(rows, engine):
    """从素材里推一组能直接跑通的默认参数，素材里缺参数时用行业默认值兜底。"""
    lead = rows[0]
    meta = lead.get("meta") or {}
    model = (group_model_name(rows) or "").lower()
    turbo = any(k in model for k in ("turbo", "lightning", "4step", "8step"))
    steps = meta.get("steps")
    cfg = meta.get("cfg_scale")
    sampler = meta.get("sampler")
    if not isinstance(steps, int):
        steps = 8 if turbo else 28
    if not isinstance(cfg, (int, float)):
        cfg = 1.5 if turbo else 6.0
    sampler = sampler if isinstance(sampler, str) and sampler else "euler_ancestral"
    return {
        "steps": int(steps), "cfg": float(cfg), "sampler": sampler,
        "scheduler": "simple" if turbo else "karras",
        "seed": meta.get("seed") if isinstance(meta.get("seed"), int) else 0,
        "width": meta.get("width") if isinstance(meta.get("width"), int) else 1024,
        "height": meta.get("height") if isinstance(meta.get("height"), int) else 1024,
        "turbo": turbo,
    }


def classify_segment(seg):
    low = seg.lower()
    for label, keywords in SEGMENT_RULES:
        if any(k in low for k in keywords):
            return label
    return "主体与动作"


def split_prompt(prompt):
    """把提示词拆成人能看懂的小块。

    标签流（a, b, c）按逗号拆；整段叙事（一整句话）保持完整，
    否则一个逗号就切断句子，拆出来的表格会看不懂。
    """
    text = str(prompt or "").strip()
    if not text:
        return []
    blocks = [b.strip() for b in text.split("\n") if b.strip()]
    segs = []
    for block in blocks:
        if len(block) > 90 or block.count(",") <= 2 and len(block) > 60:
            segs.append(block)
            continue
        for piece in block.split(","):
            piece = piece.strip()
            if piece:
                segs.append(piece)
    return segs


def build_workflow(engine, params, prompt, negative, model, prefix):
    if engine == "flux2":
        nodes, links = flux2_workflow.build(
            prompt=prompt, negative=negative, width=params["width"], height=params["height"],
            steps=max(8, min(params["steps"], 40)), seed=params["seed"], prefix=prefix)
        return flux2_workflow.ui_format(nodes, links)
    if engine == "flux1":
        nodes, links = flux_workflow.build(
            prompt=prompt, negative=negative, width=params["width"], height=params["height"],
            steps=max(8, min(params["steps"], 40)), seed=params["seed"], prefix=prefix)
        return flux_workflow.ui_format(nodes, links)
    nodes, links = sdxl_workflow.build(
        prompt=prompt, negative=negative, width=params["width"], height=params["height"],
        steps=max(4, min(params["steps"], 60)), cfg=params["cfg"], seed=params["seed"],
        model=model, sampler=params["sampler"], scheduler=params["scheduler"], prefix=prefix)
    return sdxl_workflow.ui_format(nodes, links)


def node_table(workflow):
    rows = []
    for item in workflow["nodes"]:
        if item["type"] == "Note":
            continue
        rows.append((item["title"], item["type"]))
    return rows


def credit_block(item):
    licence = item.get("license") or {}
    lines = [
        f"- 原图作者：{item.get('username') or '见原页面'}",
        f"- 使用的底模：{item.get('model_name') or '见原页面'}（{item.get('base_model') or '未知'}）",
        f"- 原页面链接：{item.get('civitai_url') or '见原页面'}",
    ]
    if licence:
        lines.append(f"- 商用许可：{licence.get('allowCommercialUse')}｜允许二创：{licence.get('allowDerivatives')}｜需署名：{'否' if licence.get('allowNoCredit') else '是'}")
    else:
        lines.append("- 授权信息：以 Civitai 原页面标注为准，二次使用前请自行确认")
    lines.append("- 本包只出售中文讲解、参数注释与工作流整理，不包含任何模型权重文件")
    return "\n".join(lines)


def prompt_doc(rows, friendly, engine, params):
    lead = rows[0]
    meta = lead.get("meta") or {}
    prompt = str(meta.get("prompt") or "")
    negative = str(meta.get("negative_prompt") or "")
    segs = split_prompt(prompt)

    out = [f"# 提示词与参数 · {friendly} 主案例", ""]
    out.append(f"> 抓取自 Civitai 热门榜 · {lead.get('posted_at') or ''}")
    out.append("")
    out.append("## 一、参数总表")
    out.append("")
    out.append("| 项目 | 数值 |")
    out.append("| --- | --- |")
    out.append(f"| 采样器 sampler | {params['sampler']} |")
    out.append(f"| 调度器 scheduler | {params['scheduler']} |")
    out.append(f"| 步数 steps | {params['steps']}{'（Turbo 类模型，少步数即可）' if params['turbo'] else ''} |")
    out.append(f"| CFG | {params['cfg']} |")
    out.append(f"| 尺寸 | {params['width']}x{params['height']} |")
    out.append(f"| 种子 seed | {params['seed']} |")
    out.append("")

    if prompt:
        out.append("## 二、正向提示词逐段拆解")
        out.append("")
        out.append("| # | 英文片段 | 位置 | 它在画面里管什么 |")
        out.append("| --- | --- | --- | --- |")
        for i, seg in enumerate(segs, 1):
            label = classify_segment(seg)
            out.append(f"| {i} | {seg.replace('|', '/')} | {label} | {segment_hint(label, seg)} |")
        out.append("")

    out.append("## 三、负向提示词")
    out.append("")
    out.append("```")
    out.append(negative or "（原素材未提供，建议补：worst quality, low quality, blurry, extra fingers, watermark, text）")
    out.append("```")
    out.append("")

    out.append("## 四、参数逐项中文讲解")
    out.append("")
    for key in ("steps", "cfg_scale", "sampler", "seed", "size", "negative_prompt"):
        out.append(f"- **{key}**：{PARAM_HELP[key]}")
    out.append("")

    out.append("## 五、英文原文（复制即用）")
    out.append("")
    out.append("```")
    out.append(prompt or "（原素材未提供）")
    out.append("```")
    out.append("")

    out.append("## 六、中文意译")
    out.append("")
    out.append("<!-- 待补：把上面那段英文按“主体 -> 动作 -> 环境 -> 光线 -> 镜头 -> 风格”写成一段通顺中文 -->")
    out.append("")

    out.append("## 七、授权与署名")
    out.append("")
    out.append(credit_block(lead))
    out.append("")

    out.append(f"## 附：同底模其他素材（共 {max(0, len(rows) - 1)} 条）")
    out.append("")
    for i, row in enumerate(rows[1:], 2):
        m = row.get("meta") or {}
        out.append(f"### 素材 {i} · 作者 {row.get('username') or '未知'}")
        out.append("")
        out.append(f"- 底模：{row.get('model_name') or '—'}")
        out.append(f"- 参数：{m.get('steps') or '默认'} 步 / CFG {m.get('cfg_scale') or '默认'} / {m.get('sampler') or params['sampler']} / 种子 {m.get('seed') or '—'}")
        out.append(f"- 链接：{row.get('civitai_url') or '—'}")
        out.append("- 正向提示词：")
        out.append("")
        out.append("```")
        out.append(str(m.get("prompt") or "（无）"))
        out.append("```")
        out.append("")
        if m.get("negative_prompt"):
            out.append("- 负向提示词：")
            out.append("")
            out.append("```")
            out.append(str(m.get("negative_prompt")))
            out.append("```")
            out.append("")
    return "\n".join(out) + "\n"


def segment_hint(label, seg):
    hints = {
        "画质词": "只影响精细度，不改变画面内容，删掉会变糙但不会变样",
        "镜头与构图": "决定观众站在哪看、看多近，改这里是换构图最快的方式",
        "光线": "决定画面是通透还是闷，删掉容易变得平",
        "风格": "决定“像照片还是像画”，是整段提示词里最影响风格的一句",
        "环境背景": "决定场景和氛围，换成自己的产品/场景时优先改这里",
        "主体与动作": "画面的主角，改成你自己的主体时保留这句以外的结构最稳",
    }
    return hints.get(label, "")


def usage_doc(friendly, engine, model, params, workflow):
    engine_note = {
        "sdxl": "SDXL 家族底模（放到 ComfyUI/models/checkpoints/）",
        "flux1": "FLUX.1 底模（放到 ComfyUI/models/unet/ 或 checkpoints/，另需 t5xxl + clip_l 文本编码器和 ae VAE）",
        "flux2": "FLUX.2 底模（放到 ComfyUI/models/unet/，另需 qwen_3_4b 文本编码器和 flux2-vae）",
    }[engine]
    vram = {"sdxl": "8GB 显存起步，12GB 更稳", "flux1": "12GB 显存起步（fp8 量化版）", "flux2": "8GB 显存可跑 4B 量化版"}[engine]
    lines = [f"# 使用说明 · {friendly}", ""]
    lines.append("## 这套包解决什么问题")
    lines.append("")
    lines.append("不用翻墙、不用看懂英文，照着三步走就能复刻出示例图的效果。")
    lines.append("")
    lines.append("## 一、准备工作")
    lines.append("")
    lines.append(f"1. 装好 ComfyUI（绘世启动器 / 官方版都行）。")
    lines.append(f"2. 准备底模：{engine_note}")
    lines.append(f"3. 显存要求：{vram}。")
    lines.append("")
    lines.append("## 二、三步出图")
    lines.append("")
    lines.append("1. 把 `03-工作流.json` 直接拖进 ComfyUI 的窗口，节点会自动排好。")
    lines.append("2. 点第 1 个节点「加载底模」，选中你放进去的模型文件。")
    lines.append("3. 点「运行」，出的图在 ComfyUI/output/ 目录里。")
    lines.append("")
    lines.append("## 三、每个节点是干什么的")
    lines.append("")
    lines.append("| 节点 | 类型 | 作用 |")
    lines.append("| --- | --- | --- |")
    for title, ntype in node_table(workflow):
        lines.append(f"| {title} | {ntype} | |")
    lines.append("")
    lines.append("## 四、怎么改成自己想要的样子")
    lines.append("")
    lines.append("- 换主体：只改正向提示词里描述主体的那几句，其他别动，构图最稳。")
    lines.append("- 换风格：把风格词换掉，或者直接换第 1 个节点里的底模。")
    lines.append("- 换比例：改第 4 个节点的宽高，竖图 832x1216，横图 1216x832。")
    lines.append("- 换构图：改第 5 个节点的 seed（种子）。")
    lines.append("")
    lines.append("## 五、常见报错")
    lines.append("")
    lines.append("| 现象 | 原因 | 怎么办 |")
    lines.append("| --- | --- | --- |")
    lines.append("| 节点红框 + 找不到模型 | 底模没放对目录或名字不一致 | 把文件放进对应目录，重启 ComfyUI，再重新选一次 |")
    lines.append("| 出图全黑 / 全灰 | VAE 不对 | 底模自带的 VAE 一般可用；发灰就换同系列 VAE |")
    lines.append("| 画面糊成一团 | CFG 太高或步数太少 | Turbo 类模型把 CFG 降到 1-2 |")
    lines.append("| 显存不足报错 | 尺寸太大 | 降到 768x768 或换 fp8/量化版模型 |")
    lines.append("")
    lines.append("## 六、合规提示")
    lines.append("")
    lines.append("- 本包出售的是中文讲解、参数注释和工作流整理，不含任何模型权重。")
    lines.append("- 商用前请按「02-提示词与参数.md」末尾的授权信息，回原页面确认许可范围。")
    lines.append("- AI 生成的图片对外发布时，请按《人工智能生成合成内容标识办法》加标识。")
    return "\n".join(lines) + "\n"


def video_script(friendly, params, model):
    return f"""# 3-5 分钟录屏脚本 · {friendly}

> 一边录屏一边照着念，语速放慢，动作和台词对上即可。

## 0:00-0:30 开场（说清解决什么问题）
"不会翻墙、看不懂英文也没关系。这个包里有 {friendly} 的完整参数和一键工作流，跟着做十分钟就能出同款图。"

## 0:30-1:30 装环境
- 打开绘世启动器 → 启动 ComfyUI
- 指出底模该放哪个目录：`{model}`
- 说一句显存要求：{params['steps']} 步这套参数，显存不够就把尺寸降到 768

## 1:30-2:30 导入工作流
- 把 `03-工作流.json` 拖进 ComfyUI 窗口
- 逐个点开节点，念出中文注释：加载底模 → 正向提示词 → 负向提示词 → 画布尺寸 → 采样器 → 解码 → 保存
- 强调"只有第 1 个节点需要换模型，其他都不用动"

## 2:30-3:30 第一次出图
- 检查尺寸 {params['width']}x{params['height']}、步数 {params['steps']}、CFG {params['cfg']}
- 点运行，展示进度条 → 出图
- 打开 output 文件夹，展示成品

## 3:30-4:30 改提示词
- 只改描述主体的那几句，把主体换成自己的产品或者人物
- 再跑一次，对比两张图的差别，说明"其他部分不动，构图就稳"

## 4:30-5:00 收尾
- 提醒三个坑：模型目录放错、VAE 选错、CFG 太高
- 说一句售后："参数看不懂可以直接问我，包里有中文逐项讲解"
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="work/data/latest.json")
    parser.add_argument("--out", default="work/packs/latest")
    parser.add_argument("--images", default=None, help="已下载的示例图目录")
    parser.add_argument("--per-folder", type=int, default=0, help="每个底模最多收几条素材，0 表示不限")
    args = parser.parse_args()

    bundle = json.loads(pathlib.Path(args.data).read_text(encoding="utf-8"))
    items = (bundle.get("data") or {}).get("civitai_flux") or []
    if not items:
        print("no civitai items in the bundle")
        sys.exit(1)

    grouped = collections.OrderedDict()
    for item in items:
        folder, engine, friendly = pick_base_model(item)
        grouped.setdefault(folder, {"engine": engine, "friendly": friendly, "rows": []})
        grouped[folder]["rows"].append(item)

    out_root = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    index = [f"# Civitai 热门素材傻瓜包 · {bundle.get('date')}", ""]
    index.append("按底模分类，每一类里都有一套能直接跑的工作流 + 中文参数讲解。")
    index.append("")
    index.append("| 文件夹 | 底模 | 素材数 | 主案例 |")
    index.append("| --- | --- | --- | --- |")

    ordered = sorted(grouped.items(), key=lambda kv: -len(kv[1]["rows"]))
    for folder, pack in ordered:
        rows = pack["rows"]
        if args.per_folder:
            rows = rows[:args.per_folder]
        engine, friendly = pack["engine"], pack["friendly"]
        params = guess_params(rows, engine)
        model = clean_model_name(group_model_name(rows), PLACEHOLDER_MODEL)
        target = out_root / folder
        (target / "01-示例图").mkdir(parents=True, exist_ok=True)

        (target / "02-提示词与参数.md").write_text(
            prompt_doc(rows, friendly, engine, params), encoding="utf-8")

        workflow = build_workflow(engine, params, str((rows[0].get("meta") or {}).get("prompt") or ""),
                                 str((rows[0].get("meta") or {}).get("negative_prompt") or ""),
                                 model, "PACK")
        (target / "03-工作流.json").write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")

        (target / "04-使用说明.md").write_text(
            usage_doc(friendly, engine, model, params, workflow), encoding="utf-8")
        (target / "05-录屏脚本.md").write_text(
            video_script(friendly, params, model), encoding="utf-8")

        copied = 0
        if args.images:
            source_dir = pathlib.Path(args.images)
            for row in rows:
                name = row.get("local_file")
                if name and (source_dir / name).exists():
                    (target / "01-示例图" / name).write_bytes((source_dir / name).read_bytes())
                    copied += 1
        if copied == 0:
            links = [f"{r.get('civitai_url')}" for r in rows if r.get("civitai_url")]
            (target / "01-示例图" / "图片来源.txt").write_text(
                "示例图原链接（本机网络下载不到时先留链接，能翻墙后补图）：\n"
                + "\n".join(links) + "\n", encoding="utf-8")

        index.append(f"| {folder} | {friendly} | {len(rows)} | {rows[0].get('civitai_url') or '—'} |")

    (out_root / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"pack built at {out_root}")
    for folder, pack in ordered:
        print(f"  {folder:12s} {pack['friendly']:12s} {len(pack['rows'])} items")


if __name__ == "__main__":
    main()