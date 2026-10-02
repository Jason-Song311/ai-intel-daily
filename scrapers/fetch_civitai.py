"""Civitai trending FLUX images with full generation metadata.

civitai.com is unreachable from the mainland network, so this runs on the
GitHub Actions runner. Only public API endpoints are used.
"""

import pathlib
import sys
from urllib.parse import urlencode

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from common import http_get_bytes, http_get_json  # noqa: E402

IMAGES = "https://civitai.com/api/v1/images"
MODEL_VERSION = "https://civitai.com/api/v1/model-versions/{vid}"


def _text(*values):
    return " ".join(str(v) for v in values if v)


def _is_flux(raw, meta):
    haystack = _text(
        meta.get("Model"),
        meta.get("baseModel"),
        meta.get("Model type"),
        raw.get("baseModel"),
        raw.get("modelName"),
    ).lower()
    return "flux" in haystack


def _license(model_version_id):
    if not model_version_id:
        return {}
    try:
        data = http_get_json(MODEL_VERSION.format(vid=model_version_id), timeout=30, retries=1)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    return {
        "modelId": data.get("modelId"),
        "model": (data.get("model") or {}).get("name"),
        "version": data.get("name"),
        "baseModel": data.get("baseModel"),
        "allowNoCredit": data.get("allowNoCredit"),
        "allowCommercialUse": data.get("allowCommercialUse"),
        "allowDerivatives": data.get("allowDerivatives"),
        "allowDifferentLicense": data.get("allowDifferentLicense"),
    }


def fetch(limit=100, keep=15, download=5, image_dir=None, require_flux=False):
    query = urlencode({
        "limit": limit,
        "sort": "Most Reactions",
        "period": "Day",
        "nsfw": "None",
        "withMeta": "true",
    })
    data = http_get_json(f"{IMAGES}?{query}", timeout=45)
    raw_items = data.get("items") or []
    sample_keys = sorted(raw_items[0].keys()) if raw_items else []
    meta_present = sum(1 for item in raw_items if item.get("meta"))
    picked = []
    model_counter = {}
    for raw in raw_items:
        meta = raw.get("meta") or {}
        prompt = meta.get("prompt")
        if not prompt:
            continue
        name = meta.get("Model") or raw.get("modelName") or "unknown"
        model_counter[name] = model_counter.get(name, 0) + 1
        flux = _is_flux(raw, meta)
        if require_flux and not flux:
            continue
        stats = raw.get("stats") or {}
        picked.append({
            "is_flux": flux,
            "id": raw.get("id"),
            "image_url": raw.get("url"),
            "posted_at": raw.get("createdAt"),
            "username": raw.get("username"),
            "model_version_id": raw.get("modelVersionId"),
            "model_name": meta.get("Model") or raw.get("modelName"),
            "base_model": meta.get("baseModel") or raw.get("baseModel"),
            "stats": {
                "likes": stats.get("likeCount") or 0,
                "hearts": stats.get("heartCount") or 0,
                "comments": stats.get("commentCount") or 0,
            },
            "meta": {
                "prompt": prompt,
                "negative_prompt": meta.get("negativePrompt"),
                "sampler": meta.get("sampler"),
                "steps": meta.get("steps"),
                "cfg_scale": meta.get("cfgScale"),
                "seed": meta.get("seed"),
                "width": meta.get("width") or raw.get("width"),
                "height": meta.get("height") or raw.get("height"),
                "clip_skip": meta.get("clipSkip"),
            },
            "civitai_url": raw.get("url"),
        })
    flux_count = sum(1 for item in picked if item["is_flux"])
    picked.sort(key=lambda x: (x["is_flux"], x["stats"]["hearts"] + x["stats"]["likes"],
                               x["stats"]["comments"]), reverse=True)
    picked = picked[:keep]
    top_models = sorted(model_counter.items(), key=lambda kv: -kv[1])[:8]
    notes = [
        f"api items: {len(raw_items)}, items with meta: {meta_present}, "
        f"items with prompt: {sum(model_counter.values())}, flux matched: {flux_count}",
        "first item keys: " + ", ".join(str(key) for key in sample_keys),
        "top models: " + ", ".join(f"{name}({count})" for name, count in top_models),
    ]

    if image_dir:
        folder = pathlib.Path(image_dir)
        folder.mkdir(parents=True, exist_ok=True)
        saved = 0
        for item in picked:
            if saved >= download or not item["image_url"]:
                break
            suffix = pathlib.Path(item["image_url"].split("?")[0]).suffix or ".jpg"
            target = folder / f"{item['id']}{suffix}"
            try:
                target.write_bytes(http_get_bytes(item["image_url"], timeout=90, retries=1))
                item["local_file"] = target.name
                saved += 1
            except Exception as exc:  # noqa: BLE001
                item["download_error"] = str(exc)

    for item in picked:
        item["license"] = _license(item["model_version_id"])
    return picked, notes


if __name__ == "__main__":
    rows, status = fetch(keep=10, download=0)
    print("\n".join(status))
    print(f"flux items: {len(rows)}")
    for row in rows:
        meta = row["meta"]
        print(f"{row['stats']['hearts']:>5} hearts | {row['model_name']} | {str(meta['prompt'])[:70]}")
