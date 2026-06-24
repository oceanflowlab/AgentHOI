import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def _ensure_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    if x is None:
        return []
    return [x]


def _parse_box(box: Any) -> Tuple[float, float, float, float, float]:
    """
    支持 [x1,y1,x2,y2] 或 [x1,y1,x2,y2,score]。
    返回 (x1,y1,x2,y2,score)，无 score 时默认 1.0。
    """
    if not isinstance(box, Sequence) or len(box) < 4:
        raise ValueError(f"非法框格式: {box}")
    x1, y1, x2, y2 = map(float, box[:4])
    score = float(box[4]) if len(box) >= 5 else 1.0
    return x1, y1, x2, y2, score


def _safe_actions(actions: Any) -> str:
    action_list = _ensure_list(actions)
    action_list = [str(a).replace("_", " ").strip() for a in action_list if str(a).strip()]
    if not action_list:
        return "unknown action"
    # 去重并保持顺序
    seen = set()
    uniq = []
    for a in action_list:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return " | ".join(uniq)


def _pair_color(index: int) -> Tuple[int, int, int]:
    """给每组人-物体对分配稳定颜色，同一组使用同色框和同色文字。"""
    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (188, 246, 12),
        (250, 190, 212),
        (0, 128, 128),
        (220, 190, 255),
        (170, 110, 40),
        (128, 0, 0),
        (170, 255, 195),
        (128, 128, 0),
        (255, 216, 177),
        (0, 0, 128),
    ]
    return palette[index % len(palette)]


def _draw_label(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, color: Tuple[int, int, int], font: ImageFont.ImageFont) -> None:
    # 兼容 Pillow 版本：优先用 textbbox，失败再降级
    try:
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
        tw, th = right - left, bottom - top
    except Exception:
        tw, th = draw.textsize(text, font=font)

    pad_x, pad_y = 5, 3
    draw.rectangle(
        [x, y, x + tw + pad_x * 2, y + th + pad_y * 2],
        fill=(255, 255, 255),
        outline=color,
        width=2,
    )
    draw.text((x + pad_x, y + pad_y), text, fill=color, font=font)


def visualize_one_image(
    item: Dict[str, Any],
    image_root: Path,
    output_dir: Path,
    font: ImageFont.ImageFont,
    score_thresh: float,
) -> bool:
    filename = item.get("filename", "")
    if not filename:
        return False

    image_path = image_root / filename
    if not image_path.exists():
        # 兼容 json 内只保存文件名，不含子目录
        image_path = image_root / Path(filename).name
        if not image_path.exists():
            return False

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return False

    draw = ImageDraw.Draw(image)

    object_categories = _ensure_list(item.get("Object Category", []))
    interaction_classes = _ensure_list(item.get("Interaction Class", []))
    box_infos = _ensure_list(item.get("Box", []))

    for i, box_info in enumerate(box_infos):
        if not isinstance(box_info, dict):
            continue

        human_box = box_info.get("humanBox", [])
        obj_box = box_info.get("objBox", [])
        if not human_box or not obj_box:
            continue

        try:
            hx1, hy1, hx2, hy2, hscore = _parse_box(human_box)
            ox1, oy1, ox2, oy2, oscore = _parse_box(obj_box)
        except Exception:
            continue

        pair_score = hscore * oscore
        if pair_score < score_thresh:
            continue

        obj_name = str(object_categories[i]) if i < len(object_categories) else "object"
        actions = interaction_classes[i] if i < len(interaction_classes) else box_info.get("Interaction Class", [])

        action_text = _safe_actions(actions)
        object_text = obj_name
        pair_color = _pair_color(i)

        # 同一组人-物体对使用同色框
        draw.rectangle([hx1, hy1, hx2, hy2], outline=pair_color, width=3)
        draw.rectangle([ox1, oy1, ox2, oy2], outline=pair_color, width=3)

        # 按需求：人框标 action，物体框旁标 object，不标 person
        human_label_y = max(0, hy1 - 32)
        object_label_y = max(0, oy1 - 32)
        _draw_label(draw, hx1, human_label_y, action_text, pair_color, font)
        _draw_label(draw, ox1, object_label_y, object_text, pair_color, font)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / Path(filename).name
    image.save(save_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化 HOI JSON 预测结果")
    parser.add_argument(
        "--json",
        default="hico_pipe/output/4o_logit_style/4o_box.json",
        help="预测结果 JSON 路径",
    )
    parser.add_argument(
        "--image-root",
        default="data/hico_20160224_det/images/test2015",
        help="图片根目录",
    )
    parser.add_argument(
        "--output-dir",
        default="hico_pipe/output/4o_logit_style/vis",
        help="可视化结果输出目录",
    )
    parser.add_argument("--max-images", type=int, default=-1, help="最多可视化多少张，-1 表示全部")
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.0,
        help="按 human_score * object_score 过滤，默认不过滤",
    )
    parser.add_argument("--font-size", type=int, default=24, help="标签字体大小")
    args = parser.parse_args()

    json_path = Path(args.json)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON 顶层必须是 list")

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", args.font_size)
    except Exception:
        font = ImageFont.load_default()

    total = len(data) if args.max_images < 0 else min(len(data), args.max_images)
    success = 0

    for item in tqdm(data[:total], desc="Visualizing"):
        if visualize_one_image(item, image_root, output_dir, font, args.score_thresh):
            success += 1

    print(f"Done. Saved {success}/{total} images to: {output_dir}")


if __name__ == "__main__":
    main()
