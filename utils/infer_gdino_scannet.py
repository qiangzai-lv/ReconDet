import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from huggingface_hub.utils import tqdm
from mmengine.config import Config
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mmdet.apis import inference_detector, init_detector


SCANNET_CLASSES = [
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
    'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
    'shower curtain', 'toilet', 'sink', 'bathtub', 'garbage bin'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Grounding DINO on a ScanNet scene and visualize boxes.')
    parser.add_argument('--image', default=None, help='Run one image only.')
    parser.add_argument('--scene', default='scene0000_00')
    parser.add_argument(
        '--data-root', default='/root/shared-nvme/data/ScanNet_processed')
    parser.add_argument(
        '--output-dir', default='work_dirs/gdino_scannet_visualization')
    parser.add_argument('--frame-step', type=int, default=1)
    parser.add_argument('--max-images', type=int, default=-1)
    parser.add_argument(
        '--config',
        default='configs/gdino/grounding_dino_swin-t_pretrain_obj365.py')
    parser.add_argument(
        '--checkpoint',
        default=(
            '/root/shared-nvme/data/pretrain/grounding_dino_swin-t_pretrain_'
            'obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'))
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--num-queries', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


COLORS = [
    (230, 57, 70), (29, 53, 87), (69, 123, 157), (42, 157, 143),
    (233, 196, 106), (244, 162, 97), (138, 43, 226), (0, 150, 136),
    (255, 112, 67), (63, 81, 181), (124, 179, 66), (255, 193, 7),
    (121, 85, 72), (0, 172, 193), (216, 27, 96), (76, 175, 80),
    (255, 87, 34), (96, 125, 139),
]


def collect_images(args):
    if args.image is not None:
        return [Path(args.image)]
    scene_dir = Path(args.data_root) / 'posed_images' / args.scene
    paths = sorted(scene_dir.glob('*.jpg'))[::args.frame_step]
    if args.max_images > 0:
        paths = paths[:args.max_images]
    return paths


def visualize(image, detections, output_path):
    visualization = image.convert('RGBA')
    draw = ImageDraw.Draw(visualization)
    font = ImageFont.load_default()
    for detection in detections:
        box, score, label = detection
        color = COLORS[label % len(COLORS)]
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=color + (255,), width=3)
        text = f'{SCANNET_CLASSES[label]} {score:.2f}'
        text_box = draw.textbbox((x1, y1), text, font=font)
        text_height = text_box[3] - text_box[1]
        text_y = max(0, y1 - text_height - 4)
        draw.rectangle(
            (x1, text_y, text_box[2] + 6, text_y + text_height + 4),
            fill=color + (230,))
        draw.text(
            (x1 + 3, text_y + 2), text,
            fill=(255, 255, 255, 255), font=font)
    visualization.convert('RGB').save(output_path, quality=95)


def resize_grounding_dino_queries(model, num_queries):
    query_embedding = model.query_embedding
    original_queries = query_embedding.num_embeddings
    if not 1 <= num_queries <= original_queries:
        raise ValueError(
            f'num_queries must be between 1 and {original_queries}, '
            f'got {num_queries}.')

    query_weight = query_embedding.weight[:num_queries].detach().clone()
    model.query_embedding = nn.Embedding.from_pretrained(
        query_weight, freeze=True)
    model.num_queries = num_queries

    if model.dn_query_generator is not None:
        model.dn_query_generator.num_matching_queries = num_queries
    if model.test_cfg is not None:
        max_per_img = model.test_cfg.get('max_per_img', num_queries)
        model.test_cfg['max_per_img'] = min(max_per_img, num_queries)

    print(f'GroundingDINO queries: {original_queries} -> {num_queries}')


def main():
    args = parse_args()
    image_paths = collect_images(args)
    image_paths = [path for path in image_paths if path.is_file()]
    if not image_paths:
        print('No input images found.')
        return

    cfg = Config.fromfile(args.config)
    checkpoint = args.checkpoint or cfg.get('load_from', None)
    if checkpoint is None:
        print('No Grounding DINO checkpoint was specified.')
        return

    # Activation checkpointing is a training-only memory optimization and
    # would require fairscale. Disable it for this inference utility.
    cfg.model.encoder.num_cp = 0
    model = init_detector(cfg, checkpoint=checkpoint, device=args.device)
    resize_grounding_dino_queries(model, args.num_queries)
    text_prompt = ' . '.join(SCANNET_CLASSES) + ' .'
    output_dir = Path(args.output_dir) / args.scene
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_index, image_path in tqdm(enumerate(image_paths, start=1)):
        result = inference_detector(
            model,
            str(image_path),
            text_prompt=text_prompt,
            custom_entities=True)
        instances = result.pred_instances
        boxes = instances.bboxes.float().cpu()
        scores = instances.scores.float().cpu()
        labels = instances.labels.cpu()
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            score = float(score)
            label = int(label)
            if score < args.score_thr:
                continue
            if 0 <= label < len(SCANNET_CLASSES):
                detections.append((box.tolist(), score, label))
                print(
                    f'[{image_index}/{len(image_paths)}] '
                    f'{image_path.name} class={SCANNET_CLASSES[label]} '
                    f'score={score:.4f} bbox_xyxy_pixels='
                    f'[{box[0]:.1f}, {box[1]:.1f}, '
                    f'{box[2]:.1f}, {box[3]:.1f}]')
        image = Image.open(image_path).convert('RGB')
        visualize(image, detections, output_dir / image_path.name)


if __name__ == '__main__':
    main()
