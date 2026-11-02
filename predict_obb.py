from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else BASE / path).resolve()


def class_names(value):
    if isinstance(value, dict):
        indexed = {int(key): str(name) for key, name in value.items()}
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError('Class IDs phải liên tiếp, bắt đầu từ 0.')
        names = [indexed[i] for i in range(len(indexed))]
    elif isinstance(value, list):
        names = [str(name) for name in value]
    else:
        raise ValueError('Cấu hình names phải là danh sách hoặc bảng class ID: tên.')
    if not names or len(set(names)) != len(names) or any(not name.strip() or '\n' in name or '\r' in name for name in names):
        raise ValueError('Tên lớp phải duy nhất, không trống và chỉ có một dòng.')
    return names


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def make_splits(records, ratios, seed):
    """Allocate all records once, ensuring each configured split is nonempty."""
    records = list(records)
    random.Random(seed).shuffle(records)
    active = [name for name, ratio in ratios.items() if ratio > 0]
    if len(records) < len(active):
        raise ValueError(f'Cần ít nhất {len(active)} ảnh hợp lệ cho {len(active)} tập. '
                         'Thêm ảnh hoặc đặt tỷ lệ tập không dùng thành 0.')
    counts = {name: (max(1, int(len(records)*ratios[name])) if name in active else 0) for name in ratios}
    while sum(counts.values()) < len(records):
        name = max(active, key=lambda s: len(records)*ratios[s]-counts[s])
        counts[name] += 1
    while sum(counts.values()) > len(records):
        name = max((s for s in active if counts[s] > 1), key=lambda s: counts[s]-len(records)*ratios[s])
        counts[name] -= 1
    result, offset = {}, 0
    for name, count in counts.items():
        result[name] = records[offset:offset+count]
        offset += count
    return result


def main():
    parser = argparse.ArgumentParser(description='Tự gán nhãn YOLO OBB và xuất dataset mới.')
    parser.add_argument('--config', default='config_obb/predict.yaml')
    args = parser.parse_args()
    # Imports are delayed so importing this file never loads a model or starts prediction.
    import yaml
    from PIL import Image
    from ultralytics import YOLO

    config_path = resolve_path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))
    if not isinstance(cfg, dict):
        raise ValueError('File cấu hình predict phải là một YAML mapping.')
    model_path = resolve_path(cfg['model'])
    source = resolve_path(cfg['source'])
    output = resolve_path(cfg['output'])
    dataset_path = resolve_path(cfg['dataset_config'])
    dataset = yaml.safe_load(dataset_path.read_text(encoding='utf-8-sig'))
    names = class_names(dataset['names'])
    if 'nc' in dataset and int(dataset['nc']) != len(names):
        raise ValueError('nc không khớp số lượng names trong data.yaml.')
    if not model_path.is_file():
        raise FileNotFoundError(f'Không tìm thấy model: {model_path}')
    if not source.is_dir():
        raise FileNotFoundError(f'Hãy tạo thư mục và thêm ảnh vào: {source}')
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Thư mục output và source phải tách biệt, không lồng nhau.')
    images = sorted((p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS),
                    key=lambda p: p.relative_to(source).as_posix())
    if not images:
        raise ValueError(f'Không tìm thấy ảnh trong {source}')
    ratios = {key: float(cfg['split_ratios'].get(key, 0)) for key in ('train', 'val', 'test')}
    if any(not math.isfinite(v) or v < 0 for v in ratios.values()) or not math.isclose(sum(ratios.values()), 1, abs_tol=1e-8):
        raise ValueError('split_ratios phải không âm và có tổng bằng 1.')
    if ratios['train'] <= 0 or ratios['val'] <= 0:
        raise ValueError('Dataset train cần tỷ lệ train và val lớn hơn 0.')
    make_splits(images, ratios, int(cfg.get('seed', 1234)))
    conf, iou = float(cfg.get('confidence', .25)), float(cfg.get('iou', .7))
    if not (0 <= conf <= 1 and 0 <= iou <= 1):
        raise ValueError('confidence và iou phải nằm trong [0, 1].')
    imgsz, max_det = int(cfg.get('imgsz', 1024)), int(cfg.get('max_det', 1000))
    if imgsz <= 0 or max_det <= 0:
        raise ValueError('imgsz và max_det phải lớn hơn 0.')
    rotate_180 = cfg.get('rotate_180', True)
    if not isinstance(rotate_180, bool):
        raise ValueError('rotate_180 phải là true hoặc false trong YAML.')

    model = YOLO(str(model_path))
    if model.task != 'obb':
        raise ValueError(f'Model phải thuộc task obb; model hiện tại: {model.task}')
    model_names = class_names(model.names)
    if set(model_names) != set(names):
        raise ValueError(f'Lớp trong best.pt không khớp data.yaml.\nModel: {model_names}\nConfig: {names}')
    class_map = {index: names.index(name) for index, name in enumerate(model_names)}

    run = output / datetime.now().strftime('dataset_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True, exist_ok=False)
    staging = run / '_pending'
    staging.mkdir()
    if cfg.get('save_predict', True):
        (run / 'predict').mkdir()
    report = {
        'status': 'running', 'model': str(model_path), 'source': str(source),
        'config': cfg, 'names': names, 'total_source_images': len(images),
        'exported_images': 0, 'boxes': 0, 'empty_images': 0,
        'class_counts': {name: 0 for name in names}, 'skipped': [],
    }
    write_json(run / 'predict_report.json', report)
    accepted = []
    counts = Counter()
    try:
        with (run / 'manifest.jsonl').open('w', encoding='utf-8') as manifest:
            for index, path in enumerate(images):
                relative = path.relative_to(source).as_posix()
                key = f'{index:08d}'  # Unique even for identical stems/extensions in different folders.
                print(f'[{index+1}/{len(images)}] {relative}', flush=True)
                # Decode failures are logged; inference/device failures stop the run explicitly.
                try:
                    with Image.open(path) as image:
                        input_image = image.convert('RGB')
                    if rotate_180:
                        input_image = input_image.transpose(Image.Transpose.ROTATE_180)
                except (OSError, ValueError) as exc:
                    report['skipped'].append({'source': relative, 'reason': str(exc)})
                    continue
                # Pass the rotated RGB image itself, not the original file path.
                options = dict(source=input_image, conf=conf, iou=iou, imgsz=imgsz,
                               max_det=max_det, save=False, save_txt=False, verbose=False)
                if cfg.get('device') is not None:
                    options['device'] = str(cfg['device'])
                result = model.predict(**options)[0]
                if result.obb is None:
                    raise ValueError(f'Model không trả về dữ liệu OBB: {relative}')
                if cfg.get('save_predict', True):
                    Image.fromarray(result.plot()[:, :, ::-1]).save(run / 'predict' / f'{key}.jpg')
                corners = result.obb.xyxyxyxyn.cpu().tolist()
                class_ids = result.obb.cls.cpu().tolist()
                confidences = result.obb.conf.cpu().tolist()
                lines, predicted = [], []
                invalid = False
                for points, cls, confidence in zip(corners, class_ids, confidences):
                    values = [coordinate for point in points for coordinate in point]
                    # Do not clip OBB corners: clipping can turn a rectangle into a distorted polygon.
                    if not all(math.isfinite(v) and -1e-6 <= v <= 1+1e-6 for v in values):
                        invalid = True
                        break
                    target_cls = class_map[int(cls)]
                    values = [max(0., min(1., v)) for v in values]  # Floating-point tolerance only.
                    lines.append(f'{target_cls} '+' '.join(f'{v:.8f}' for v in values))
                    predicted.append({'class_id': target_cls, 'confidence': float(confidence)})
                if invalid:
                    # Skip the whole image so omitted objects do not become false negative labels.
                    report['skipped'].append({'source': relative, 'predict': f'predict/{key}.jpg' if cfg.get('save_predict', True) else None,
                                              'reason': 'OBB có đỉnh nằm ngoài biên ảnh hoặc tọa độ không hữu hạn.'})
                    continue
                # Save exactly the pixel orientation used for inference, without EXIF ambiguity.
                Image.fromarray(result.orig_img[:, :, ::-1]).save(staging / f'{key}.png')
                (staging / f'{key}.txt').write_text('\n'.join(lines)+ ('\n' if lines else ''), encoding='utf-8')
                accepted.append(key)
                counts.update(item['class_id'] for item in predicted)
                report['boxes'] += len(lines)
                report['empty_images'] += int(not lines)
                manifest.write(json.dumps({'id': key, 'source': relative, 'rotation_degrees': 180 if rotate_180 else 0,
                                           'predictions': predicted}, ensure_ascii=False)+'\n')
                manifest.flush()

        splits = make_splits(accepted, ratios, int(cfg.get('seed', 1234)))
        exported_yaml = {'path': run.as_posix()}
        for split, keys in splits.items():
            if ratios[split] == 0:
                continue
            image_dir, label_dir = run / 'images' / split, run / 'labels' / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for key in keys:
                (staging / f'{key}.png').replace(image_dir / f'{key}.png')
                (staging / f'{key}.txt').replace(label_dir / f'{key}.txt')
            exported_yaml[split] = f'images/{split}'
        staging.rmdir()
        exported_yaml.update(nc=len(names), names=dict(enumerate(names)))
        (run / 'data.yaml').write_text(yaml.safe_dump(exported_yaml, allow_unicode=True, sort_keys=False), encoding='utf-8')
        (run / 'classes.txt').write_text('\n'.join(names)+'\n', encoding='utf-8')
        report.update(status='completed', exported_images=len(accepted),
                      split_counts={split: len(keys) for split, keys in splits.items()},
                      class_counts={name: counts[i] for i, name in enumerate(names)})
        print(f'Đã xuất {len(accepted)} ảnh, {report["boxes"]} OBB: {run}')
        print(f'Bỏ qua {len(report["skipped"])} ảnh. Chi tiết: predict_report.json')
    except BaseException as exc:
        report.update(status='incomplete', error=str(exc), processed_images=len(accepted))
        raise
    finally:
        write_json(run / 'predict_report.json', report)


if __name__ == '__main__':
    main()
