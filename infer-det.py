from models import TRTModule  # isort:skip
import argparse
from pathlib import Path

import cv2
import torch
import time

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    images = path_to_list(args.imgs)
    save_path = Path(args.out_dir)

    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    for image in images:
        save_image = save_path / image.name
        bgr = cv2.imread(str(image))
        print(" input image shape w: %d , h: %d" % (bgr.shape[0], bgr.shape[1]))
        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        # dwdh = torch.from_numpy()
        tensor = torch.asarray(tensor, device=device)
        print(tensor.shape)
        # inference
        data = None
        start_time = time.time()
        for i in range(100):
            data = Engine(tensor)
            # data 5 tensor: num bbox scores class_num
        end_time = time.time()
        print('infer time take: %.4f s' % ((end_time - start_time) / 100))
        print('output size: ', data)
        bboxes, scores, labels = det_postprocess(data)
        # for i in range(50):
        #     bboxes, scores, labels = det_postprocess(data)
        # #     num_dets, bboxes, scores, labels = data[0][0], data[1][0], data[2][
        # #         0], data[3][0]
        # post_end_time = time.time()
        # print(' nms postprocess take time is %.4f s' % ((post_end_time - end_time) / 50))
        # print('bboxes, scores, labels: ', bboxes, scores, labels)
        bboxes, scores, labels = bboxes, scores, labels

        if bboxes.numel() == 0:
            # if no bounding box
            print(f'{image}: no object!')
            continue
        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            score = round(float(score), 4)

            print('bbox, score, label:', bbox[:3], score, cls_id)

            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw,
                        f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255],
                        thickness=2)
        if args.show:
            cv2.imshow('result', draw)
            cv2.waitKey(0)
        else:
            cv2.imwrite(str(save_image), draw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='./weight/yolov8s.engine', help='Engine file')
    parser.add_argument('--imgs', type=str, default='./test_images/zidane.jpg', help='Images file')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--out-dir',
                        type=str,
                        default='./output',
                        help='Path to output file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
