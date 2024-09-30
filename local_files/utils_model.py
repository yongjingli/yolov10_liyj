import os.path
import cv2
from ultralytics import YOLOv10
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.models.yolov10.predict import YOLOv10DetectionPredictor
from ultralytics.data.loaders import LoadImagesAndVideos
import torch
import numpy as np
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.engine.results import Results


def img_pre_process(im, imgsz=[640, 640], device="cuda:0", use_fp16=False, stride=32):
    def pre_transform(im):
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(imgsz, auto=same_shapes and True, stride=stride)
        return [letterbox(image=x) for x in im]

    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = np.stack(pre_transform(im))
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

    im = im.to(device)
    im = im.half() if use_fp16 else im.float()  # uint8 to fp16/32
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return im


def postprocess(preds, img, orig_imgs, max_det=300, conf=0.25, classes=None, model_names=None, batch=None):
    if isinstance(preds, dict):
        preds = preds["one2one"]

    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    if preds.shape[-1] == 6:
        pass
    else:
        preds = preds.transpose(-1, -2)
        bboxes, scores, labels = ops.v10postprocess(preds, max_det, preds.shape[-1] - 4)
        bboxes = ops.xywh2xyxy(bboxes)
        preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)

    mask = preds[..., 4] > conf
    if classes is not None:
        mask = mask & (preds[..., 5:6] == torch.tensor(classes, device=preds.device).unsqueeze(0)).any(2)

    preds = [p[mask[idx]] for idx, p in enumerate(preds)]

    if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

    results = []
    for i, pred in enumerate(preds):
        orig_img = orig_imgs[i]
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        img_path = batch[0][i]
        results.append(Results(orig_img, path=img_path, names=model_names, boxes=pred))
    return results


def load_model(weights, device="cuda:0"):
    weights_path = "/home/pxn-lyj/Egolee/programs/yolov10_liyj/local_files/data/yolov10s.pt"
    model, ckpt = attempt_load_one_weight(weights_path)
    model.eval()
    model.to(device)
    return model


if __name__ == "__main__":
    print("Start")
    '''
    模型辅助类的utils，如前处理和后处理等
    '''
    print("End")