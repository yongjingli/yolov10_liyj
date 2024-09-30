import sys
sys.path.insert(0, "/home/pxn-lyj/Egolee/programs/yolov10_liyj")
import os
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

from utils_model import load_model, img_pre_process, postprocess
from utils_tools import GpuMemoryCalculator
from utils_tools import time_synchronized


def debug_test_model_performace(device="cuda:0"):
    # weights_path = "/home/pxn-lyj/Egolee/programs/yolov10_liyj/local_files/data/yolov10n.pt"
    # weights_path = "/home/pxn-lyj/Egolee/programs/yolov10_liyj/local_files/data/yolov10s.pt"
    weights_path = "/home/pxn-lyj/Egolee/programs/yolov10_liyj/local_files/data/yolov10m.pt"
    # weights_path = "/home/pxn-lyj/Egolee/programs/yolov10_liyj/local_files/data/yolov10b.pt"
    # weights_path = "/home/pxn-lyj/Egolee/programs/yolov10_liyj/local_files/data/yolov10l.pt"
    # weights_path = "/home/pxn-lyj/Egolee/programs/yolov10_liyj/local_files/data/yolov10x.pt"
    source_root = "/home/pxn-lyj/Egolee/programs/yolov10_liyj/local_files/data/TV_series_0210_scene_005_images"

    batch_size = 4
    imgsz = [768, 1280]
    max_count = 100
    use_half = 1

    # in case for pytorch context memory
    imgs = torch.zeros(batch_size, 3, imgsz[0], imgsz[1]).to(device)
    dataset = LoadImagesAndVideos(source_root, batch=batch_size, vid_stride=1)

    gpu_memory_calculator = GpuMemoryCalculator()
    count = 0
    m_allocateds = []
    m_reserveds = []
    m_max_reserveds = []
    all_gpu_usages = []
    gpu_usage_1s = []
    infer_times = []

    gpu_usage_0 = gpu_memory_calculator.get_device_gpu_memory_usage()
    m_allocated_0 = gpu_memory_calculator.get_torch_memory_allocated()
    m_reserved_0 = gpu_memory_calculator.get_torch_memory_reserved()
    m_max_reserved_0 = gpu_memory_calculator.get_torch_max_memory_reserved()

    model, ckpt = attempt_load_one_weight(weights_path, device=device)
    model_names = model.names

    # warmup
    if use_half:
        imgs = imgs.half()
        model.half()

    for _ in range(2):
        warmup_out = model(imgs)

    while 1:
        for count_batch, batch in enumerate(dataset):
            paths, im0s, s = batch
            im = img_pre_process(im0s, imgsz=imgsz, device=device, use_fp16=False, stride=32)
            if use_half:
                im = im.half()

            t1 = time_synchronized()
            with torch.no_grad():
                preds = model(im)
            t2 = time_synchronized()

            results = postprocess(preds, im, im0s, max_det=300, conf=0.25, classes=None, model_names=model_names, batch=batch)



            m_allocated_1 = gpu_memory_calculator.get_torch_memory_allocated()
            m_reserved_1 = gpu_memory_calculator.get_torch_memory_reserved()
            m_max_reserved_1 = gpu_memory_calculator.get_torch_max_memory_reserved()

            m_allocated = m_allocated_1 - m_allocated_0
            m_reserved = m_reserved_1 - m_reserved_0
            m_max_reserved = m_max_reserved_1 - m_max_reserved_0

            gpu_usage_1 = gpu_memory_calculator.get_device_gpu_memory_usage()
            all_gpu_usage = gpu_usage_1 - gpu_usage_0

            infer_time = (t2 - t1) * 1000

            print("m_allocated:{}, m_reserved:{}, m_max_reserved:{}, all_gpu_usage:{},"
                  " used memory: {}, infer_time: {}ms".format(m_allocated, m_reserved, m_max_reserved,
                                                              all_gpu_usage, gpu_usage_1, infer_time))

            result = results[0]
            plotted_img = result.plot(
                line_width=None,
                boxes=True,
                conf=True,
                labels=True,
                im_gpu=None,
            )

            cv2.imwrite("debug_test_model_performace.jpg", plotted_img)

            count = count + 1
            m_allocateds.append(m_allocated)
            m_reserveds.append(m_reserved)
            m_max_reserveds.append(m_max_reserved)
            all_gpu_usages.append(all_gpu_usage)
            gpu_usage_1s.append(gpu_usage_1)
            infer_times.append(infer_time)

        if count > max_count:
            break

    mean_m_allocated = round(np.mean(m_allocateds[20:]), 2)
    mean_m_reserved = round(np.mean(m_reserveds[20:]), 2)
    mean_m_max_reserved = round(np.mean(m_max_reserveds[20:]), 2)
    mean_all_gpu_usage = round(np.mean(all_gpu_usages[20:]), 2)
    mean_gpu_usage_1 = round(np.mean(gpu_usage_1s[20:]), 2)
    mean_infer_time = round(np.mean(infer_times[20:]), 2)

    print("mean_m_allocated: {}M, mean_m_reserved: {}M, mean_m_max_reserved: {}M, mean_all_gpu_usage: {}M,,"
          " mean_infer_time: {}ms, ".format(mean_m_allocated, mean_m_reserved, mean_m_max_reserved, mean_all_gpu_usage, mean_infer_time))


if __name__ == "__main__":
    print("Start")
    # weights: https://github.com/THU-MIG/yolov10/releases/tag/v1.1
    debug_test_model_performace()
    print("End")