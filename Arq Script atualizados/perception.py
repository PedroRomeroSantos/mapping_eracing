import os
import sys
import numpy as np
import cv2
from ultralytics import YOLO
import torch

fsds_lib_path = os.path.join(
    os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python"
)
sys.path.insert(0, fsds_lib_path)
import fsds

FX = 640.0
CX = 640.0

CONF_THRESHOLD = 0.85

CAM_RGB   = "ZED_RGB"
CAM_DEPTH = "ZED_Depth"
IMG_SCENE = 0
IMG_DEPTH = 2

class PerceptionModule:
    DIST_DUPLICATA = 1.2
    Z_MIN = 0.5
    Z_MAX = 35.0

    def __init__(self, client, model_path: str):
        self.client = client
        self.model  = YOLO(model_path)
        if torch.cuda.is_available():
            self.model.to("cuda")

    def capturar_e_processar(self):
        frame_bgr, depth_map = self._capturar_imagens()
        if frame_bgr is None:
            return np.empty((0, 3)), None
        return self._processar(frame_bgr, depth_map)

    def _capturar_imagens(self):
        resp_rgb = self.client.simGetImages([
                fsds.ImageRequest(
                    camera_name     = CAM_RGB,
                    image_type      = IMG_SCENE,
                    pixels_as_float = False,
                    compress        = True,    
                )
        ])[0]

        resp_depth = self.client.simGetImages([
                fsds.ImageRequest(
                    camera_name     = CAM_DEPTH,
                    image_type      = IMG_DEPTH,
                    pixels_as_float = True,  
                    compress        = False,
                )
        ])[0]

        #decodifica RGB 
        img_1d = np.frombuffer(resp_rgb.image_data_uint8, dtype=np.uint8)
        frame  = cv2.imdecode(img_1d, cv2.IMREAD_COLOR)

        #decodifica Depth
        depth = np.array(resp_depth.image_data_float, dtype=np.float32)
        depth = depth.reshape(resp_depth.height, resp_depth.width)

        return frame, depth

    def _processar(self, frame_bgr, depth_map):
        results   = self.model(frame_bgr, conf=CONF_THRESHOLD, verbose=False)[0]
        debug_img = results.plot()

        boxes  = results.boxes.xywh.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy()

        h, w      = depth_map.shape
        raw_cones = []

        for (cx, cy, *_), label in zip(boxes, labels):
            u, v = int(cx), int(cy)
            if not (0 <= u < w and 0 <= v < h):
                continue
            Z = float(depth_map[v, u])
            if np.isnan(Z) or not (self.Z_MIN < Z < self.Z_MAX):
                continue
            X = (u - CX) * Z / FX
            raw_cones.append([X, Z, float(label)])

        cones = self._filtrar_duplicatas(raw_cones)
        return (np.array(cones) if cones else np.empty((0, 3))), debug_img

    def _filtrar_duplicatas(self, raw):
        final = []
        for c in raw:
            pos = np.array(c[:2])
            if not any(np.linalg.norm(pos - np.array(f[:2])) < self.DIST_DUPLICATA
                       for f in final):
                final.append(c)
        return final
