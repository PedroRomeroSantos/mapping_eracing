import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================
# CONFIGURAÇÃO GLOBAL DO FSDS (FORA DA CLASSE)
# ==============================================================
fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
if fsds_lib_path not in sys.path:
    sys.path.insert(0, fsds_lib_path)

import fsds  # Agora o fsds está disponível para todas as funções do arquivo

# ==============================
# PARAMETROS (Mantidos)
# ==============================
C_X = 512.0
C_Y = 360.0
F_X = 512.0
F_Y = 512.0
CONF_THRESHOLD = 0.8
CAMERA_HEIGHT = 0.0 
ANGLE_DEG = 30.0

class PerceptionSystem:
    def __init__(self):
        print("Inicializando Sistema de Percepção (FSDS + YOLO)...")
        
        # Conexão com o Simulador
        self.client = fsds.FSDSClient()
        self.client.confirmConnection()
        
        # YOLO SETUP
        model_path = "/home/pedro/mapping_eracing/16_01.pt" # Ajuste para o seu caminho real
        self.model = YOLO(model_path)

    def ray_lidar_intersection(self, points, ray, angle_deg=1.0):
        if points.shape[0] == 0: return None
        norms = np.linalg.norm(points, axis=1)
        valid = norms > 0.1
        points = points[valid]
        norms = norms[valid]
        if points.shape[0] == 0: return None
        dirs = points / norms[:, None]
        cos_angles = dirs @ ray
        mask = cos_angles > np.cos(np.deg2rad(angle_deg))
        candidates = points[mask]
        if candidates.shape[0] == 0: return None
        distances = np.linalg.norm(candidates, axis=1)
        return candidates[np.argmin(distances)]

    def detect_cones(self):
        # Agora o fsds.ImageRequest funcionará sem erro de NameError
        [img] = self.client.simGetImages(
            [fsds.ImageRequest("ZED_RGB", fsds.ImageType.Scene, False, True)],
            vehicle_name="FSCar"
        )
        image = np.frombuffer(img.image_data_uint8, dtype=np.uint8)
        image_bgr = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return []

        lidar = self.client.getLidarData(lidar_name="Lidar1", vehicle_name="FSCar")
        points = np.array(lidar.point_cloud, dtype=np.float32).reshape((-1, 3))

        results = self.model(image_bgr, verbose=False)
        result = results[0]

        detections_out = []
        for box in result.boxes:
            u, v, w, h = box.xywh[0].cpu().numpy()
            conf = float(box.conf[0])
            class_Id = int(box.cls[0])

            if conf < CONF_THRESHOLD:
                continue

            x_cam = (u - C_X) / F_X
            y_cam = (v - C_Y) / F_Y
            ray_cam = np.array([1.0, x_cam, y_cam], dtype=np.float32)
            ray_cam /= np.linalg.norm(ray_cam)

            hit = self.ray_lidar_intersection(points, ray_cam, angle_deg=ANGLE_DEG)

            if hit is not None:
                hit[2] -= CAMERA_HEIGHT
                x_depth = float(hit[0])
                y_lateral = float(hit[1])
                detections_out.append([y_lateral, x_depth, class_Id])

        return np.array(detections_out)