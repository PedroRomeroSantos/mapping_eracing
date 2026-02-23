import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

fsds_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
if fsds_path not in sys.path: sys.path.insert(0, fsds_path)
import fsds 

FX, CX = 640.0, 640.0
CONF_THRESHOLD = 0.85
ANGLE_MATCH = 30.0 

class PerceptionSystem:
    def __init__(self):
        self.client = fsds.FSDSClient()
        self.client.confirmConnection()
        self.model = YOLO("/home/pedro/mapping_eracing/16_01.pt")

    def get_lidar_match(self, points, u, v):
        ray_2d = np.array([1.0, (u - CX)/FX])
        ray_2d /= np.linalg.norm(ray_2d)

        points_2d = points[:, :2]
        norms = np.linalg.norm(points_2d, axis=1)
        
        valid_mask = norms > 0.1
        valid_points = points[valid_mask]
        valid_points_2d = points_2d[valid_mask]
        valid_norms = norms[valid_mask]

        if valid_points.size == 0: return None

        dirs_2d = valid_points_2d / valid_norms[:, None]
        cos_angles = dirs_2d @ ray_2d
        
        mask = cos_angles > np.cos(np.deg2rad(ANGLE_MATCH))
        candidates = valid_points[mask]

        if candidates.size == 0: return None
    
        return candidates[np.argmin(np.linalg.norm(candidates[:, :2], axis=1))]

    def detect_cones(self):
        [img] = self.client.simGetImages([fsds.ImageRequest("ZED_RGB", fsds.ImageType.Scene)])
        image = cv2.imdecode(np.frombuffer(img.image_data_uint8, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        lidar_data = self.client.getLidarData(lidar_name="Lidar1")
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape((-1, 3))

        if image is None: return np.array([])
            
        results = self.model(image, verbose=False, conf=CONF_THRESHOLD)
        raw_detections = []

        for box in results[0].boxes:
            u, v = float(box.xywh[0][0]), float(box.xywh[0][1])
            class_id = int(box.cls[0])
            hit = self.get_lidar_match(points, u, v)

            if hit is not None:
                #LiDAR só para a profundidade
                x_frente = float(hit[0])
                
                #posição lateral
                y_lateral = (u - CX) * x_frente / FX
                
                if 0.5 < x_frente < 35.0:
                    raw_detections.append([y_lateral, x_frente, class_id])

        final_cones = []
        for c in raw_detections:
            if not any(np.linalg.norm(np.array(c[:2]) - np.array(f[:2])) < 1.2 for f in final_cones):
                final_cones.append(c)

        return np.array(final_cones)
