import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

fsds_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
if fsds_path not in sys.path: sys.path.insert(0, fsds_path)
import fsds 

FX, CX = 512.0, 512.0
FY, CY = 512.0, 360.0
CONF_THRESHOLD = 0.8
ANGLE_MATCH = 30.0 

class PerceptionSystem:
    def __init__(self):
        print("Iniciando Percepção...")
        self.client = fsds.FSDSClient()
        self.client.confirmConnection()
        
        self.model = YOLO("/home/pedro/mapping_eracing/16_01.pt")

    def get_lidar_match(self, points, u, v):
        """
        Esta função substitui o simples 'depthMatrix[v,u]'.
        Ela projeta um raio do pixel (u,v) e acha o ponto LiDAR mais próximo.
        """
        #raio (direção) que sai da câmera pelo pixel
        ray = np.array([1.0, (u - CX)/FX, (v - CY)/FY])
        ray /= np.linalg.norm(ray)

        #filtra pontos do LiDAR
        norms = np.linalg.norm(points, axis=1)
        valid_points = points[norms > 0.1]
        if valid_points.size == 0: return None

        #ponto LiDAR mais alinhado com o raio
        dirs = valid_points / np.linalg.norm(valid_points, axis=1)[:, None]
        cos_angles = dirs @ ray
        
        #pontos dentro de um ângulo de 30 graus
        mask = cos_angles > np.cos(np.deg2rad(ANGLE_MATCH))
        candidates = valid_points[mask]

        if candidates.size == 0: return None
        
        # ponto mais próximo (X_frente, Y_lateral, Z_altura)
        return candidates[np.argmin(np.linalg.norm(candidates, axis=1))]

    def detect_cones(self):
        [img] = self.client.simGetImages([fsds.ImageRequest("ZED_RGB", fsds.ImageType.Scene)])
        image = cv2.imdecode(np.frombuffer(img.image_data_uint8, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        lidar_data = self.client.getLidarData(lidar_name="Lidar1")
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape((-1, 3))

        if image is None: return np.array([])
            
        results = self.model(image, verbose=False)
        detections_3d = []

        #itera sobre as detecções
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD: continue

            #centro do cone (u, v) e a classe
            u, v = float(box.xywh[0][0]), float(box.xywh[0][1])
            class_id = int(box.cls[0])

            #achar a distância no LiDAR
            hit = self.get_lidar_match(points, u, v)

            if hit is not None:
                x_frente = float(hit[0])   #profundidade z
                y_lateral = float(hit[1])  # Esquerda/Direita label
                
                if x_frente < 30.0: #horizonte de 30 metros
                    detections_3d.append([y_lateral, x_frente, class_id])

        return np.array(detections_3d)
