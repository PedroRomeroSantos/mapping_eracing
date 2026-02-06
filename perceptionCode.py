import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

import time

# ==============================
# PARAMETROS DA CAMERA
# ==============================
C_X = 512.0   # centro em pixels
C_Y = 360.0
F_X = 512.0   # focal em pixels
F_Y = 512.0

CONF_THRESHOLD = 0.8

CAMERA_HEIGHT = 0.0 # metros (camera acima do LiDAR)
ANGLE_DEG = 30.0      # abertura angular para associação LiDAR

# ==============================
# FSDS SETUP
# ==============================
fsds_lib_path = os.path.join(
    os.path.expanduser("~"),
    "Formula-Student-Driverless-Simulator",
    "python"
)
sys.path.insert(0, fsds_lib_path)
import fsds
client = fsds.FSDSClient()
client.confirmConnection()

# ==============================
# YOLO
# ==============================
model = YOLO(
    "/home/felipe_capovilla/Documents/E-Racing/Perception/Modelos/16_01.engine"
)

# ==============================
# FUNCAO: INTERSECAO RAY + LIDAR
# ==============================
def ray_lidar_intersection(points, ray, angle_deg=1.0):
    """
    points: Nx3 LiDAR points (x=frente, y=lateral, z=altura)
    ray: vetor unitario 3D
    """

    if points.shape[0] == 0:
        return None

    norms = np.linalg.norm(points, axis=1)
    valid = norms > 0.1
    points = points[valid]
    norms = norms[valid]

    if points.shape[0] == 0:
        return None

    dirs = points / norms[:, None]
    cos_angles = dirs @ ray

    mask = cos_angles > np.cos(np.deg2rad(angle_deg))
    candidates = points[mask]

    if candidates.shape[0] == 0:
        return None

    distances = np.linalg.norm(candidates, axis=1)
    return candidates[np.argmin(distances)]


# ==============================
# LOOP PRINCIPAL
# ==============================
while True:


    # -------- CAMERA RGB --------
    [img] = client.simGetImages(
        [fsds.ImageRequest(
            camera_name="ZED_RGB",
            image_type=fsds.ImageType.Scene,
            pixels_as_float=False,
            compress=True
        )],
        vehicle_name="FSCar"
    )

    image = np.frombuffer(img.image_data_uint8, dtype=np.uint8)
    image_bgr = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if image_bgr is None:
        print("Erro ao decodificar imagem")
        continue

    # -------- LIDAR --------
    lidar = client.getLidarData(
        lidar_name="Lidar1",
        vehicle_name="FSCar"
    )

    points = np.array(lidar.point_cloud, dtype=np.float32)
    points = points.reshape((-1, 3))

    # -------- YOLO --------
    results = model(image_bgr, verbose=False)
    result = results[0]

    detections_out = []

    for box in result.boxes:
        u, v, w, h = box.xywh[0].cpu().numpy()
        conf = float(box.conf[0])
        class_Id = int(box.cls[0])



        # ----- Pixel -> raio camera -----
        x_cam = (u - C_X) / F_X
        y_cam = (v - C_Y) / F_Y

        # Frame: x=frente, y=lateral, z=altura
        ray_cam = np.array([1.0, x_cam, y_cam], dtype=np.float32)
        ray_cam /= np.linalg.norm(ray_cam)

        # ----- Sem rotacao -----
        ray_lidar = ray_cam

        # ----- Interseção -----
        hit = ray_lidar_intersection(
            points,
            ray_lidar,
            angle_deg=ANGLE_DEG
        )

        if hit is None:
            continue

        # ----- Corrigir altura -----
        hit[2] -= CAMERA_HEIGHT

        x_depth = float(hit[0])
        y_lateral = float(hit[1])

        if(conf > CONF_THRESHOLD):
            detections_out.append([y_lateral, x_depth, class_Id])

        # ----- DEBUG VISUAL -----
        cv2.circle(image_bgr, (int(u), int(v)), 5, (0, 255, 0), -1)
        cv2.putText(
            image_bgr,
            f"x={x_depth:.1f} y={y_lateral:.1f}",
            (int(u), int(v) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1
        )

    # -------- OUTPUT --------
    print(detections_out)

    # -------- VISUAL --------
    #cv2.imshow("Camera + YOLO", result.plot())

    if cv2.waitKey(1) & 0xFF == 27:
        break


cv2.destroyAllWindows()
