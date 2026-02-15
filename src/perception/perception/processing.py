import rclpy 
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch

# =========================================================
# CONFIGURAÇÕES DE CÂMERA (Baseadas na ZED 720p)
# =========================================================
FX, FY = 640.0, 640.0
CX, CY = 640.0, 360.0
CONF_THRESHOLD = 0.85 

class PerceptionNode(Node): 
    def __init__(self):
        super().__init__('perception_node') 
        
        # Subscribers e Publishers
        self.imageReceiver = self.create_subscription(Image, '/fsds/camera/ZED_RGB', self.imageCallback, 10)
        self.depthReceiver = self.create_subscription(Image, '/fsds/camera/ZED_Depth', self.depthCallback, 10)
        self.perceptionOutput = self.create_publisher(Float32MultiArray, '/cones', 10)
        self.yoloDebug = self.create_publisher(Image, '/perception/debug', 10)

        # Carregamento do Modelo
        model_path = "/home/pedro/mapping_eracing/16_01.pt" 
        self.get_logger().info(f'Carregando modelo PyTorch: {model_path}')
        
        try:
            self.model = YOLO(model_path)
            if torch.cuda.is_available():
                self.model.to('cuda')
                self.get_logger().info(f'GPU ATIVA: {torch.cuda.get_device_name(0)}')
        except Exception as e:
            self.get_logger().error(f'Erro ao carregar modelo: {e}')

        self.bridge = CvBridge()
        self.latest_depth_img = None

    def depthCallback(self, msg):
        self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def imageCallback(self, msg):
        if self.latest_depth_img is None: return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            results = self.model.predict(source=frame, conf=CONF_THRESHOLD, device='cuda', verbose=False)
            
            r = results[0]
            self.yoloDebug.publish(self.bridge.cv2_to_imgmsg(r.plot(), encoding='bgr8'))

            boxes = r.boxes.xywh.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            
            raw_detections = []
            h_img, w_img = self.latest_depth_img.shape

            for (x, y, w, h), label in zip(boxes, classes):
                u, v = int(x), int(y)
                if 0 <= u < w_img and 0 <= v < h_img:
                    Z = self.latest_depth_img[v, u]
                    if not np.isnan(Z) and 0.5 < Z < 40.0:
                        X = (u - CX) * Z / FX
                        raw_detections.append({'pos': [float(X), float(Z)], 'id': int(label)})

            # --- CORREÇÃO DO SPATIAL NMS ---
            # Mantemos como lista de listas para a comparação funcionar
            filtered_cones = []
            for det in raw_detections:
                is_duplicate = False
                for accepted in filtered_cones:
                    # Agora 'accepted' é [X, Z, ID], então [:2] funciona!
                    dist = np.linalg.norm(np.array(det['pos']) - np.array(accepted[:2]))
                    if dist < 1.2:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered_cones.append([det['pos'][0], det['pos'][1], float(det['id'])])

            if filtered_cones:
                out_msg = Float32MultiArray()
                # Achata a lista apenas no momento de enviar a mensagem
                out_msg.data = np.array(filtered_cones).flatten().tolist()
                self.perceptionOutput.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'Erro Percepção: {e}')

def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()