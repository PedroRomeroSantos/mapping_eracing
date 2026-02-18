import rclpy 
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch

FX, CX = 640.0, 640.0
CONF_THRESHOLD = 0.85 

class PerceptionNode(Node): 
    def __init__(self):
        super().__init__('perception_node') 
        
        self.sub_img = self.create_subscription(Image, '/fsds/camera/ZED_RGB', self.img_cb, 10)
        self.sub_depth = self.create_subscription(Image, '/fsds/camera/ZED_Depth', self.depth_cb, 10)
        self.pub_cones = self.create_publisher(Float32MultiArray, '/cones', 10)
        self.pub_debug = self.create_publisher(Image, '/perception/debug', 10)

        self.model = YOLO("/home/pedro/mapping_eracing/16_01.pt")
        if torch.cuda.is_available():
            self.model.to('cuda')

        self.bridge = CvBridge()
        self.depth_img = None

    def depth_cb(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def img_cb(self, msg):
        if self.depth_img is None: return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            results = self.model(frame, conf=CONF_THRESHOLD, verbose=False)[0] #não floda o terminal com aviso
            
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(results.plot(), 'bgr8'))

            boxes = results.boxes.xywh.cpu().numpy()
            labels = results.boxes.cls.cpu().numpy()
            
            raw_cones = []
            h, w = self.depth_img.shape

            for (x, y, _, _), label in zip(boxes, labels):
                u, v = int(x), int(y)

                if 0 <= u < w and 0 <= v < h:
                    Z = self.depth_img[v, u]
                    
                    if not np.isnan(Z) and 0.5 < Z < 35.0:
                        X = (u - CX) * Z / FX
                        raw_cones.append([float(X), float(Z), float(label)])

            #filtro de duplicatas 
            final_cones = []
            for c in raw_cones:
                #só adiciona se não tem vizinho perto (1.2m) na lista final
                if not any(np.linalg.norm(np.array(c[:2]) - np.array(f[:2])) < 1.2 for f in final_cones):
                    final_cones.append(c) #adiciona X, Z, ID linearmente

            if final_cones:
                msg_out = Float32MultiArray()
                msg_out.data = np.array(final_cones).flatten().tolist()
                self.pub_cones.publish(msg_out)

        except Exception as e:
            self.get_logger().error(f'Erro: {e}')

def main():
    rclpy.init()
    node = PerceptionNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()