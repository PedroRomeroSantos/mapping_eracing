import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2
from cv_bridge import CvBridge
import torch

# Parâmetros da câmera
fx = 960
fy = 540
cx = 960
cy = 540

COLOR_THRESHOLD = 0.75

def positionMap(pixelU:int, pixelV:int, distance:float):
    """Converte coordenadas do pixel + distância para coordenadas 3D na câmera."""
    bottomPart = np.sqrt(1 + ((pixelU-cx)**2)/(fx**2) + ((pixelV-cy)**2)/(fy**2))
    z = distance / bottomPart
    x = (pixelU - cx) * (z / fx)
    y = (pixelV - cy) * (z / fy)
    return x, y, z

class Processing(Node):
    def __init__(self):
        super().__init__('processing')

        # Subscribers
        self.image_sub = self.create_subscription(Image, '/fsds/camera/ZED2iImage', self.imageCallback, 1)
        self.depth_sub = self.create_subscription(Image, '/fsds/camera/ZED2iDepth', self.depthCallback, 1)

        # Publisher
        self.cones_pub = self.create_publisher(Float32MultiArray, '/cones_depth', 10)

        # YOLOv5
        self.model = torch.hub.load(
            '/home/pedro/perception_ws/yolov5',  #caminho da pasta YOLOv5 com hubconf.py
            'custom',
            path='/home/pedro/perception_ws/yolov5/1280.pt',  #caminho do arquivo .pt
            source='local'
        )


        self.bridge = CvBridge()

     
        self.cones = []          
        self.conesDepth = []     

        self.get_logger().info('Processing node has been started.')

    def imageCallback(self, image_msg):
        """Recebe imagem RGB, detecta cones e desenha pontos nos centros detectados."""
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        results = self.model(frame)
        
        detections = results.xyxy[0]
        self.cones = []  

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            mid_x = int((x1.item() + x2.item()) / 2)
            mid_y = int((y1.item() + y2.item()) / 2)

            if abs(conf) > COLOR_THRESHOLD:
                self.cones.append([mid_x, mid_y])


    def depthCallback(self, depth_msg):
        """Recebe imagem de profundidade e calcula coordenadas 3D dos cones."""

        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        self.conesDepth = []

        for mid_x, mid_y in self.cones:
            u = int(mid_x)
            v = int(mid_y)
            distance = depth_image[v, u] 
            x, y, z = positionMap(u, v, distance)
            self.conesDepth.append([x, z])  # só x, z + cor/confa

        msg = Float32MultiArray()
        msg.data = np.array(self.conesDepth, dtype=np.float32).flatten().tolist()
        self.cones_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = Processing()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
