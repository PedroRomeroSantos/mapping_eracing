import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import torch

class YOLOTest(Node):
    def __init__(self):
        super().__init__('yolo_test_node')

        # Subscriber da câmera RGB
        self.subscription = self.create_subscription(
            Image,
            '/fsds/camera/ZED2iImage',
            self.listener_callback,
            1
        )
        self.get_logger().info("YOLO Test node has started.")

        # YOLOv5
        self.model = torch.hub.load(
            '/home/pedro/perception_ws/yolov5',  #caminho da pasta YOLOv5 com hubconf.py
            'custom',
            path='/home/pedro/perception_ws/yolov5/1280.pt',  #caminho do arquivo .pt
            source='local'
        )

        # Bridge para converter mensagens ROS → OpenCV
        self.bridge = CvBridge()

        # Janela única do OpenCV
        cv2.namedWindow("YOLO Detections", cv2.WINDOW_NORMAL)

        # Dicionário de labels customizados
        self.label_map = {
            0: "Cone Azul",
            1: "Cone Amarelo"
        }

    def listener_callback(self, msg):
        # Converte imagem ROS → OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Rodar YOLO
        results = self.model(frame)
        detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, cls]

        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            class_id = int(cls.item())

            label_text = 'CONE'

            # Cor diferente por classe
            color = (255, 0, 0) 

            # Desenha bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Desenha o ponto central do bounding box
            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)
            cv2.circle(frame, (mid_x, mid_y), 5, (0, 0, 255), -1)

        # Atualiza a mesma janela
        cv2.imshow("YOLO Detections", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YOLOTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
