import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO

fx = 960
fy = 540
cx = 960
cy = 540

COLOR_THRESHOLD = 0.75 

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('PerceptionNode')
        self.imageReceiver = self.create_subscription(Image,'/fsds/camera/ZED2iImage',self.imageCallback,10)
        self.depthReceiver = self.create_subscription(Image,'/fsds/camera/ZED2iDepth',self.depthCallback,10)

        self.yoloDebug = self.create_publisher(Image,'yoloDebug',10)
        self.perceptionOutput = self.create_publisher(Float32MultiArray,'cones',10)

        self.model = YOLO('/home/felipe-capovilla/Documents/best.onnx')

        self.bridge = CvBridge()
        self.cones=[]
        self.cones3D=[]


    
    def imageCallback(self, msg):
   
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    
        results = self.model(frame)
        r = results[0]  

      
        imgDebug = r.plot()

   
        imgDebug_msg = self.bridge.cv2_to_imgmsg(imgDebug, encoding='bgr8')
        self.yoloDebug.publish(imgDebug_msg)

    
        detections = r.boxes.xywh.cpu().numpy()    
        classes = r.boxes.cls.cpu().numpy()      
        confs = r.boxes.conf.cpu().numpy()        

   
        self.cones.clear()

    
        for (xCentro, yCentro, w, h), conf, label in zip(detections, confs, classes):
            if conf < COLOR_THRESHOLD:
                continue

            self.cones.append([float(xCentro), float(yCentro), int(label)])
            
        
    def depthCallback(self, msg):
        depthMatrix = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

        cones3D_map = []

        for cone in self.cones:
            u = int(cone[0])  
            v = int(cone[1])  
            label = cone[2]

            
            Z = depthMatrix[v, u]

            
            X_c = (u - cx) * Z / fx
            Y_c = 0  
            Z_c = Z

        
            X_w = X_c
            Z_w = Z_c
            if(Z_w > 30):
                continue
            cones3D_map.append([X_w, Z_w, label])

    
        msg_out = Float32MultiArray()
        msg_out.data = np.array(cones3D_map, dtype=np.float32).flatten().tolist()
        self.perceptionOutput.publish(msg_out)

    
        self.cones.clear()
        self.cones3D = cones3D_map




def main():
    rclpy.init()
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__=="__main__":
    main()