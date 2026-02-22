import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

class GlobalController(Node):
    def __init__(self):
        super().__init__('global_controller')
        self.sub_gps = self.create_subscription(Float32MultiArray, '/carGPS', self.gps_cb, 10)
        self.sub_wps = self.create_subscription(Float32MultiArray, '/waypoints', self.wps_cb, 10)
        self.pub_cmd = self.create_publisher(Float32MultiArray, '/cmd_bridge', 10)
        
        self.car_x = 0.0
        self.car_z = 0.0
        self.look_ahead = 6.0
        self.kp_steer = 1.8
        self.throttle = 0.28

    def gps_cb(self, msg):
        if len(msg.data) >= 3:
            self.car_x = msg.data[0]
            self.car_z = msg.data[2]

    def wps_cb(self, msg):
        try:
            wps = np.array(msg.data).reshape(-1, 2)
        except:
            return

        if len(wps) == 0:
            self.send_controls(0.0, 0.0, 0.5)
            return

        dists = np.sqrt((wps[:, 0] - self.car_x)**2 + (wps[:, 1] - self.car_z)**2)
        target_idx = np.argmin(np.abs(dists - self.look_ahead))
        target = wps[target_idx]

        lx = target[0] - self.car_x
        lz = target[1] - self.car_z

        dist_sq = lx**2 + lz**2
        if dist_sq > 0.5:
            steer = (2 * lx) / dist_sq
            steer_cmd = np.clip(steer * self.kp_steer, -1.0, 1.0)
            self.send_controls(float(steer_cmd), self.throttle, 0.0)
        else:
            self.send_controls(0.0, 0.0, 0.2)

    def send_controls(self, steer, throttle, brake):
        msg = Float32MultiArray()
        msg.data = [steer, throttle, brake]
        self.pub_cmd.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = GlobalController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()