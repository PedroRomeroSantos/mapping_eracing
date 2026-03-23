import os
import sys
import math
import airsim

fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
sys.path.insert(0, fsds_lib_path)

import fsds
import msgpack as _msgpack

_OrigUnpacker = _msgpack.Unpacker
class _BigUnpacker(_OrigUnpacker):
    def __init__(self, *args, **kwargs):
        for k in ("max_bin_len", "max_str_len", "max_array_len"):
            kwargs[k] = 16 * 1024 * 1024
        super().__init__(*args, **kwargs)
_msgpack.Unpacker = _BigUnpacker

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from perception    import PerceptionModule
from path_local    import PathPlanner
from pure_pursuit  import PurePursuitController

MODEL_PATH = "/home/pedro/mapping_eracing/16_01.pt"
VEHICLE    = "FSCar"

def waypoint_mais_proximo(waypoints):
    """Waypoint com menor profundidade, mensagem Point para ROS2."""
    if len(waypoints) == 0:
        return None
    idx = waypoints[:, 1].argmin()
    msg = Point()
    msg.x = float(waypoints[idx, 0])
    msg.y = 0.0
    msg.z = float(waypoints[idx, 1])
    return msg
  
def velocidade_atual(car_state) -> float:
    """Velocidade escalar em m/s a partir do estado do carro."""
    v = car_state.kinematics_estimated.linear_velocity
    return math.sqrt(v.x_val**2 + v.y_val**2 + v.z_val**2)

def main():
    rclpy.init()
    node = Node("waypoint_publisher")
  
    pub_wp   = node.create_publisher(Point,            "/waypoint_go",       10)
    pub_img  = node.create_publisher(Image,            "/camera/yolo_debug", 10)
    pub_acc  = node.create_publisher(Float32,          "car_acceleration",    1)
    pub_ori  = node.create_publisher(Float32MultiArray,"car_orientation",     1)
    bridge   = CvBridge()
  
    client = fsds.FSDSClient()
    client.confirmConnection()
    client.enableApiControl(True, VEHICLE)  

    percepcao  = PerceptionModule(client=client, model_path=MODEL_PATH)
    planner    = PathPlanner(visualizar=True)
    controller = PurePursuitController(client=client, vehicle_name=VEHICLE)

    print("main ON — controle ativo\n")
    while True:
      
        car_state = client.getCarState(VEHICLE)
        speed     = velocidade_atual(car_state)
        cones, debug_img = percepcao.capturar_e_processar()

        if debug_img is not None:
            pub_img.publish(bridge.cv2_to_imgmsg(debug_img, encoding="bgr8"))

        waypoints = planner.calcular_trajetoria(cones)
        controls = controller.step(waypoints, speed)

        print(
            f"[PP] v={speed:5.2f} m/s | "
            f"steer={controls.steering:+.3f} | "
            f"thr={controls.throttle:.2f} | "
            f"brk={controls.brake:.2f} | "
            f"wps={len(waypoints)}"
        )
      
        wp = waypoint_mais_proximo(waypoints)
        if wp is not None:
            pub_wp.publish(wp)

        rclpy.spin_once(node, timeout_sec=0)

        imu = client.getImuData(imu_name="Imu", vehicle_name=VEHICLE)
        pitch, roll, yaw = airsim.to_eularian_angles(imu.orientation)

        acc_msg = Float32()
        ori_msg = Float32MultiArray()
        acc_msg.data = imu.linear_acceleration.x_val
        ori_msg.data = [pitch, roll, yaw]
        pub_acc.publish(acc_msg)
        pub_ori.publish(ori_msg)

if __name__ == "__main__":
    main()
