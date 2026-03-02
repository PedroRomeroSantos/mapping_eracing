import os
import sys
import airsim

fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
sys.path.insert(0, fsds_lib_path)


import fsds
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float32,Float32MultiArray
import msgpack as _msgpack
from perception import PerceptionModule
from path_local  import PathPlanner

# 1 MB para 16 MB, debug formato imagem impossível
_OrigUnpacker = _msgpack.Unpacker
class _BigUnpacker(_OrigUnpacker):
    def __init__(self, *args, **kwargs):
        kwargs['max_bin_len'] = 16 * 1024 * 1024
        kwargs['max_str_len'] = 16 * 1024 * 1024
        kwargs['max_array_len'] = 16 * 1024 * 1024
        super().__init__(*args, **kwargs)
_msgpack.Unpacker = _BigUnpacker


MODEL_PATH = "/home/pedro/mapping_eracing/16_01.pt"
VEHICLE    = "FSCar"

def waypoint_mais_proximo(waypoints):
    #o waypoint com menor profundidade e formata no tipo mensagem Point
    if len(waypoints) == 0:
        return None
    idx = waypoints[:, 1].argmin()
    msg = Point()
    msg.x = float(waypoints[idx, 0])   # lateral
    msg.y = 0.0                         # altura 
    msg.z = float(waypoints[idx, 1])   # profundidade
    return msg




def main():
    rclpy.init()
    node = Node("waypoint_publisher")
    
    acceleration = node.create_publisher(Float32,"car_acceleration",1)
    orientation = node.create_publisher(Float32MultiArray,'car_orientation',1)
    pub  = node.create_publisher(Point, "/waypoint_go", 10)
    

    client = fsds.FSDSClient()
    client.confirmConnection()

    percepcao = PerceptionModule(client=client, model_path=MODEL_PATH)
    planner   = PathPlanner(visualizar=True)

    print("main ON\n")

    while True:
            car_state = client.getCarState(VEHICLE)
            cones, _ = percepcao.capturar_e_processar()
            waypoints = planner.calcular_trajetoria(cones)

            wp = waypoint_mais_proximo(waypoints)
            if wp is not None:
                pub.publish(wp)
                rclpy.spin_once(node, timeout_sec=0)


            imuData = client.getImuData(imu_name='Imu',vehicle_name='FSCar')
            accMsg = Float32()
            orientationMsg = Float32MultiArray()

            (pitch,roll,yaw) = airsim.to_eularian_angles(imuData.orientation)
            orientationMsg.data = [pitch,roll,yaw]
            accMsg.data = imuData.linear_acceleration.x_val
            acceleration.publish(accMsg)
            orientation.publish(orientationMsg)


if __name__ == "__main__":
    main()