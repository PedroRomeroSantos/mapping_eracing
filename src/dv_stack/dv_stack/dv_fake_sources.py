#!/usr/bin/env python3
import math, numpy as np, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from visualization_msgs.msg import Marker, MarkerArray

def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw/2.0); q.w = math.cos(yaw/2.0)
    return q

def pista_elipse(a=30.0, b=15.0, W=6.0, n=300):
    """
    Gera uma pista elíptica fechada.
    a = semi-eixo X
    b = semi-eixo Y
    W = largura da pista
    n = número de pontos
    """
    t = np.linspace(0, 2*math.pi, n, endpoint=False)
    cx = a*np.cos(t)
    cy = b*np.sin(t)

    # tangentes e normais
    dx, dy = np.gradient(cx), np.gradient(cy)
    tvec = np.stack([dx, dy],1)
    tvec /= np.maximum(np.linalg.norm(tvec, axis=1, keepdims=True), 1e-9)
    nL = np.stack([-tvec[:,1], tvec[:,0]],1)

    left  = np.stack([cx,cy],1)+(W/2.0)*nL
    right = np.stack([cx,cy],1)-(W/2.0)*nL
    return np.stack([cx,cy],1), left, right

class FakeTrack(Node):
    def __init__(self):
        super().__init__('fake_track')
        self.traj, self.left, self.right = pista_elipse()
        self.i = 0

        self.pub_odom = self.create_publisher(Odometry,'/odom',10)
        self.pub_map  = self.create_publisher(MarkerArray,'/map_cones',10)

        # timers
        self.timer_pose = self.create_timer(0.05, self.tick)        # carro se move
        self.timer_map  = self.create_timer(1.0, self.publish_map)  # cones 1 Hz

    def publish_map(self):
        msg = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        msg.markers.append(clear)

        mid = 0
        for arr, color in [(self.left,(0.0,0.0,1.0)), (self.right,(1.0,1.0,0.0))]:
            for x,y in arr:
                m = Marker()
                m.header.frame_id='map'
                m.header.stamp=self.get_clock().now().to_msg()
                m.ns='cones'; m.id=mid; mid+=1
                m.type=Marker.SPHERE; m.action=Marker.ADD
                m.pose.position.x=float(x); m.pose.position.y=float(y); m.pose.position.z=0.0
                m.scale.x=m.scale.y=m.scale.z=0.4
                m.color.r=float(color[0]); m.color.g=float(color[1]); m.color.b=float(color[2]); m.color.a=1.0
                msg.markers.append(m)

        self.pub_map.publish(msg)

    def tick(self):
        # carro avança na elipse
        self.i = (self.i+1)%len(self.traj)
        x,y = self.traj[self.i]
        x2,y2 = self.traj[(self.i+1)%len(self.traj)]
        yaw = math.atan2(y2-y, x2-x)

        od = Odometry()
        od.header.frame_id='map'; od.child_frame_id='base_link'
        od.header.stamp=self.get_clock().now().to_msg()
        od.pose.pose.position.x=float(x); od.pose.pose.position.y=float(y)
        od.pose.pose.orientation=yaw_to_quat(yaw)
        od.twist.twist.linear.x=3.0
        self.pub_odom.publish(od)

def main():
    rclpy.init()
    node=FakeTrack()
    rclpy.spin(node)
    node.destroy_node(); rclpy.shutdown()

if __name__=="__main__":
    main()





