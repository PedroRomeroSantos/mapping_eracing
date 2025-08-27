#!/usr/bin/env python3
import math, numpy as np, rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion

def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw/2.0); q.w = math.cos(yaw/2.0)
    return q

def rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s],[s, c]], float)

# ---------- Geradores ----------

def pista_minhoca(L=100.0, n=200, W=4.0, amp=10.0, n_voltas=4,
                  raio_loop=15.0, n_circ=60):
    """
    Linha central: serpenteia no eixo X (senóide) e termina
    com um arco circular para voltar pro início.
    """
    # serpente
    xs = np.linspace(0, L, n)
    ys = amp * np.sin(xs * (n_voltas*math.pi/L))
    cx, cy = xs, ys

    # arco circular de retorno
    th = np.linspace(0, math.pi, n_circ)  # meia-volta
    circ_x = raio_loop*np.cos(th) + L
    circ_y = raio_loop*np.sin(th) + 0.0
    cx = np.concatenate([cx, circ_x])
    cy = np.concatenate([cy, circ_y])

    # tangente -> normais -> bordas
    dx = np.gradient(cx); dy = np.gradient(cy)
    t = np.stack([dx, dy],1); t /= np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-9)
    nL = np.stack([-t[:,1], t[:,0]],1)

    left  = np.stack([cx, cy],1) + (W/2.0)*nL
    right = np.stack([cx, cy],1) - (W/2.0)*nL
    return left, right

def carregar_csv4_pairs(path):
    """CSV com 4 colunas: xL,yL,xR,yR (sem cabeçalho)."""
    arr = np.loadtxt(path, delimiter=',', dtype=float)
    if arr.ndim == 1: arr = arr.reshape(1,4)
    if arr.shape[1] != 4:
        raise RuntimeError(f"CSV precisa ter 4 colunas, tem {arr.shape[1]}")
    return arr[:,:2], arr[:,2:4]

# ---------- Nó fake ----------
class DVFakeSources(Node):
    def __init__(self):
        super().__init__('dv_fake_sources')

        # parâmetros principais
        self.declare_parameter('freq_hz',20.0)
        self.declare_parameter('tipo_pista','minhoca')  # minhoca ou csv4
        self.declare_parameter('comprimento_pista',100.0)
        self.declare_parameter('largura_pista',4.0)
        self.declare_parameter('csv_pairs','')  # caminho do csv 4-colunas

        # “sensor”/ruído
        self.declare_parameter('vel_media',4.0)
        self.declare_parameter('alcance_m',25.0)
        self.declare_parameter('fov_graus',120.0)
        self.declare_parameter('sigma_meas_cone',0.05)
        self.declare_parameter('prob_dropout',0.10)
        self.declare_parameter('sigma_odom_xy',0.03)
        self.declare_parameter('sigma_odom_yaw_graus',1.0)

        tipo = self.get_parameter('tipo_pista').value
        L = float(self.get_parameter('comprimento_pista').value)
        W = float(self.get_parameter('largura_pista').value)

        if tipo == 'csv4':
            path = str(self.get_parameter('csv_pairs').value)
            if not path:
                raise RuntimeError("tipo_pista=csv4 requer csv_pairs:=/caminho/track_pairs.csv")
            self.left_world, self.right_world = carregar_csv4_pairs(path)
        else:
            self.left_world, self.right_world = pista_minhoca(L=L,W=W)

        # estado
        self.v = float(self.get_parameter('vel_media').value)
        self.range = float(self.get_parameter('alcance_m').value)
        self.fov = math.radians(float(self.get_parameter('fov_graus').value))
        self.sig_meas = float(self.get_parameter('sigma_meas_cone').value)
        self.p_drop = float(self.get_parameter('prob_dropout').value)
        self.sig_odom_xy = float(self.get_parameter('sigma_odom_xy').value)
        self.sig_odom_yaw = math.radians(float(self.get_parameter('sigma_odom_yaw_graus').value))

        self.s = 0
        self.traj = 0.5*(self.left_world+self.right_world)  # linha central
        self.Ntraj = len(self.traj)

        # pubs
        self.pub_odom  = self.create_publisher(Odometry,'/odom',10)
        self.pub_cones = self.create_publisher(Float32MultiArray,'/cones',10)
        self.timer = self.create_timer(1.0/float(self.get_parameter('freq_hz').value),self.tick)

    def pose_true(self, i):
        i0, i1 = i%self.Ntraj, (i+1)%self.Ntraj
        p0, p1 = self.traj[i0], self.traj[i1]
        yaw = math.atan2(p1[1]-p0[1], p1[0]-p0[0])
        return p0[0], p0[1], yaw

    def tick(self):
        self.s = (self.s+1)%self.Ntraj
        x_true,y_true,yaw_true = self.pose_true(self.s)

        # publica odom com ruído
        odom = Odometry()
        now = self.get_clock().now().to_msg()
        odom.header.stamp = now; odom.header.frame_id='odom'; odom.child_frame_id='base_link'
        odom.pose.pose.position.x = x_true+np.random.normal(0,self.sig_odom_xy)
        odom.pose.pose.position.y = y_true+np.random.normal(0,self.sig_odom_xy)
        odom.pose.pose.orientation = yaw_to_quat(yaw_true+np.random.normal(0,self.sig_odom_yaw))
        odom.twist.twist.linear.x = self.v
        self.pub_odom.publish(odom)

        # publica cones no frame base_link
        Rt = rot(yaw_true).T
        car_pos = np.array([x_true,y_true])
        detections = []
        for side, arr in [(1.0,self.left_world),(2.0,self.right_world)]:
            for pw in arr:
                pb = Rt @ (pw-car_pos)
                xb,yb = float(pb[0]), float(pb[1])
                if xb<0.2 or xb>self.range: continue
                ang = math.atan2(yb,xb)
                if abs(ang)>self.fov/2: continue
                if np.random.rand()<self.p_drop: continue
                xb+=np.random.normal(0,self.sig_meas); yb+=np.random.normal(0,self.sig_meas)
                detections.extend([xb,yb,side])
        self.pub_cones.publish(Float32MultiArray(data=detections))

def main():
    rclpy.init()
    node = DVFakeSources()
    rclpy.spin(node)
    node.destroy_node(); rclpy.shutdown()

if __name__=='__main__':
    main()

