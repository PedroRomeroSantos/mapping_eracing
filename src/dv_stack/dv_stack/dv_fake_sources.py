#!/usr/bin/env python3
# dv_fake_sources.py
import math, numpy as np, rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion

# ---------- util ----------
def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion(); q.z = math.sin(yaw/2.0); q.w = math.cos(yaw/2.0); return q

def rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], float)

# ---------- geradores de pista (retornam (left,right) em mundo) ----------
def pista_S(L=120.0, n=80, W=4.0, amp=5.0, esc=15.0):
    xs = np.linspace(0.0, L, n)
    ys = amp * np.sin(xs/esc)
    dys = (amp/esc) * np.cos(xs/esc)
    t = np.stack([np.ones_like(xs), dys], 1)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    nL = np.stack([-t[:,1], t[:,0]], 1)
    left  = np.stack([xs, ys], 1) + (W/2.0)*nL
    right = np.stack([xs, ys], 1) - (W/2.0)*nL
    return left, right

def pista_reta(L=120.0, n=60, W=4.0):
    xs = np.linspace(0.0, L, n); ys = np.zeros_like(xs)
    nL = np.stack([np.zeros_like(xs), np.ones_like(xs)], 1)
    left  = np.stack([xs, ys], 1) + (W/2.0)*nL
    right = np.stack([xs, ys], 1) - (W/2.0)*nL
    return left, right

def pista_oval(raio_x=25.0, raio_y=15.0, voltas=1.0, n=140, W=4.0):
    th = np.linspace(0, 2*math.pi*voltas, n, endpoint=False)
    cx = raio_x * np.cos(th); cy = raio_y * np.sin(th)
    # tangente
    tx = -raio_x * np.sin(th); ty =  raio_y * np.cos(th)
    t = np.stack([tx, ty], 1); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    nL = np.stack([-t[:,1], t[:,0]], 1)
    left  = np.stack([cx, cy], 1) + (W/2.0)*nL
    right = np.stack([cx, cy], 1) - (W/2.0)*nL
    # desloca pra começar em x>=0
    off = np.array([raio_x, 0.0])
    return left + off, right + off

def pista_hairpin(L=80.0, n=90, W=4.0, curva=10.0):
    xs = np.linspace(0.0, L, n)
    ys = curva * np.tanh((xs - 0.5*L)/5.0)
    dys = (curva/5.0) * (1/np.cosh((xs - 0.5*L)/5.0))**2
    t = np.stack([np.ones_like(xs), dys], 1)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    nL = np.stack([-t[:,1], t[:,0]], 1)
    left  = np.stack([xs, ys], 1) + (W/2.0)*nL
    right = np.stack([xs, ys], 1) - (W/2.0)*nL
    return left, right

def pista_fig8(raio=20.0, n=160, W=4.0):
    th = np.linspace(0, 2*math.pi, n, endpoint=False)
    cx = raio*np.sin(th); cy = raio*np.sin(th)*np.cos(th)  # lemniscata simples
    # tangente
    dth = 2*math.pi/n
    tx = np.gradient(cx, dth); ty = np.gradient(cy, dth)
    t = np.stack([tx, ty], 1); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    nL = np.stack([-t[:,1], t[:,0]], 1)
    left  = np.stack([cx, cy], 1) + (W/2.0)*nL
    right = np.stack([cx, cy], 1) - (W/2.0)*nL
    off = np.array([raio, 0.0])
    return left + off, right + off

def pista_slalom(L=120.0, n=30, W=4.0, amp=4.0, esc=12.0):
    xs = np.linspace(0.0, L, n)
    ys = amp*np.sin(xs/esc)
    left  = np.stack([xs, ys + W/2.0], 1)
    right = np.stack([xs, ys - W/2.0], 1)
    return left, right

def carregar_csv(path):
    # espera arquivo com 2 colunas x,y sem cabeçalho
    arr = np.loadtxt(path, delimiter=',', dtype=float)
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1,2)
    return arr

# ---------- nó fake ----------
class DVFakeSources(Node):
    """
    Publica:
      - /odom (nav_msgs/Odometry) com ruído
      - /cones (Float32MultiArray): [x_b, y_b, lado]* com ruído, alcance e FOV
    """
    def __init__(self):
        super().__init__('dv_fake_sources')

        # parâmetros
        self.declare_parameter('freq_hz', 20.0)
        self.declare_parameter('tipo_pista', 'S')  # S, reta, oval, hairpin, slalom, fig8, csv
        self.declare_parameter('comprimento_pista', 120.0)
        self.declare_parameter('largura_pista', 4.0)
        self.declare_parameter('n_cones', 90)

        # oval/fig8 extras (opcionais)
        self.declare_parameter('oval_raio_x', 25.0)
        self.declare_parameter('oval_raio_y', 15.0)
        self.declare_parameter('fig8_raio', 20.0)

        # CSV
        self.declare_parameter('csv_left', '')
        self.declare_parameter('csv_right', '')

        # “sensor”/ruído
        self.declare_parameter('vel_media', 4.0)
        self.declare_parameter('alcance_m', 25.0)
        self.declare_parameter('fov_graus', 120.0)
        self.declare_parameter('sigma_meas_cone', 0.05)
        self.declare_parameter('prob_dropout', 0.10)
        self.declare_parameter('sigma_odom_xy', 0.03)
        self.declare_parameter('sigma_odom_yaw_graus', 1.0)

        self.freq = float(self.get_parameter('freq_hz').value)
        self.tipo = str(self.get_parameter('tipo_pista').value).lower()
        L = float(self.get_parameter('comprimento_pista').value)
        W = float(self.get_parameter('largura_pista').value)
        N = int(self.get_parameter('n_cones').value)

        # gerar pistas
        if self.tipo == 'reta':
            self.left_world, self.right_world = pista_reta(L=L, n=max(20, N//2), W=W)
        elif self.tipo == 'oval':
            rx = float(self.get_parameter('oval_raio_x').value)
            ry = float(self.get_parameter('oval_raio_y').value)
            self.left_world, self.right_world = pista_oval(raio_x=rx, raio_y=ry, n=max(40, N), W=W)
        elif self.tipo == 'hairpin':
            self.left_world, self.right_world = pista_hairpin(L=L, n=max(40, N), W=W)
        elif self.tipo == 'slalom':
            self.left_world, self.right_world = pista_slalom(L=L, n=max(20, N//3), W=W)
        elif self.tipo == 'fig8':
            r = float(self.get_parameter('fig8_raio').value)
            self.left_world, self.right_world = pista_fig8(raio=r, n=max(60, N), W=W)
        elif self.tipo == 'csv':
            pL = str(self.get_parameter('csv_left').value)
            pR = str(self.get_parameter('csv_right').value)
            if not pL or not pR:
                raise RuntimeError("tipo_pista=csv requer csv_left e csv_right.")
            self.left_world  = carregar_csv(pL)
            self.right_world = carregar_csv(pR)
        else:
            self.left_world, self.right_world = pista_S(L=L, n=max(40, N//2), W=W)

        # estado "verdade" no centro da pista S (para orientar o carro)
        # para todas as pistas, caminhamos ao longo de x ou do parâmetro s (aproximação)
        self.s = 0.0
        self.L = L
        self.v = float(self.get_parameter('vel_media').value)
        self.range = float(self.get_parameter('alcance_m').value)
        self.fov = math.radians(float(self.get_parameter('fov_graus').value))
        self.sig_meas = float(self.get_parameter('sigma_meas_cone').value)
        self.p_drop = float(self.get_parameter('prob_dropout').value)
        self.sig_odom_xy = float(self.get_parameter('sigma_odom_xy').value)
        self.sig_odom_yaw = math.radians(float(self.get_parameter('sigma_odom_yaw_graus').value))

        # publishers
        self.pub_odom  = self.create_publisher(Odometry, '/odom', 10)
        self.pub_cones = self.create_publisher(Float32MultiArray, '/cones', 10)

        self.timer = self.create_timer(1.0/self.freq, self.tick)
        self.last_time = self.get_clock().now()

        self.get_logger().info(f"Fake: tipo_pista={self.tipo}, cones: L={len(self.left_world)} R={len(self.right_world)}")

    # --- orientação do carro ao longo do "eixo" (aproximações por tipo) ---
    def pose_true_from_s(self, s: float):
        if self.tipo in ('reta', 'slalom'):
            x = s; y = 0.0 if self.tipo=='reta' else 4.0*math.sin(s/12.0)
            yaw = math.atan2((0.0 if self.tipo=='reta' else (4.0/12.0)*math.cos(s/12.0)), 1.0)
            return x, y, yaw
        if self.tipo == 'hairpin':
            y = 10.0*math.tanh((s - 0.5*self.L)/5.0); x = s
            dy = (10.0/5.0)*(1/math.cosh((s - 0.5*self.L)/5.0))**2
            yaw = math.atan2(dy, 1.0); return x, y, yaw
        if self.tipo == 'oval':
            # percorre oval em ângulo
            rx = float(self.get_parameter('oval_raio_x').value)
            ry = float(self.get_parameter('oval_raio_y').value)
            th = (s / self.L) * 2*math.pi
            x = rx*math.cos(th) + rx
            y = ry*math.sin(th)
            tx = -rx*math.sin(th); ty =  ry*math.cos(th)
            yaw = math.atan2(ty, tx); return x, y, yaw
        if self.tipo == 'fig8':
            r = float(self.get_parameter('fig8_raio').value)
            th = (s / self.L) * 2*math.pi
            x = r*math.sin(th) + r
            y = r*math.sin(th)*math.cos(th)
            # derivadas
            dth = 2*math.pi/self.L
            tx = r*math.cos(th); ty = r*(math.cos(2*th)-math.sin(2*th))/2.0
            yaw = math.atan2(ty, tx); return x, y, yaw
        # padrão S
        amp, esc = 5.0, 15.0
        x = s; y = amp*math.sin(s/esc)
        yaw = math.atan2((amp/esc)*math.cos(s/esc), 1.0)
        return x, y, yaw

    # ----------------- laço principal -----------------
    def tick(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        if dt <= 0.0:
            return
        self.last_time = now

        # avança e faz loop
        self.s = (self.s + self.v*dt) % self.L
        x_true, y_true, yaw_true = self.pose_true_from_s(self.s)

        # 1) publica /odom com ruído
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        x_meas = x_true + np.random.normal(0.0, self.sig_odom_xy)
        y_meas = y_true + np.random.normal(0.0, self.sig_odom_xy)
        yaw_meas = yaw_true + np.random.normal(0.0, self.sig_odom_yaw)
        odom.pose.pose.position.x = float(x_meas)
        odom.pose.pose.position.y = float(y_meas)
        odom.pose.pose.orientation = yaw_to_quat(float(yaw_meas))
        odom.twist.twist.linear.x = float(self.v)
        self.pub_odom.publish(odom)

        # 2) gera /cones a partir de left/right no mundo
        car_pos = np.array([x_true, y_true], float)
        Rt = rot(yaw_true).T  # mundo->base
        detections = []
        for side, arr in [(1.0, self.left_world), (2.0, self.right_world)]:
            for pw in arr:
                pb = Rt @ (pw - car_pos)
                xb, yb = float(pb[0]), float(pb[1])
                # alcance + FOV frontal
                if xb < 0.2 or xb > self.range:
                    continue
                ang = math.atan2(yb, xb)
                if abs(ang) > self.fov/2.0:
                    continue
                if np.random.rand() < self.p_drop:
                    continue
                xb += np.random.normal(0.0, self.sig_meas)
                yb += np.random.normal(0.0, self.sig_meas)
                detections.extend([xb, yb, side])
        self.pub_cones.publish(Float32MultiArray(data=np.array(detections, dtype=np.float32).tolist()))

def main():
    rclpy.init()
    node = DVFakeSources()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
