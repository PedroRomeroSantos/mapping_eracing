#!/usr/bin/env python3
# dv_planner_node.py
import math
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray


# ===================== utilidades geométricas =====================
def _rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=float)

def carro_para_mundo(pts_body: np.ndarray, pose_xyz: np.ndarray) -> np.ndarray:
    """Converte pontos no frame do carro (x para frente, y para esquerda) para mundo (map)."""
    x, y, yaw = pose_xyz
    if pts_body.size == 0:
        return np.empty((0, 2), dtype=float)
    return (pts_body @ _rot(yaw).T) + np.array([x, y], dtype=float)

# ===================== 1) mapa incremental de cones =====================
class MapaCones:
    """
    Guarda marcos (cones) no mundo e os atualiza com suavização quando são vistos de novo.
    Cada marco: [x, y, lado, contagem]
    """
    def __init__(self, raio_assoc=1.0, alpha_suave=0.3):
        self.raio = float(raio_assoc)
        self.alpha = float(alpha_suave)
        self.marcos = np.empty((0, 4), dtype=float)  # x,y,lado,contagem

    def _associar(self, p: np.ndarray, lado: int) -> int:
        if len(self.marcos) == 0:
            return -1
        mask = (self.marcos[:, 2] == lado)
        if not np.any(mask):
            return -1
        cand = self.marcos[mask]
        d = np.hypot(cand[:, 0] - p[0], cand[:, 1] - p[1])
        j = int(np.argmin(d))
        if d[j] <= self.raio:
            return np.where(mask)[0][j]
        return -1

    def atualizar(self, cones_mundo: np.ndarray, lados: np.ndarray):
        for p, lado in zip(cones_mundo, lados):
            idx = self._associar(p, int(lado))
            if idx >= 0:
                # média exponencial para estabilizar
                self.marcos[idx, 0] = (1 - self.alpha) * self.marcos[idx, 0] + self.alpha * p[0]
                self.marcos[idx, 1] = (1 - self.alpha) * self.marcos[idx, 1] + self.alpha * p[1]
                self.marcos[idx, 3] += 1
            else:
                self.marcos = np.vstack([self.marcos, [p[0], p[1], float(lado), 1.0]])

    def lados(self):
        esq = self.marcos[self.marcos[:, 2] == 1.0][:, :2]
        der = self.marcos[self.marcos[:, 2] == 2.0][:, :2]
        return esq, der

# ===================== 2) centerline (linha central) =====================
def _ordena_por_eixo_principal(pontos: np.ndarray):
    if len(pontos) < 2:
        return pontos, np.array([1.0, 0.0])
    P = pontos - pontos.mean(axis=0)
    _, _, vh = np.linalg.svd(P, full_matrices=False)
    eixo = vh[0]
    t = (pontos - pontos.mean(axis=0)) @ eixo
    ordem = np.argsort(t)
    return pontos[ordem], eixo

def centerline_por_pareamento(esq: np.ndarray, der: np.ndarray) -> np.ndarray:
    """
    Separa esquerda/direita, ordena ao longo do eixo longo e emparelha por projeção.
    Se faltar lado, cai para uma espinha simples com todos os marcos.
    """
    if len(esq) >= 3 and len(der) >= 3:
        esq_ord, eixo = _ordena_por_eixo_principal(esq)
        t_esq = (esq_ord - esq_ord.mean(axis=0)) @ eixo
        t_der = (der - esq_ord.mean(axis=0)) @ eixo
        cl = []
        for i in range(len(esq_ord)):
            j = int(np.argmin(np.abs(t_der - t_esq[i])))
            cl.append(0.5 * (esq_ord[i] + der[j]))
        cl = np.array(cl)
        # remove pontos quase duplicados
        keep = [0]
        for k in range(1, len(cl)):
            if np.hypot(*(cl[k] - cl[keep[-1]])) > 0.2:
                keep.append(k)
        return cl[keep]
    # fallback: usa todos
    todos = np.vstack([esq, der]) if (len(esq) + len(der)) else np.empty((0, 2))
    if len(todos) < 2:
        return todos
    cl, _ = _ordena_por_eixo_principal(todos)
    return cl

def suaviza_caminho(path: np.ndarray, passo: float = 0.5) -> np.ndarray:
    """Reamostra por comprimento de arco com interpolação linear (sem dependências pesadas)."""
    if path is None or len(path) < 2:
        return path
    P = np.asarray(path, float)
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])))]
    if s[-1] < 1e-3:
        return P
    s_new = np.arange(0.0, s[-1], max(1e-3, passo))
    x = np.interp(s_new, s, P[:, 0])
    y = np.interp(s_new, s, P[:, 1])
    return np.column_stack([x, y])

# ===================== 3) waypoints =====================
def yaw_e_curvatura(path: np.ndarray, eps: float = 1e-6):
    P = np.asarray(path, float)
    dx = np.gradient(P[:, 0])
    dy = np.gradient(P[:, 1])
    ds = np.hypot(dx, dy) + eps
    yaw = np.arctan2(dy, dx)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    kappa = (dx * ddy - dy * ddx) / (ds**3 + eps)
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])))]
    return yaw, kappa, s

def perfil_velocidade(s, kappa, v_reta=7.0, a_lat_max=3.5, a_acc=2.0, a_dec=3.0):
    k = np.abs(kappa)
    v_curva = np.sqrt(np.maximum(1e-6, a_lat_max) / np.maximum(1e-6, k))
    v0 = np.minimum(v_reta, v_curva)
    v = v0.copy()
    for i in range(1, len(v)):
        ds = max(1e-6, s[i] - s[i - 1])
        v[i] = min(v[i], math.sqrt(max(0.0, v[i - 1] ** 2 + 2 * a_acc * ds)))
    for i in range(len(v) - 2, -1, -1):
        ds = max(1e-6, s[i + 1] - s[i])
        v[i] = min(v[i], math.sqrt(max(0.0, v[i + 1] ** 2 + 2 * a_dec * ds)))
    return np.maximum(v, 0.5)

def gera_waypoints(path: np.ndarray) -> np.ndarray:
    if path is None or len(path) < 2:
        return np.empty((0, 6), dtype=float)
    yaw, kappa, s = yaw_e_curvatura(path)
    v = perfil_velocidade(s, kappa)
    return np.column_stack([path[:, 0], path[:, 1], yaw, kappa, s, v])

def corta_waypoints_a_frente(waypoints: np.ndarray, car_xy: np.ndarray, janela_m: float = 35.0):
    if waypoints.size == 0:
        return waypoints
    d = np.hypot(waypoints[:, 0] - car_xy[0], waypoints[:, 1] - car_xy[1])
    i0 = int(np.argmin(d))
    s0 = waypoints[i0, 4]
    mask = (waypoints[:, 4] >= s0) & (waypoints[:, 4] <= s0 + janela_m)
    sub = waypoints[mask]
    if len(sub) < 5:
        j1 = min(len(waypoints) - 1, i0 + 50)
        sub = waypoints[i0:j1]
    return sub

# ===================== Nó ROS 2 =====================
class DVPlannerNode(Node):
    def __init__(self):
        super().__init__('dv_planner_node')

        # Parâmetros configuráveis
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cones_topic', '/cones')
        self.declare_parameter('frame_map', 'map')
        self.declare_parameter('associacao_raio', 1.0)
        self.declare_parameter('suavizacao_alpha', 0.3)
        self.declare_parameter('ds_metros', 0.5)
        self.declare_parameter('janela_waypoints_m', 35.0)

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.cones_topic = self.get_parameter('cones_topic').get_parameter_value().string_value
        self.frame_map = self.get_parameter('frame_map').get_parameter_value().string_value
        self.ds = self.get_parameter('ds_metros').get_parameter_value().double_value
        self.janela_wp = self.get_parameter('janela_waypoints_m').get_parameter_value().double_value

        raio = self.get_parameter('associacao_raio').get_parameter_value().double_value
        alpha = self.get_parameter('suavizacao_alpha').get_parameter_value().double_value
        self.mapa = MapaCones(raio_assoc=raio, alpha_suave=alpha)

        # Estado
        self.pose_atual = np.array([0.0, 0.0, 0.0], dtype=float)
        self.tem_pose = False
        self.cones_body = np.empty((0, 3), dtype=float)

        # ROS I/O
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 50)
        self.sub_cones = self.create_subscription(Float32MultiArray, self.cones_topic, self.cones_cb, 10)
        self.pub_path = self.create_publisher(Path, '/planner/path', 10)
        self.pub_wps = self.create_publisher(Float32MultiArray, '/planner/waypoints', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/map_cones', 10)


        self.timer = self.create_timer(0.05, self.tick)  # 20 Hz

    def publicar_cones_markers(self):
        # pega cones já no MAPA
        esq, der = self.mapa.lados()

        msg = MarkerArray()

        # limpa tudo antes (evita fantasmas)
        clear = Marker()
        clear.action = Marker.DELETEALL
        msg.markers.append(clear)

        # helper para criar um marcador
        def make_marker(x, y, mid, r, g, b):
            m = Marker()
            m.header.frame_id = self.frame_map
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "cones_map"
            m.id = mid
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.0
            m.scale.x = m.scale.y = m.scale.z = 0.35  # diâmetro da bolinha
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
            m.lifetime.sec = 0
            return m

        # esquerda = azul; direita = amarelo
        mid = 0
        for x, y in esq:
            msg.markers.append(make_marker(x, y, mid, 0.0, 0.5, 1.0)); mid += 1
        for x, y in der:
            msg.markers.append(make_marker(x, y, mid, 1.0, 0.85, 0.0)); mid += 1

        self.pub_markers.publish(msg)
    

    # --------- callbacks ---------
    def odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose_atual[:] = [x, y, yaw]
        self.tem_pose = True

    def cones_cb(self, msg: Float32MultiArray):
        # Esperado: [x_b, y_b, lado, x_b, y_b, lado, ...]
        data = np.array(msg.data, dtype=float)
        if data.size % 3 != 0 or data.size == 0:
            self.cones_body = np.empty((0, 3), dtype=float)
            return
        self.cones_body = data.reshape(-1, 3)

    # --------- ciclo principal ---------
    def tick(self):
        if not self.tem_pose:
            return

        # 1) pegar detecções no frame do carro
        if len(self.cones_body) > 0:
            P_body = self.cones_body[:, :2]
            sides = self.cones_body[:, 2].astype(int)
            # 2) converter para mundo
            P_world = carro_para_mundo(P_body, self.pose_atual)
            # 3) atualizar mapa
            self.mapa.atualizar(P_world, sides)

        # 4) gerar linha central e suavizar
        esq, der = self.mapa.lados()
        center = centerline_por_pareamento(esq, der)
        path = suaviza_caminho(center, passo=self.ds)

        # 5) waypoints
        wps = gera_waypoints(path)
        wps_win = corta_waypoints_a_frente(wps, self.pose_atual[:2], janela_m=self.janela_wp)

        # 6) publicar
        if path is not None and len(path) >= 2:
            self.publicar_path(path)
        if len(wps_win) > 0:
            self.publicar_waypoints(wps_win)
        self.publicar_cones_markers()


    # --------- publicadores ---------
    def publicar_path(self, path_xy: np.ndarray):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_map
        for x, y in path_xy:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def publicar_waypoints(self, wps: np.ndarray):
        # Flatten: [x,y,yaw,kappa,s,v] * N
        arr = Float32MultiArray()
        arr.data = wps.astype(np.float32).reshape(-1).tolist()
        self.pub_wps.publish(arr)

def main():
    rclpy.init()
    node = DVPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
