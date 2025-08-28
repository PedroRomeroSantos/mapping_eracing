#!/usr/bin/env python3
# dv_planner_node.py  (versão focada em VORONOI)
import os, math, numpy as np, rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

from scipy.spatial import Voronoi, KDTree
from scipy.interpolate import splprep, splev

# matplotlib headless p/ salvar PNG
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ------------------------- util geom -------------------------
def _rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=float)

def carro_para_mundo(pts_body: np.ndarray, pose_xyz: np.ndarray) -> np.ndarray:
    x, y, yaw = pose_xyz
    if pts_body.size == 0:
        return np.empty((0, 2), dtype=float)
    return (pts_body @ _rot(yaw).T) + np.array([x, y], dtype=float)

def suaviza_caminho(path_xy: np.ndarray, smooth=0.3, ds=0.5) -> np.ndarray:
    if path_xy is None or len(path_xy) < 3: return path_xy
    x, y = path_xy[:,0], path_xy[:,1]
    s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    if s[-1] < 1.0: return path_xy
    tck, _ = splprep([x, y], u=s, s=smooth*len(x))
    s_new = np.arange(0, s[-1], ds)
    x_new, y_new = splev(s_new, tck)
    return np.column_stack([x_new, y_new])


# ------------------------- mapa incremental -------------------------
class MapaCones:
    """Cada marco: [x, y, lado, contagem] lado: 1.0=esq, 2.0=dir"""
    def __init__(self, raio_assoc=1.0, alpha_suave=0.3):
        self.raio = float(raio_assoc)
        self.alpha = float(alpha_suave)
        self.marcos = np.empty((0, 4), dtype=float)

    def _associar(self, p: np.ndarray, lado: int) -> int:
        if len(self.marcos) == 0: return -1
        mask = (self.marcos[:, 2] == float(lado))
        if not np.any(mask): return -1
        idxs = np.where(mask)[0]
        cand = self.marcos[idxs]
        d = np.hypot(cand[:,0]-p[0], cand[:,1]-p[1])
        j = int(np.argmin(d))
        return idxs[j] if d[j] <= self.raio else -1

    def atualizar(self, cones_mundo: np.ndarray, lados: np.ndarray):
        for p, lado in zip(cones_mundo, lados):
            i = self._associar(p, int(lado))
            if i >= 0:
                self.marcos[i,0] = (1-self.alpha)*self.marcos[i,0] + self.alpha*p[0]
                self.marcos[i,1] = (1-self.alpha)*self.marcos[i,1] + self.alpha*p[1]
                self.marcos[i,3] += 1
            else:
                self.marcos = np.vstack([self.marcos, [p[0], p[1], float(lado), 1.0]])

    def lados(self, min_count: int = 1):
        if len(self.marcos) == 0:
            return np.empty((0,2)), np.empty((0,2))
        m = self.marcos[self.marcos[:,3] >= float(min_count)]
        esq = m[m[:,2] == 1.0][:,:2]
        der = m[m[:,2] == 2.0][:,:2]
        return esq, der


# ------------------------- voronoi: grafo + rastreio -------------------------
def _build_voronoi_graph(cones_all, largura_min=3.0, largura_max=7.0):
    """Filtra arestas cujo ponto médio está entre as bordas (metade da largura no Voronoi)."""
    vor = Voronoi(cones_all)
    verts = vor.vertices
    edges = []
    for (_, _), (p, q) in zip(vor.ridge_points, vor.ridge_vertices):
        if p == -1 or q == -1: continue
        a, b = verts[p], verts[q]
        if not (np.isfinite(a).all() and np.isfinite(b).all()): continue
        mid = 0.5*(a+b)
        d = np.sort(np.hypot(cones_all[:,0]-mid[0], cones_all[:,1]-mid[1]))[:2]
        # metade da largura porque Voronoi está equidistante às duas "paredes"
        if len(d)==2 and (largura_min/2.0) <= d[0] <= (largura_max/2.0) and (largura_min/2.0) <= d[1] <= (largura_max/2.0):
            edges.append((p, q))
    mask_ok = np.isfinite(verts).all(axis=1)
    remap = -np.ones(len(verts), dtype=int)
    keep = np.where(mask_ok)[0]
    remap[keep] = np.arange(len(keep))
    verts_ok = verts[keep]
    edges_ok = [(remap[u], remap[v]) for (u,v) in edges if remap[u]>=0 and remap[v]>=0]
    return verts_ok, edges_ok

def _trace_ordered_polyline(verts, edges, seed_xy, seed_dir, max_turn_deg=100):
    """Percorre o grafo avançando na direção seed_dir e evitando retrocessos bruscos."""
    if len(verts) == 0 or len(edges) == 0:
        return np.empty((0,2))
    adj = [[] for _ in range(len(verts))]
    for u,v in edges:
        adj[u].append(v); adj[v].append(u)

    kdt = KDTree(verts); i = int(kdt.query(seed_xy)[1])
    used = np.zeros(len(verts), dtype=bool)
    path = [i]; used[i] = True
    dref = np.array([math.cos(seed_dir), math.sin(seed_dir)], float)
    curr = i; last_dir = dref

    for _ in range(4*len(verts)):
        cand = [j for j in adj[curr] if not used[j]]
        if not cand: break
        best, best_score, best_vec = None, -1e9, None
        for j in cand:
            v = verts[j] - verts[curr]; nv = np.linalg.norm(v)
            if nv < 1e-6: continue
            v = v / nv
            turn = math.degrees(math.acos(np.clip(np.dot(v, last_dir), -1.0, 1.0)))
            if turn > max_turn_deg:  # guinada absurda
                continue
            score = np.dot(v, dref)   # avanço na direção de marcha
            if score > best_score:
                best, best_score, best_vec = j, score, v
        if best is None: break
        path.append(best); used[best] = True
        last_dir = best_vec; curr = best
    return np.array([verts[k] for k in path], float)

def centerline_voronoi(cones_all: np.ndarray, largura_min=3.0, largura_max=7.0,
                       seed_xy=None, seed_yaw=0.0) -> np.ndarray:
    if cones_all is None or len(cones_all) < 4:
        return np.empty((0,2))
    verts, edges = _build_voronoi_graph(cones_all, largura_min, largura_max)
    if len(verts) == 0 or len(edges) == 0:
        return np.empty((0,2))
    if seed_xy is None: seed_xy = cones_all.mean(0)
    poly = _trace_ordered_polyline(verts, edges, np.asarray(seed_xy,float), float(seed_yaw))
    if len(poly) >= 3:
        return suaviza_caminho(poly, smooth=0.3, ds=0.5)
    return poly


# ------------------------- waypoints -------------------------
def yaw_e_curvatura(path: np.ndarray, eps: float = 1e-6):
    P = np.asarray(path, float)
    dx = np.gradient(P[:,0]); dy = np.gradient(P[:,1])
    ds = np.hypot(dx,dy) + eps
    yaw = np.arctan2(dy, dx)
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    kappa = (dx*ddy - dy*ddx) / (ds**3 + eps)
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(P[:,0]), np.diff(P[:,1])))]
    return yaw, kappa, s

def perfil_velocidade(s, kappa, v_reta=7.0, a_lat_max=3.5, a_acc=2.0, a_dec=3.0):
    k = np.abs(kappa)
    v_curva = np.sqrt(np.maximum(1e-6, a_lat_max) / np.maximum(1e-6, k))
    v = np.minimum(v_reta, v_curva)
    for i in range(1, len(v)):
        ds = max(1e-6, s[i]-s[i-1]); v[i] = min(v[i], math.sqrt(max(0.0, v[i-1]**2 + 2*a_acc*ds)))
    for i in range(len(v)-2, -1, -1):
        ds = max(1e-6, s[i+1]-s[i]); v[i] = min(v[i], math.sqrt(max(0.0, v[i+1]**2 + 2*a_dec*ds)))
    return np.maximum(v, 0.5)

def gera_waypoints(path: np.ndarray) -> np.ndarray:
    if path is None or len(path) < 2:
        return np.empty((0,6), dtype=float)
    yaw, kappa, s = yaw_e_curvatura(path)
    v = perfil_velocidade(s, kappa)
    return np.column_stack([path[:,0], path[:,1], yaw, kappa, s, v])

def corta_waypoints_a_frente(waypoints: np.ndarray, car_xy: np.ndarray, janela_m: float = 35.0):
    if waypoints.size == 0: return waypoints
    d = np.hypot(waypoints[:,0]-car_xy[0], waypoints[:,1]-car_xy[1])
    i0 = int(np.argmin(d)); s0 = waypoints[i0,4]
    mask = (waypoints[:,4] >= s0) & (waypoints[:,4] <= s0 + janela_m)
    sub = waypoints[mask]
    if len(sub) < 5:
        j1 = min(len(waypoints)-1, i0+50)
        sub = waypoints[i0:j1]
    return sub


# ------------------------- nó ROS2 -------------------------
class DVPlannerNode(Node):
    def __init__(self):
        super().__init__('dv_planner_node')

        # parâmetros principais
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cones_topic', '/cones')
        self.declare_parameter('frame_map', 'map')
        self.declare_parameter('associacao_raio', 1.2)
        self.declare_parameter('suavizacao_alpha', 0.3)
        self.declare_parameter('ds_metros', 0.5)
        self.declare_parameter('janela_waypoints_m', 35.0)

        # planejamento local + debug plot
        self.declare_parameter('plan_radius', 35.0)
        self.declare_parameter('fwd_fov_deg', 120.0)
        self.declare_parameter('largura_min', 3.0)
        self.declare_parameter('largura_max', 7.0)
        self.declare_parameter('min_count', 2)
        self.declare_parameter('plot_debug', True)
        self.declare_parameter('plot_debug_period_s', 1.0)
        self.declare_parameter('plot_debug_path', os.path.expanduser('~/.ros/planner_debug.png'))

        # lê parâmetros
        self.odom_topic = self.get_parameter('odom_topic').value
        self.cones_topic = self.get_parameter('cones_topic').value
        self.frame_map = self.get_parameter('frame_map').value
        self.ds = float(self.get_parameter('ds_metros').value)
        self.janela_wp = float(self.get_parameter('janela_waypoints_m').value)

        self.plan_radius = float(self.get_parameter('plan_radius').value)
        self.fwd_fov = math.radians(float(self.get_parameter('fwd_fov_deg').value))
        self.largura_min = float(self.get_parameter('largura_min').value)
        self.largura_max = float(self.get_parameter('largura_max').value)
        self.min_count = int(self.get_parameter('min_count').value)

        self.plot_debug = bool(self.get_parameter('plot_debug').value)
        self.plot_period = float(self.get_parameter('plot_debug_period_s').value)
        self.plot_path = str(self.get_parameter('plot_debug_path').value)
        self._last_plot_t = 0.0

        # mapa
        raio = float(self.get_parameter('associacao_raio').value)
        alpha = float(self.get_parameter('suavizacao_alpha').value)
        self.mapa = MapaCones(raio_assoc=raio, alpha_suave=alpha)

        # estado
        self.pose_atual = np.array([0.0, 0.0, 0.0], float)
        self.tem_pose = False
        self.cones_body = np.empty((0,3), float)

        # I/O
        self.sub_odom  = self.create_subscription(Odometry, self.odom_topic,  self.odom_cb, 50)
        self.sub_cones = self.create_subscription(Float32MultiArray, self.cones_topic, self.cones_cb, 10)

        self.pub_path        = self.create_publisher(Path, '/planner/path', 10)
        self.pub_wps         = self.create_publisher(Float32MultiArray, '/planner/waypoints', 10)
        self.pub_wp_markers  = self.create_publisher(MarkerArray, '/waypoints_markers', 10)
        self.pub_map_markers = self.create_publisher(MarkerArray, '/map_cones', 10)

        self.timer = self.create_timer(0.05, self.tick)  # 20 Hz

    # callbacks
    def odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self.pose_atual[:] = [x, y, yaw]; self.tem_pose = True

    def cones_cb(self, msg: Float32MultiArray):
        data = np.array(msg.data, float)
        if data.size % 3 != 0 or data.size == 0:
            self.cones_body = np.empty((0,3), float); return
        self.cones_body = data.reshape(-1,3)

    # publicadores
    def publicar_cones_markers(self, esq, der):
        msg = MarkerArray()
        clear = Marker(); clear.action = Marker.DELETEALL
        msg.markers.append(clear)
        def mkr(x,y,mid,rgba):
            m = Marker()
            m.header.frame_id = self.frame_map
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "cones_map"; m.id = mid
            m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position.x = float(x); m.pose.position.y = float(y); m.pose.position.z = 0.0
            m.scale.x = m.scale.y = m.scale.z = 0.35
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            return m
        mid = 0
        for x,y in esq: msg.markers.append(mkr(x,y,mid,(0.0,0.5,1.0,1.0))); mid += 1
        for x,y in der: msg.markers.append(mkr(x,y,mid,(1.0,0.85,0.0,1.0))); mid += 1
        self.pub_map_markers.publish(msg)

    def publicar_path(self, path_xy):
        msg = Path(); msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_map
        for x,y in path_xy:
            ps = PoseStamped(); ps.header = msg.header
            ps.pose.position.x = float(x); ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0; msg.poses.append(ps)
        self.pub_path.publish(msg)

    def publicar_waypoints(self, wps):
        arr = Float32MultiArray(); arr.data = wps.astype(np.float32).reshape(-1).tolist()
        self.pub_wps.publish(arr)

    def publicar_waypoints_markers(self, wps):
        msg = MarkerArray(); clr = Marker(); clr.action = Marker.DELETEALL
        msg.markers.append(clr)
        now = self.get_clock().now().to_msg()
        for i, row in enumerate(wps):
            x,y = float(row[0]), float(row[1])
            m = Marker()
            m.header.frame_id = self.frame_map; m.header.stamp = now
            m.ns = "waypoints"; m.id = i
            m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position.x = x; m.pose.position.y = y; m.pose.position.z = 0.0
            m.scale.x = m.scale.y = m.scale.z = 0.25
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.3, 1.0
            msg.markers.append(m)
        self.pub_wp_markers.publish(msg)

    def salvar_plot_debug(self, left, right, path, wps, car_xy, out_path):
        if left is None: left = np.empty((0,2))
        if right is None: right = np.empty((0,2))
        plt.figure(figsize=(7,5), dpi=120)
        if len(left):  plt.plot(left[:,0],  left[:,1],  '.', ms=3, label='cones L', color='tab:blue')
        if len(right): plt.plot(right[:,0], right[:,1], '.', ms=3, label='cones R', color='tab:orange')
        if path is not None and len(path)>1:
            plt.plot(path[:,0], path[:,1], '-', lw=2, label='path', color='tab:green')
        if wps is not None and len(wps)>0:
            plt.plot(wps[:,0], wps[:,1], 'o', ms=3, label='waypoints', color='tab:green', alpha=0.6)
        if car_xy is not None:
            plt.plot([car_xy[0]],[car_xy[1]], '*', ms=10, label='car', color='tab:red')
        plt.axis('equal'); plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(out_path); plt.close()

    # loop
    def tick(self):
        if not self.tem_pose: return

        # 1) detecções → mundo → mapa
        if len(self.cones_body) > 0:
            P_body = self.cones_body[:,:2]
            sides  = self.cones_body[:,2].astype(int)
            P_world = carro_para_mundo(P_body, self.pose_atual)
            self.mapa.atualizar(P_world, sides)

        # 2) cones confiáveis + filtro local (raio/FOV à frente)
        esq_all, der_all = self.mapa.lados(min_count=self.min_count)
        car_xy = self.pose_atual[:2]; yaw = self.pose_atual[2]

        def filtra_locais(pts):
            if pts is None or len(pts)==0: return np.empty((0,2))
            v = pts - car_xy
            dist = np.hypot(v[:,0], v[:,1])
            ang  = np.arctan2(v[:,1], v[:,0]) - yaw
            ang  = (ang + math.pi) % (2*math.pi) - math.pi
            keep = (dist <= self.plan_radius) & (np.abs(ang) <= self.fwd_fov/2.0)
            return pts[keep]

        esq = filtra_locais(esq_all); der = filtra_locais(der_all)

        # 3) centerline via Voronoi (grafo+rastreio)
        cones_all = np.vstack([esq, der]) if (len(esq)+len(der)) else np.empty((0,2))
        cl = centerline_voronoi(cones_all,
                                largura_min=self.largura_min,
                                largura_max=self.largura_max,
                                seed_xy=car_xy,
                                seed_yaw=yaw)

        path = suaviza_caminho(cl, smooth=0.3, ds=self.ds) if cl is not None else None

        # 4) waypoints (corta janela à frente)
        wps = gera_waypoints(path)
        wps_win = corta_waypoints_a_frente(wps, car_xy, janela_m=self.janela_wp)

        # 5) publicar
        if path is not None and len(path) >= 2:
            self.publicar_path(path)
        if len(wps_win) > 0:
            self.publicar_waypoints(wps_win)
            self.publicar_waypoints_markers(wps_win[:, :2])
        self.publicar_cones_markers(esq_all, der_all)

        # 6) plot PNG periódico
        if self.plot_debug:
            t_now = self.get_clock().now().nanoseconds * 1e-9
            if t_now - self._last_plot_t >= self.plot_period:
                self.salvar_plot_debug(esq_all, der_all, path,
                                       wps_win[:, :2] if len(wps_win)>0 else None,
                                       car_xy, self.plot_path)
                self._last_plot_t = t_now


def main():
    rclpy.init()
    node = DVPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
