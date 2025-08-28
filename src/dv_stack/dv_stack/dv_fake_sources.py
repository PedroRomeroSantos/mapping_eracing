#!/usr/bin/env python3
import math, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32MultiArray

from scipy.spatial import Voronoi, KDTree

# =================== utils ===================
def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw/2.0); q.w = math.cos(yaw/2.0)
    return q

def rot2d(th: float):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]], float)

# =================== pista elíptica ===================
def pista_elipse(a=30.0, b=15.0, W=6.0, n=300):
    t = np.linspace(0, 2*math.pi, n, endpoint=False)
    cx = a*np.cos(t); cy = b*np.sin(t)
    dx, dy = np.gradient(cx), np.gradient(cy)
    tv = np.stack([dx, dy], 1)
    tv /= np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-9)
    nL = np.stack([-tv[:,1], tv[:,0]], 1)
    center = np.stack([cx, cy], 1)
    left   = center + (W/2.0)*nL
    right  = center - (W/2.0)*nL
    return center, left, right

# =================== Voronoi: grafo + rastreio ===================
def _build_voronoi_graph(cones_all, largura_min=3.0, largura_max=7.0):
    """Filtra arestas do Voronoi cujo ponto médio está a metade da largura aceitável das 'bordas'."""
    vor = Voronoi(cones_all)
    verts = vor.vertices
    edges = []
    for (i, j), (p, q) in zip(vor.ridge_points, vor.ridge_vertices):
        if p == -1 or q == -1:
            continue
        a, b = verts[p], verts[q]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        mid = 0.5*(a+b)
        d = np.sort(np.hypot(cones_all[:,0]-mid[0], cones_all[:,1]-mid[1]))[:2]
        if len(d) == 2 and (largura_min/2.0) <= d[0] <= (largura_max/2.0) and (largura_min/2.0) <= d[1] <= (largura_max/2.0):
            edges.append((p, q))
    # remover vértices inválidos e remapear
    mask_ok = np.isfinite(verts).all(axis=1)
    remap = -np.ones(len(verts), dtype=int)
    keep = np.where(mask_ok)[0]
    remap[keep] = np.arange(len(keep))
    verts_ok = verts[keep]
    edges_ok = [(remap[u], remap[v]) for (u,v) in edges if remap[u] >= 0 and remap[v] >= 0]
    return verts_ok, edges_ok

def _trace_ordered_polyline(verts, edges, seed_xy, seed_dir, max_turn_deg=100):
    """Percorre o grafo avançando na direção 'seed_dir' e evitando retrocessos bruscos."""
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
        candidates = [j for j in adj[curr] if not used[j]]
        if not candidates: break
        best = None; best_score = -1e9; best_vec = None
        for j in candidates:
            v = verts[j] - verts[curr]
            nv = np.linalg.norm(v)
            if nv < 1e-6: continue
            v = v / nv
            turn = math.degrees(math.acos(np.clip(np.dot(v, last_dir), -1.0, 1.0)))
            if turn > max_turn_deg:  # guinada muito grande
                continue
            score = np.dot(v, dref)  # avanço na direção de marcha
            if score > best_score:
                best_score, best, best_vec = score, j, v
        if best is None: break
        path.append(best); used[best] = True
        last_dir = best_vec; curr = best
    return np.array([verts[k] for k in path], float)

def voronoi_centerline_from_cones(cones_all, largura_min, largura_max, seed_xy, seed_yaw):
    verts, edges = _build_voronoi_graph(cones_all, largura_min, largura_max)
    if len(verts) == 0 or len(edges) == 0:
        return np.empty((0,2))
    poly = _trace_ordered_polyline(verts, edges, np.asarray(seed_xy,float), float(seed_yaw))
    return poly

# =================== Nó ===================
class FakeTrack(Node):
    def __init__(self):
        super().__init__('fake_track')

        # ----- parâmetros -----
        self.declare_parameter('a', 30.0)
        self.declare_parameter('b', 15.0)
        self.declare_parameter('W', 6.0)
        self.declare_parameter('n', 300)
        self.declare_parameter('freq_hz', 20.0)

        # “sensor” sintético (para /cones):
        self.declare_parameter('alcance_m', 28.0)
        self.declare_parameter('fov_graus', 120.0)

        # Voronoi centerline (para debug):
        self.declare_parameter('largura_tol', 0.7)     # tolerância sobre W => [W- tol, W+tol]
        self.declare_parameter('voro_max_turn_deg', 100.0)

        a = float(self.get_parameter('a').value)
        b = float(self.get_parameter('b').value)
        W = float(self.get_parameter('W').value)
        n = int(self.get_parameter('n').value)

        self.center, self.left_world, self.right_world = pista_elipse(a=a, b=b, W=W, n=n)
        self.N = len(self.center)
        self.i = 0

        self.range = float(self.get_parameter('alcance_m').value)
        self.fov = math.radians(float(self.get_parameter('fov_graus').value))
        self.dt = 1.0/float(self.get_parameter('freq_hz').value)

        self.largura_min = max(0.5, W - float(self.get_parameter('largura_tol').value))
        self.largura_max = W + float(self.get_parameter('largura_tol').value)
        self.voro_max_turn = float(self.get_parameter('voro_max_turn_deg').value)

        # ----- publishers -----
        self.pub_odom  = self.create_publisher(Odometry, '/odom', 10)
        self.pub_cones = self.create_publisher(Float32MultiArray, '/cones', 10)

        # latched para não piscar no RViz
        qos_latched = QoSProfile(depth=1,
                                 durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                                 history=QoSHistoryPolicy.KEEP_LAST)
        self.pub_map = self.create_publisher(MarkerArray, '/map_cones', qos_latched)
        self.pub_voro = self.create_publisher(MarkerArray, '/debug/voronoi_centerline', qos_latched)

        # publica mapa/centerline uma vez e inicia animação
        self.publish_map_once()
        self.publish_voronoi_once(seed_idx=0)
        self.timer = self.create_timer(self.dt, self.tick)

    # -------- mapa de cones (latched) --------
    def publish_map_once(self):
        msg = MarkerArray(); mid = 0
        def add(arr, rgba):
            nonlocal mid
            for x,y in arr:
                m = Marker()
                m.header.frame_id = 'map'
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = "cones"; m.id = mid; mid += 1
                m.type = Marker.SPHERE; m.action = Marker.ADD
                m.pose.position.x = float(x); m.pose.position.y = float(y); m.pose.position.z = 0.0
                m.scale.x = m.scale.y = m.scale.z = 0.4
                m.color.r, m.color.g, m.color.b, m.color.a = rgba
                msg.markers.append(m)
        add(self.left_world,  (0.0, 0.5, 1.0, 1.0))   # azul
        add(self.right_world, (1.0, 0.85, 0.0, 1.0))  # amarelo
        self.pub_map.publish(msg)

    # -------- voronoi centerline (latched) --------
    def publish_voronoi_once(self, seed_idx=0):
        car_xy = self.center[seed_idx]
        car_yaw = math.atan2(*(self.center[(seed_idx+1)%self.N] - self.center[seed_idx])[::-1])
        cones_all = np.vstack([self.left_world, self.right_world])
        cl = voronoi_centerline_from_cones(
            cones_all,
            largura_min=self.largura_min,
            largura_max=self.largura_max,
            seed_xy=car_xy,
            seed_yaw=car_yaw
        )
        msg = MarkerArray()
        if cl is None or len(cl) < 2:
            self.pub_voro.publish(msg); return
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "voronoi"; m.id = 0
        m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.scale.x = 0.12
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 1.0
        m.points = [Point(x=float(x), y=float(y), z=0.0) for (x,y) in cl]
        msg.markers.append(m)
        self.pub_voro.publish(msg)

    # -------- animação + /odom + /cones --------
    def tick(self):
        i0, i1 = self.i % self.N, (self.i + 1) % self.N
        p0, p1 = self.center[i0], self.center[i1]
        yaw = math.atan2(p1[1]-p0[1], p1[0]-p0[0])

        # /odom (no frame 'map')
        od = Odometry()
        od.header.stamp = self.get_clock().now().to_msg()
        od.header.frame_id = 'map'; od.child_frame_id = 'base_link'
        od.pose.pose.position.x = float(p0[0]); od.pose.pose.position.y = float(p0[1])
        od.pose.pose.orientation = yaw_to_quat(yaw)
        od.twist.twist.linear.x = 2.0
        self.pub_odom.publish(od)

        # /cones (no frame 'base_link')
        Rt = rot2d(yaw).T
        car = p0
        detections = []
        for side, arr in [(1.0, self.left_world), (2.0, self.right_world)]:
            for pw in arr:
                pb = Rt @ (pw - car)
                xb, yb = float(pb[0]), float(pb[1])
                if xb < 0.2 or xb > self.range:   # só à frente e dentro do alcance
                    continue
                if abs(math.atan2(yb, xb)) > self.fov/2.0:
                    continue
                detections.extend([xb, yb, side])
        self.pub_cones.publish(Float32MultiArray(data=detections))

        self.i = (self.i + 1) % self.N

def main():
    rclpy.init()
    node = FakeTrack()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
