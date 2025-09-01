#!/usr/bin/env python3
# dv_planner_node.py

# importa funções matemáticas básicas
import math
# importa NumPy para operações vetoriais/matriciais eficientes
import numpy as np
# importa o cliente ROS2 em Python
import rclpy
# importa a classe base de nós ROS2
from rclpy.node import Node
# mensagem simples de array de floats (usada para cones e waypoints)
from std_msgs.msg import Float32MultiArray
# mensagens de navegação (odometria e caminho/Path)
from nav_msgs.msg import Odometry, Path
# mensagem de pose com carimbo para compor um Path
from geometry_msgs.msg import PoseStamped
# mensagens de marcadores para visualização no RViz
from visualization_msgs.msg import Marker, MarkerArray
from delaunay_triangulation import DelaunayTriangulator

# ===================== utilidades geométricas =====================

# define uma função que cria a matriz de rotação 2D a partir do yaw
def _rot(yaw: float) -> np.ndarray:
    # calcula cosseno e seno do ângulo
    c, s = math.cos(yaw), math.sin(yaw)
    # monta e retorna a matriz de rotação 2x2
    return np.array([[c, -s], [s, c]], dtype=float)

# define função que transforma pontos do frame do carro para o frame do mapa
def carro_para_mundo(pts_body: np.ndarray, pose_xyz: np.ndarray) -> np.ndarray:
    """Converte pontos no frame do carro (x para frente, y para esquerda) para mundo (map)."""
    # separa a pose em x, y e yaw
    x, y, yaw = pose_xyz
    # se não há pontos, retorna array vazio (forma 0x2)
    if pts_body.size == 0:
        return np.empty((0, 2), dtype=float)
    # aplica rotação (para alinhar com mapa) e soma a translação (x,y) do carro
    return (pts_body @ _rot(yaw).T) + np.array([x, y], dtype=float)


# ===================== 1) mapa incremental de cones =====================

# classe para manter e atualizar um mapa de cones no frame do mundo
class MapaCones:
    """
    Guarda marcos (cones) no mundo e os atualiza com suavização quando são vistos de novo.
    Cada marco: [x, y, lado, contagem]
    """

    # construtor com parâmetros de associação (raio) e suavização (alpha)
    def __init__(self, raio_assoc=1.0, alpha_suave=0.3):
        # raio máximo para associar detecções a cones existentes
        self.raio = float(raio_assoc)
        # fator de suavização exponencial (0..1)
        self.alpha = float(alpha_suave)
        # matriz de marcos: colunas [x, y, lado, contagem]
        self.marcos = np.empty((0, 4), dtype=float)  # x,y,lado,contagem

    # método interno que busca um cone existente próximo (mesmo lado)
    def _associar(self, p: np.ndarray, lado: int) -> int:
        # se não há cones no mapa, não há associação
        if len(self.marcos) == 0:
            return -1
        # cria máscara para o lado (1.0=esq, 2.0=dir)
        mask = (self.marcos[:, 2] == lado)
        # se não há cones desse lado, retorna -1
        if not np.any(mask):
            return -1
        # filtra os candidatos daquele lado
        cand = self.marcos[mask]
        # calcula distâncias euclidianas do ponto p aos candidatos
        d = np.hypot(cand[:, 0] - p[0], cand[:, 1] - p[1])
        # pega o índice do candidato mais próximo
        j = int(np.argmin(d))
        # se a distância é menor ou igual ao raio, retorna o índice global no vetor marcos
        if d[j] <= self.raio:
            return np.where(mask)[0][j]
        # senão, não associou
        return -1

    # atualiza o mapa com um conjunto de detecções de cones no mundo
    def atualizar(self, cones_mundo: np.ndarray, lados: np.ndarray):
        # percorre cada detecção (posição p e lado)
        for p, lado in zip(cones_mundo, lados):
            # tenta associar com um cone existente
            idx = self._associar(p, int(lado))
            # se já existia, faz suavização exponencial e incrementa contagem
            if idx >= 0:
                # atualiza x suavizado
                self.marcos[idx, 0] = (1 - self.alpha) * self.marcos[idx, 0] + self.alpha * p[0]
                # atualiza y suavizado
                self.marcos[idx, 1] = (1 - self.alpha) * self.marcos[idx, 1] + self.alpha * p[1]
                # incrementa contagem de observações
                self.marcos[idx, 3] += 1
            else:
                # se não existia, cria um novo marco (x,y,lado,contagem=1)
                self.marcos = np.vstack([self.marcos, [p[0], p[1], float(lado), 1.0]])

    # retorna arrays separados de cones da esquerda e da direita (apenas x,y)
    def lados(self):
        # seleciona linhas do lado esquerdo e pega colunas x,y
        esq = self.marcos[self.marcos[:, 2] == 1.0][:, :2]
        # seleciona linhas do lado direito e pega colunas x,y
        der = self.marcos[self.marcos[:, 2] == 2.0][:, :2]
        # retorna as duas matrizes
        return esq, der


# ===================== 2) centerline (linha central) =====================

# ordena pontos ao longo do eixo principal (via SVD/PCA simples)
def _ordena_por_eixo_principal(pontos: np.ndarray):
    # se há menos de 2 pontos, devolve como está e um eixo padrão
    if len(pontos) < 2:
        return pontos, np.array([1.0, 0.0])
    # centraliza os pontos subtraindo a média
    P = pontos - pontos.mean(axis=0)
    # faz SVD para obter o eixo de maior variação (primeiro vetor de vh)
    _, _, vh = np.linalg.svd(P, full_matrices=False)
    # vetor unitário do eixo principal
    eixo = vh[0]
    # projeta os pontos ao longo desse eixo
    t = (pontos - pontos.mean(axis=0)) @ eixo
    # obtém ordem crescente das projeções
    ordem = np.argsort(t)
    # retorna pontos ordenados e o eixo
    return pontos[ordem], eixo

# gera uma linha central por pareamento: média de pares L↔R com projeções similares
def centerline_delaunay(esq: np.ndarray, der: np.ndarray) -> np.ndarray:
    """
    Gera a linha central usando triangulação de Delaunay dos cones.
    """
    cones = np.vstack([esq, der]) if len(esq) + len(der) > 0 else np.empty((0, 2))
    if len(cones) < 3:
        return cones
    triangulator = DelaunayTriangulator(cones)
    triangulator.compute_triangles()
    edges = triangulator.get_edges()
    mids = []
    for a, b in edges:
        pa, pb = cones[a], cones[b]
        mids.append((pa + pb) / 2)
    mids = np.array(mids)
    if len(mids) < 2:
        return mids
    # Ordena os pontos médios ao longo do eixo principal
    mids_ord, _ = _ordena_por_eixo_principal(mids)
    return mids_ord

# suaviza/reamostra a linha central por comprimento de arco (interp linear)
def suaviza_caminho(path: np.ndarray, passo: float = 0.5) -> np.ndarray:
    """Reamostra por comprimento de arco com interpolação linear (sem dependências pesadas)."""
    # se caminho for vazio ou com 1 ponto, retorna como está
    if path is None or len(path) < 2:
        return path
    # garante tipo float e forma (N,2)
    P = np.asarray(path, float)
    # acumula o comprimento de arco s ao longo dos segmentos
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])))]
    # se comprimento total é muito pequeno, retorna original
    if s[-1] < 1e-3:
        return P
    # define novos parâmetros de amostragem igualmente espaçados
    s_new = np.arange(0.0, s[-1], max(1e-3, passo))
    # interpola x ao longo de s
    x = np.interp(s_new, s, P[:, 0])
    # interpola y ao longo de s
    y = np.interp(s_new, s, P[:, 1])
    # empilha x e y em colunas e retorna a nova polilinha
    return np.column_stack([x, y])


# ===================== 3) waypoints =====================

# calcula yaw (tangente), curvatura e s (comprimento acumulado) ao longo do path
def yaw_e_curvatura(path: np.ndarray, eps: float = 1e-6):
    # converte para array float
    P = np.asarray(path, float)
    # derivadas aproximadas de x e y por diferença finita
    dx = np.gradient(P[:, 0])
    dy = np.gradient(P[:, 1])
    # norma do vetor tangente (evita divisão por zero com eps)
    ds = np.hypot(dx, dy) + eps
    # yaw é o ângulo da tangente (dy/dx)
    yaw = np.arctan2(dy, dx)
    # segundas derivadas aproximadas
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    # curvatura kappa = (x' y'' - y' x'') / |[x',y']|^3
    kappa = (dx * ddy - dy * ddx) / (ds**3 + eps)
    # s acumulado ao longo do caminho
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])))]
    # retorna yaw, curvatura e s
    return yaw, kappa, s

# gera um perfil de velocidade limitado por aceleração lateral e acel/dec longitudinal
def perfil_velocidade(s, kappa, v_reta=7.0, a_lat_max=3.5, a_acc=2.0, a_dec=3.0):
    # usa curvatura absoluta
    k = np.abs(kappa)
    # velocidade máxima por limite de aceleração lateral (v<=sqrt(a_lat/|k|))
    v_curva = np.sqrt(np.maximum(1e-6, a_lat_max) / np.maximum(1e-6, k))
    # clippa pela velocidade de reta
    v0 = np.minimum(v_reta, v_curva)
    # copia para trabalhar adiante/atrás
    v = v0.copy()
    # passe para frente: limita aceleração positiva (a_acc)
    for i in range(1, len(v)):
        ds = max(1e-6, s[i] - s[i - 1])
        v[i] = min(v[i], math.sqrt(max(0.0, v[i - 1] ** 2 + 2 * a_acc * ds)))
    # passe para trás: limita desaceleração (a_dec)
    for i in range(len(v) - 2, -1, -1):
        ds = max(1e-6, s[i + 1] - s[i])
        v[i] = min(v[i], math.sqrt(max(0.0, v[i + 1] ** 2 + 2 * a_dec * ds)))
    # evita valores muito pequenos
    return np.maximum(v, 0.5)

# empacota os waypoints (x,y,yaw,curvatura,s,velocidade) a partir do path
def gera_waypoints(path: np.ndarray) -> np.ndarray:
    # se não há caminho suficiente, retorna vazio
    if path is None or len(path) < 2:
        return np.empty((0, 6), dtype=float)
    # calcula yaw, curvatura e s
    yaw, kappa, s = yaw_e_curvatura(path)
    # gera perfil de velocidade
    v = perfil_velocidade(s, kappa)
    # empilha colunas e retorna Nx6
    return np.column_stack([path[:, 0], path[:, 1], yaw, kappa, s, v])

# recorta somente os waypoints à frente do carro em uma janela de metros
def corta_waypoints_a_frente(waypoints: np.ndarray, car_xy: np.ndarray, janela_m: float = 35.0):
    # se não há waypoints, retorna-os
    if waypoints.size == 0:
        return waypoints
    # distância do carro para todos os waypoints
    d = np.hypot(waypoints[:, 0] - car_xy[0], waypoints[:, 1] - car_xy[1])
    # índice do mais próximo (pivô inicial)
    i0 = int(np.argmin(d))
    # s no pivô
    s0 = waypoints[i0, 4]
    # máscara para pontos dentro da janela de s à frente
    mask = (waypoints[:, 4] >= s0) & (waypoints[:, 4] <= s0 + janela_m)
    # subtrajectória à frente
    sub = waypoints[mask]
    # fallback: se poucos pontos, pega um bloco fixo após o pivô
    if len(sub) < 5:
        j1 = min(len(waypoints) - 1, i0 + 50)
        sub = waypoints[i0:j1]
    # retorna a janela final
    return sub


# ===================== Nó ROS 2 =====================

# define o nó de planejamento
class DVPlannerNode(Node):
    # construtor do nó
    def __init__(self):
        # inicializa a classe base com o nome do nó
        super().__init__('dv_planner_node')

        # ---------------- Parâmetros configuráveis ----------------
        # nome do tópico de odometria
        self.declare_parameter('odom_topic', '/odom')
        # nome do tópico de detecções de cones (no frame do carro)
        self.declare_parameter('cones_topic', '/cones')
        # nome do frame global (mapa)
        self.declare_parameter('frame_map', 'map')
        # raio de associação para juntar detecções a cones do mapa
        self.declare_parameter('associacao_raio', 1.0)
        # alpha de suavização exponencial para posições dos cones
        self.declare_parameter('suavizacao_alpha', 0.3)
        # passo de reamostragem do caminho (m)
        self.declare_parameter('ds_metros', 0.5)
        # extensão da janela de waypoints à frente (m)
        self.declare_parameter('janela_waypoints_m', 35.0)

        # ---------------- Leitura dos parâmetros ----------------
        # lê o nome do tópico de odometria
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        # lê o nome do tópico de cones
        self.cones_topic = self.get_parameter('cones_topic').get_parameter_value().string_value
        # lê o frame do mapa
        self.frame_map = self.get_parameter('frame_map').get_parameter_value().string_value
        # lê o passo de reamostragem do caminho
        self.ds = self.get_parameter('ds_metros').get_parameter_value().double_value
        # lê o tamanho da janela de waypoints
        self.janela_wp = self.get_parameter('janela_waypoints_m').get_parameter_value().double_value

        # lê o raio de associação
        raio = self.get_parameter('associacao_raio').get_parameter_value().double_value
        # lê o alpha de suavização
        alpha = self.get_parameter('suavizacao_alpha').get_parameter_value().double_value
        # instancia o mapa de cones com os parâmetros
        self.mapa = MapaCones(raio_assoc=raio, alpha_suave=alpha)

        # ---------------- Estado interno ----------------
        # pose atual do carro (x, y, yaw)
        self.pose_atual = np.array([0.0, 0.0, 0.0], dtype=float)
        # flag indicando se já recebemos alguma odometria
        self.tem_pose = False
        # detecções mais recentes no frame do carro (x_b, y_b, lado)
        self.cones_body = np.empty((0, 3), dtype=float)

        # ---------------- ROS I/O ----------------
        # inscreve callback para odometria
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 50)
        # inscreve callback para detecções de cones
        self.sub_cones = self.create_subscription(Float32MultiArray, self.cones_topic, self.cones_cb, 10)
        # publisher para o caminho planejado (Path) no RViz
        self.pub_path = self.create_publisher(Path, '/planner/path', 10)
        # publisher para os waypoints (Float32MultiArray)
        self.pub_wps = self.create_publisher(Float32MultiArray, '/planner/waypoints', 10)
        # publisher de marcadores para visualizar cones mapeados
        self.pub_markers = self.create_publisher(MarkerArray, '/map_cones', 10)

        # cria um timer para rodar o laço principal a 20 Hz (0.05 s)
        self.timer = self.create_timer(0.05, self.tick)  # 20 Hz

    # publica os cones do mapa como marcadores no RViz
    def publicar_cones_markers(self):
        # obtém arrays de cones esquerdo e direito do mapa
        esq, der = self.mapa.lados()

        # cria um MarkerArray
        msg = MarkerArray()

        # cria um marcador de limpeza (deleta todos os marcadores anteriores)
        clear = Marker()
        clear.action = Marker.DELETEALL
        # adiciona o marcador de limpeza ao array
        msg.markers.append(clear)

        # função helper que cria um marcador de esfera para um cone
        def make_marker(x, y, mid, r, g, b):
            # instancia marcador
            m = Marker()
            # define o frame de referência do marcador
            m.header.frame_id = self.frame_map
            # coloca timestamp atual
            m.header.stamp = self.get_clock().now().to_msg()
            # namespace para agrupar
            m.ns = "cones_map"
            # id único dentro do namespace
            m.id = mid
            # tipo esfera (ball)
            m.type = Marker.SPHERE
            # ação de adicionar/modificar
            m.action = Marker.ADD
            # posição x,y,z do marcador
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.0
            # escala (diâmetros da esfera em x,y,z)
            m.scale.x = m.scale.y = m.scale.z = 0.35  # diâmetro da bolinha
            # cor RGBA do marcador
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
            # sem tempo de vida (persistente até ser substituído)
            m.lifetime.sec = 0
            # retorna o marcador configurado
            return m

        # inicializa um contador de IDs
        mid = 0
        # para cada cone esquerdo, adiciona marcador azul
        for x, y in esq:
            msg.markers.append(make_marker(x, y, mid, 0.0, 0.5, 1.0)); mid += 1
        # para cada cone direito, adiciona marcador amarelo
        for x, y in der:
            msg.markers.append(make_marker(x, y, mid, 1.0, 0.85, 0.0)); mid += 1

        # publica o MarkerArray no tópico /map_cones
        self.pub_markers.publish(msg)
    

    # --------- callbacks ---------

    # callback de odometria (atualiza pose atual do carro)
    def odom_cb(self, msg: Odometry):
        # extrai posição x e y
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        # extrai orientação como quaternion
        q = msg.pose.pose.orientation
        # converte quaternion para yaw (2D)
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        # salva pose atual (x,y,yaw)
        self.pose_atual[:] = [x, y, yaw]
        # marca que já temos pose válida
        self.tem_pose = True

    # callback de cones (detecções no frame do carro)
    def cones_cb(self, msg: Float32MultiArray):
        # Esperado: [x_b, y_b, lado, x_b, y_b, lado, ...]
        # converte os dados para array numpy float
        data = np.array(msg.data, dtype=float)
        # validação: precisa ser múltiplo de 3 e não vazio
        if data.size % 3 != 0 or data.size == 0:
            # se inválido, zera detecções
            self.cones_body = np.empty((0, 3), dtype=float)
            return
        # reestrutura para N x 3 (x_b,y_b,lado)
        self.cones_body = data.reshape(-1, 3)

    # --------- ciclo principal ---------
    # função acionada pelo timer (20 Hz)
    def tick(self):
        # se ainda não temos pose, não faz nada
        if not self.tem_pose:
            return

        # 1) pegar detecções no frame do carro
        if len(self.cones_body) > 0:
            # pega colunas x_b,y_b
            P_body = self.cones_body[:, :2]
            # pega coluna de lado e converte para int
            sides = self.cones_body[:, 2].astype(int)
            # 2) converter para mundo usando pose atual
            P_world = carro_para_mundo(P_body, self.pose_atual)
            # 3) atualizar mapa com as detecções transformadas
            self.mapa.atualizar(P_world, sides)

        # 4) gerar linha central e suavizar
        # obtém cones do mapa (todos até agora)
        esq, der = self.mapa.lados()
        # calcula centerline por triangulação de Delaunay
        centerline = centerline_delaunay(esq, der)
        # reamostra/suaviza a centerline com passo ds
        path = suaviza_caminho(centerline, passo=self.ds)

        # 5) waypoints
        # gera waypoints (x,y,yaw,kappa,s,v)
        wps = gera_waypoints(path)
        # recorta apenas os waypoints à frente numa janela de metros
        wps_win = corta_waypoints_a_frente(wps, self.pose_atual[:2], janela_m=self.janela_wp)

        # 6) publicar
        # se há path válido, publica o Path no RViz
        if path is not None and len(path) >= 2:
            self.publicar_path(path)
        # se há waypoints, publica para o controlador
        if len(wps_win) > 0:
            self.publicar_waypoints(wps_win)
        # publica marcadores dos cones do mapa
        self.publicar_cones_markers()


    # --------- publicadores ---------

    # publica a polilinha de path como nav_msgs/Path
    def publicar_path(self, path_xy: np.ndarray):
        # cria mensagem Path
        msg = Path()
        # carimba tempo atual
        msg.header.stamp = self.get_clock().now().to_msg()
        # define frame do mapa
        msg.header.frame_id = self.frame_map
        # percorre cada ponto do caminho
        for x, y in path_xy:
            # cria uma pose estampada
            ps = PoseStamped()
            # usa o mesmo header do Path
            ps.header = msg.header
            # define a posição x,y no mapa
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            # adiciona a pose na lista
            msg.poses.append(ps)
        # publica no tópico /planner/path
        self.pub_path.publish(msg)

    # publica os waypoints como Float32MultiArray “achatado”
    def publicar_waypoints(self, wps: np.ndarray):
        # Flatten: [x,y,yaw,kappa,s,v] * N
        # cria mensagem de array
        arr = Float32MultiArray()
        # converte para float32 e achata
        arr.data = wps.astype(np.float32).reshape(-1).tolist()
        # publica no tópico /planner/waypoints
        self.pub_wps.publish(arr)

# função principal para rodar o nó ROS2
def main():
    # inicializa o cliente ROS
    rclpy.init()
    # instancia o nó do planner
    node = DVPlannerNode()
    # entra no loop de callbacks
    rclpy.spin(node)
    # destrói o nó ao encerrar
    node.destroy_node()
    # encerra o cliente ROS
    rclpy.shutdown()

# ponto de entrada do script
if __name__ == '__main__':
    main()
