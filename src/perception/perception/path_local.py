import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
import time

class PathPlanner(Node):
    def __init__(self):
        super().__init__('planner_local_visual')
        
        # Parâmetros relaxados para capturar os cones da sua imagem
        self.LARGURA_MIN = 1.0   
        self.LARGURA_MAX = 10.0   # Aumentado para aceitar os 7.5m que aparecem no seu gráfico
        self.DELTA_Z_MAX = 5.0    # Maior tolerância de profundidade entre cones do mesmo par
        
        self.cones_sub = self.create_subscription(Float32MultiArray, '/cones', self.conesCallback, 10)
        self.waypoints_pub = self.create_publisher(Float32MultiArray, '/trajectory_waypoints', 10)
        
        self.data_esq, self.data_dir, self.data_wps = [[], []], [[], []], [[], []]
        self.target = [None, None]

        plt.ion() 
        self.fig, self.ax = plt.subplots(figsize=(6, 8))
        self.ax.set_aspect('equal', adjustable='box') # Garante que 1m em X seja igual a 1m em Z
        
        # Elementos do gráfico
        self.line_esq, = self.ax.plot([], [], 'bo', markersize=7, label='Esq (Azul)')
        self.line_dir, = self.ax.plot([], [], 'o', color='gold', markersize=7, label='Dir (Amarelo)')
        self.line_path, = self.ax.plot([], [], 'r-', linewidth=2.5, alpha=0.6)
        self.dots_wps, = self.ax.plot([], [], 'rx', markersize=10, label='Waypoints')
        self.scat_target, = self.ax.plot([], [], 'go', markersize=12, markerfacecolor='none', label='ALVO')
        
        self.ax.set_xlim([-10, 10]); self.ax.set_ylim([0, 30])
        self.ax.grid(True, linestyle=':', alpha=0.5)
        self.ax.plot(0, 0, 'g^', markersize=15) # Carro em (0,0)
        
        self.ax.legend(loc='upper right', fontsize='small')
        plt.show(block=False)

    def conesCallback(self, msg):
        try:
            data = np.array(msg.data, dtype=np.float32).reshape(-1, 3)
        except: return

        esq = data[data[:, 2] == 0.0][:, :2]
        dir_ = data[data[:, 2] == 1.0][:, :2]
        wps = []

        if len(esq) > 0 and len(dir_) > 0:
            esq = esq[np.argsort(esq[:, 1])] # Ordena por profundidade (Z)
            indices_usados = set()
            
            for p_esq in esq:
                # Lógica Delta-Z: busca cones amarelos com profundidade similar
                diff_z = np.abs(dir_[:, 1] - p_esq[1])
                mask_z = diff_z < self.DELTA_Z_MAX
                
                if np.any(mask_z):
                    candidatos = dir_[mask_z]
                    # Dos que estão perto em Z, pega o mais próximo lateralmente (X)
                    diff_x = np.abs(candidatos[:, 0] - p_esq[0])
                    idx_local = np.argmin(diff_x)
                    
                    p_dir = candidatos[idx_local]
                    dist_total = np.linalg.norm(p_esq - p_dir)
                    
                    # Verifica se o par está dentro da nova largura de 10m
                    if dist_total < self.LARGURA_MAX and dist_total > self.LARGURA_MIN:
                        wps.append((p_esq + p_dir) / 2.0)

        wps = np.array(wps)
        self.data_esq = [esq[:, 0], esq[:, 1]] if len(esq) > 0 else [[], []]
        self.data_dir = [dir_[:, 0], dir_[:, 1]] if len(dir_) > 0 else [[], []]
        
        if len(wps) > 0:
            wps = wps[np.argsort(wps[:, 1])]
            self.data_wps = [wps[:, 0], wps[:, 1]]
            self.target = [wps[0, 0], wps[0, 1]]
            
            # Publica para o controle
            msg_out = Float32MultiArray()
            msg_out.data = wps.flatten().tolist()
            self.waypoints_pub.publish(msg_out)
        else:
            self.data_wps = [[], []]; self.target = [None, None]

    def update_plot(self):
        if not plt.fignum_exists(self.fig.number): return
        self.line_esq.set_data(self.data_esq[0], self.data_esq[1])
        self.line_dir.set_data(self.data_dir[0], self.data_dir[1])
        self.line_path.set_data(self.data_wps[0], self.data_wps[1])
        self.dots_wps.set_data(self.data_wps[0], self.data_wps[1])
        if self.target[0] is not None:
            self.scat_target.set_data([self.target[0]], [self.target[1]])
        else:
            self.scat_target.set_data([], [])
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

def main():
    rclpy.init()
    node = PathPlanner()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)
            node.update_plot()
            time.sleep(0.01)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node(); rclpy.shutdown()