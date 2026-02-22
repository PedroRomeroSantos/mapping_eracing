import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
import time

class PathPlanner(Node):
    def __init__(self):
        super().__init__('planner_local')
        
        #1 para evitar parear cones colados
        self.LARGURA_MIN = 1.0   
        #10 porque a projeção da câmera estava esticando
        self.LARGURA_MAX = 10.0   
        #tolerância de 5 na profundidade (Z) para aceitar pares em curvas
        self.DELTA_Z_MAX = 5.0    
        
        self.cones_sub = self.create_subscription(Float32MultiArray, '/cones', self.conesCallback, 10)
        self.waypoints_pub = self.create_publisher(Float32MultiArray, '/waypoints', 10)
        
        #variáveis gráfico
        self.data_esq = [[], []]
        self.data_dir = [[], []]
        self.data_wps = [[], []]
        self.target = [None, None]

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 8))
        self.ax.set_aspect('equal', adjustable='box') 
        
        #objetos gráficos
        self.line_esq, = self.ax.plot([], [], 'bo', markersize=7, label='Esq (Azul)')
        self.line_dir, = self.ax.plot([], [], 'o', color='gold', markersize=7, label='Dir (Amarelo)')
        self.line_path, = self.ax.plot([], [], 'r-', linewidth=2.5, alpha=0.6)
        self.dots_wps, = self.ax.plot([], [], 'rx', markersize=10, label='Waypoints')
        self.scat_target, = self.ax.plot([], [], 'go', markersize=12, markerfacecolor='none', label='ALVO')
        
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([0, 30])
        self.ax.grid(True, linestyle=':', alpha=0.5)
        
        #carro na origem (0,0) como um triângulo verde
        self.ax.plot(0, 0, 'g^', markersize=15) 
        
        self.ax.legend(loc='upper right', fontsize='small')
        plt.show(block=False)

    def conesCallback(self, msg):
        #mensagem no tópico /cones.
        #array plano [x1, z1, id1, x2, z2, id2, ...]

        try:
            #transforma a lista plana em uma matriz, linha é [X, Z, ID]
            data = np.array(msg.data, dtype=np.float32).reshape(-1, 3)
        except:
            return

        #cones por cor (ID 0.0 = Azul, ID 1.0 = Amarelo)
        #todas as linhas, mas só as colunas 0 e 1 (X e Z)
        esq = data[data[:, 2] == 0.0][:, :2]
        dir_ = data[data[:, 2] == 1.0][:, :2]
        wps = [] #lista para guardar os pontos médios (waypoints)

        #só tentamos parear se existirem cones dos dois lados
        if len(esq) > 0 and len(dir_) > 0:
            
            #ordenamos os cones da esquerda pela profundidade
            #criauma trajetória sequencial do mais perto para o mais longe
            esq = esq[np.argsort(esq[:, 1])] 
            
            #encontrar o par de cada cone azul
            for p_esq in esq:
                
                #calcula a diferença de profundidade (Z) entre este cone azul e TODOS os amarelos
                diff_z = np.abs(dir_[:, 1] - p_esq[1])
                
                #cria uma "máscara" (lista de True ou False) para amarelos que estão próximos em Z
                mask_z = diff_z < self.DELTA_Z_MAX
                
                #se houver pelo menos um cone amarelo compatível
                if np.any(mask_z):
                    #filtra a lista de amarelos
                    candidatos = dir_[mask_z]
                    
                    #dentre os candidatos válidos em Z, qual está mais alinhado em X? lateralmente
                    diff_x = np.abs(candidatos[:, 0] - p_esq[0])
                    idx_local = np.argmin(diff_x) #pega o índice do menor valor que seria o mais alinhado lateralmente
                    
                    p_dir = candidatos[idx_local]

                    #calcula a distância real (Euclidiana) entre o cone azul e o amarelo escolhido
                    dist_total = np.linalg.norm(p_esq - p_dir)
                    
                    #se distância for aceitável
                    if dist_total < self.LARGURA_MAX and dist_total > self.LARGURA_MIN:
                        #waypoint é a média das posições
                        ponto_medio = (p_esq + p_dir) / 2.0
                        wps.append(ponto_medio)

        wps = np.array(wps)
        # atualização dos cones
        if len(esq) > 0:
            self.data_esq = [esq[:, 0], esq[:, 1]]
        else:
            self.data_esq = [[], []]

        if len(dir_) > 0:
            self.data_dir = [dir_[:, 0], dir_[:, 1]]
        else:
            self.data_dir = [[], []]
        
        if len(wps) > 0:
            #ordena os waypoints por profundidade
            wps = wps[np.argsort(wps[:, 1])]
            self.data_wps = [wps[:, 0], wps[:, 1]]
            
            #target é o waypoint imediato
            self.target = [wps[0, 0], wps[0, 1]]
            
            msg_out = Float32MultiArray()
            #transforma [[x1,z1], [x2,z2]] em [x1, z1, x2, z2]
            msg_out.data = wps.flatten().tolist()
            self.waypoints_pub.publish(msg_out)
        else:
            self.data_wps = [[], []]
            self.target = [None, None]

    def update_plot(self):
        #atualiza matplotlib
        if not plt.fignum_exists(self.fig.number):
            return

        # Atualiza os dados de cada linha no gráfico
        self.line_esq.set_data(self.data_esq[0], self.data_esq[1])
        self.line_dir.set_data(self.data_dir[0], self.data_dir[1])
        self.line_path.set_data(self.data_wps[0], self.data_wps[1])
        self.dots_wps.set_data(self.data_wps[0], self.data_wps[1])
        
        #atualiza o círculo do alvo
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
            
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()