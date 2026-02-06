import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt

FX = 960.0
CX = 960.0

LADO_ESQUERDO = -1.0
LADO_DIREITO = 1.0
HORIZONTE_MAXIMO = 30.0   # ver mais pares na reta
LARGURA_MAXIMA = 12.0    

ID_CORES = {
    0: LADO_ESQUERDO,
    1: LADO_DIREITO,
    2: 0.0,            
}

def converter_pixel_para_metro(u, z_depth):
    x_local = (u - CX) * z_depth / FX
    return x_local

class PathPlanner(Node):
    def __init__(self):
        super().__init__('planner_simples_visual')
        
        self.cones_sub = self.create_subscription(
            Float32MultiArray, '/cones', self.receber_cones, 10
        )
        
        self.waypoints_pub = self.create_publisher(
            Float32MultiArray, '/trajectory_waypoints', 10
        )
        
        # GRÁFICO
        self.visualizacao_ativa = True
        if self.visualizacao_ativa:
            plt.ion() 
            self.fig, self.ax = plt.subplots(figsize=(6, 8))
            self.configurar_plot()

        self.get_logger().info('Planner com Visualização Iniciado!')

    def configurar_plot(self):
        self.ax.set_title('Visão Local (Carro em 0,0)')
        self.ax.set_xlabel('Lateral X (m)')
        self.ax.set_ylabel('Frente Z (m)')
        self.ax.set_xlim([-10, 10]) 
        self.ax.set_ylim([0, 35])
        self.ax.grid(True)
        
        #carro verde
        self.ax.plot(0, 0, 'g^', markersize=15, label='Carro')

    def atualizar_grafico(self, esq, dir, wps):
        """Atualiza os pontos na tela."""
        if not self.visualizacao_ativa: return

        self.ax.clear()
        self.configurar_plot()

        #Cones esquerda (Azul)
        if len(esq) > 0:
            self.ax.scatter(esq[:, 0], esq[:, 1], c='blue', label='Esq')
        
        #Cones direita (Amarelo)
        if len(dir) > 0:
            self.ax.scatter(dir[:, 0], dir[:, 1], c='orange', label='Dir')

        # Plota Waypoints (Vermelho)
        if len(wps) > 0:
            self.ax.plot(wps[:, 0], wps[:, 1], 'r--', alpha=0.5) # Linha
            self.ax.scatter(wps[:, 0], wps[:, 1], c='red', marker='x', s=80, label='Target')
            #o alvo imediato
            self.ax.scatter(wps[0, 0], wps[0, 1], c='darkred', s=150)

        self.ax.legend(loc='upper right')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def receber_cones(self, msg):
        dados_brutos = np.array(msg.data, dtype=np.float32).reshape(-1, 4)
        
        if len(dados_brutos) == 0:
            return

        cones_esquerda = []
        cones_direita = []

        #Converter e separar
        for cone in dados_brutos:
            u, v, z, id_cor = cone
            
            if z > HORIZONTE_MAXIMO or z < 0.5: continue

            lado = ID_CORES.get(int(id_cor), 0.0)
            if lado == 0.0: continue

            x_local = converter_pixel_para_metro(u, z)
            
            if lado == LADO_ESQUERDO:
                cones_esquerda.append([x_local, z])
            elif lado == LADO_DIREITO:
                cones_direita.append([x_local, z])

        cones_esquerda = np.array(cones_esquerda)
        cones_direita = np.array(cones_direita)

        # Calcular caminho
        waypoints = np.array([])
        if len(cones_esquerda) > 0 and len(cones_direita) > 0:
            waypoints = self.calcular_ponto_medio(cones_esquerda, cones_direita)

        #Publicar
        if len(waypoints) > 0:
            waypoints = waypoints[np.argsort(waypoints[:, 1])]
            msg_saida = Float32MultiArray()
            msg_saida.data = waypoints.flatten().tolist()
            self.waypoints_pub.publish(msg_saida)

        #Atualizar visualização
        self.atualizar_grafico(cones_esquerda, cones_direita, waypoints)

    def calcular_ponto_medio(self, esquerda, direita):
        pontos_medios = []
        indices_usados = set() #evita repetir o mesmo cone da direita
        
        #ordena para tentar parear por proximidade
        esquerda = esquerda[np.argsort(esquerda[:, 1])]
        direita = direita[np.argsort(direita[:, 1])]

        for cone_esq in esquerda:
            x_esq, z_esq = cone_esq[0], cone_esq[1]
            
            melhor_idx = -1
            menor_diff = float('inf')

            #procura o melhor par na direita
            for i, cone_dir in enumerate(direita):
                if i in indices_usados: continue #se já foi usado por outro cone

                z_dir = cone_dir[1]
                diff_z = abs(z_esq - z_dir)

                #Diferença de Z menor que 5m e o menor que encontrar
                if diff_z < 5.0 and diff_z < menor_diff:
                    menor_diff = diff_z
                    melhor_idx = i
            
            #achou um par válido
            if melhor_idx != -1:
                cone_dir = direita[melhor_idx]
                x_dir, z_dir = cone_dir[0], cone_dir[1]
                
                largura = abs(x_esq - x_dir)
                
                if largura < LARGURA_MAXIMA:
                    meio_x = (x_esq + x_dir) / 2.0
                    meio_z = (z_esq + z_dir) / 2.0
                    pontos_medios.append([meio_x, meio_z])
                    indices_usados.add(melhor_idx)

        return np.array(pontos_medios)

def main(args=None):
    rclpy.init(args=args)
    node = PathPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(node.fig)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()