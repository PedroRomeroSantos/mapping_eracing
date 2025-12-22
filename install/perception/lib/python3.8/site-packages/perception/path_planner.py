import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
import threading 
import time

# --- CONSTANTES DE CONFIGURAÇÃO (Global) ---
LADO_ESQUERDO = -1.0
LADO_DIREITO = 1.0
HORIZONTE_PLANEJAMENTO = 35.0  # Profundidade máxima (Z) para o planejamento local
LARGURA_MAX_PISTA = 12.0       # Largura máxima aceitável para um par de cones

# Mapeamento ID para valor numérico do Lado
ID_TO_SIDE_MAP = {
    0: LADO_ESQUERDO,  # ID 0: Esquerda (AZUL)
    1: LADO_DIREITO,   # ID 1: Direita (AMARELO)
    2: 0.0,            # ID 2: Ignorar (Neutro)
}

class PathPlanner(Node):
    def __init__(self):
        super().__init__('local_target_visualizer')
        
        # Variáveis de Estado (Apenas a vista atual)
        self.current_paired_cones = np.array([])
        self.current_midpoints = np.array([])
        self.lock = threading.Lock() 
        
        # Subscribers e Publishers
        self.cones_sub = self.create_subscription(
            Float32MultiArray, 
            '/cones_depth', 
            self.conesCallback, 
            10
        )
        
        self.waypoints_pub = self.create_publisher(
            Float32MultiArray, 
            '/trajectory_waypoints', 
            10
        )
        
        # Configuração do Matplotlib para Modo Interativo (Tempo Real)
        plt.ion() 
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_title('Alvo Local (Frame do Carro)')
        self.ax.set_xlabel('Posição Lateral X (m)')
        self.ax.set_ylabel('Profundidade Z (m)') 
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True)
        self.ax.set_xlim([-15, 15]) # Define uma área de visualização local razoável
        self.ax.set_ylim([0, 40]) # De 0 a 40m à frente

        self.get_logger().info('Local Target Visualizer started. Gráfico Matplotlib ativo.')


    def _update_plot(self):
        """Limpa o gráfico e redesenha o cenário local atualizado."""
        
        with self.lock:
            cones_data = self.current_paired_cones
            midpoints_data = self.current_midpoints
        
        if len(cones_data) == 0 and len(midpoints_data) == 0:
            # Não desenha nada se não houver dados
            return

        # Limpar o gráfico
        self.ax.clear()
        self.ax.set_title('Alvo Local (Frame do Carro)')
        self.ax.set_xlabel('Posição Lateral X (m)')
        self.ax.set_ylabel('Profundidade Z (m)')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True)
        self.ax.set_xlim([-15, 15]) 
        self.ax.set_ylim([0, 40]) 
        
        # Separa os dados para plotagem
        cones_left = cones_data[cones_data[:, 2] == LADO_ESQUERDO] 
        cones_right = cones_data[cones_data[:, 2] == LADO_DIREITO]

        # 1. Plotar Cones Esquerdo e Direito
        if len(cones_left) > 0:
            self.ax.scatter(cones_left[:, 0], cones_left[:, 1], c='blue', marker='o', s=50, label='Cones Esquerdo')
        if len(cones_right) > 0:
            self.ax.scatter(cones_right[:, 0], cones_right[:, 1], c='yellow', marker='o', s=50, label='Cones Direito')

        # 2. Plotar Midpoints (Alvos)
        if len(midpoints_data) > 0:
            # Ponto central deve ser o primeiro waypoint (target imediato)
            midpoints_sorted = midpoints_data[np.argsort(midpoints_data[:, 1])]
            
            # Ponto Central (Target Imediato)
            self.ax.scatter(midpoints_sorted[:, 0], midpoints_sorted[:, 1], c='red', marker='X', s=100, label='Waypoint (Centro)', zorder=3)


        self.ax.legend(loc='upper right')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def conesCallback(self, msg):
        """Recebe cones (X_car, Z_car, color_id), pareia, publica e atualiza a visualização."""
        
        data = np.array(msg.data, dtype=np.float32).reshape(-1, 3) 
        if len(data) < 2: return

        sides = np.array([ID_TO_SIDE_MAP.get(int(cid), 0.0) for cid in data[:, 2]])
        working_data = np.column_stack((data[:, 0], data[:, 1], sides)) 
        working_data = working_data[working_data[:, 2] != 0.0] # Remove neutros
        
        # 2. Filtragem Local (Horizonte de Planejamento)
        data_filtrada = working_data[
            # CORREÇÃO: Acessa a variável global HORIZONTE_PLANEJAMENTO
            (working_data[:, 1] < HORIZONTE_PLANEJAMENTO) & 
            (working_data[:, 1] > 0.0)
        ]
        
        if len(data_filtrada) < 2: 
            self.get_logger().warn("Poucos cones para formar um waypoint.")
            return

        # 3. Separar Lados
        cones_left_data = data_filtrada[data_filtrada[:, 2] == LADO_ESQUERDO] 
        cones_right_data = data_filtrada[data_filtrada[:, 2] == LADO_DIREITO]
        
        if len(cones_left_data) < 1 or len(cones_right_data) < 1: return

        # 4. Pareamento e Cálculo do Waypoint Local
        paired_midpoints, paired_cones_full = self._pair_and_calculate(cones_left_data, cones_right_data)

        if len(paired_midpoints) > 0:
            # 5. Armazena para Plotagem (Substitui o cenário anterior)
            with self.lock:
                 self.current_paired_cones = paired_cones_full
                 self.current_midpoints = paired_midpoints
                 
            # 6. Publica os Waypoints (para o controle)
            self._publish_waypoints(paired_midpoints)
            
            # 7. Atualiza o Gráfico (Tempo Real)
            self._update_plot()
            
    def _pair_and_calculate(self, cones_left_data, cones_right_data):
        """Pareia os cones e calcula os midpoints."""
        
        cones_left_data = cones_left_data[np.argsort(cones_left_data[:, 1])]
        midpoints = []
        all_paired_cones = []
        
        for cone_L_full in cones_left_data:
            cone_L = cone_L_full[:2] 
            
            # Encontra o cone da direita com o Z mais próximo
            delta_z = np.abs(cones_right_data[:, 1] - cone_L[1])
            closest_z_idx = np.argmin(delta_z)
            cone_R_full = cones_right_data[closest_z_idx]
            cone_R = cone_R_full[:2] 
            
            # Verificação de segurança (Largura X)
            distance_x = np.abs(cone_L[0] - cone_R[0])
            
            # CORREÇÃO: LARGURA_MAX_PISTA deve ser acessada globalmente
            if distance_x < LARGURA_MAX_PISTA and np.abs(cone_L[1] - cone_R[1]) < 10.0: 
                
                midpoint = (cone_L + cone_R) / 2.0
                midpoints.append(midpoint)
                
                # Armazena os cones inteiros para plotagem (CL e CR)
                all_paired_cones.append(cone_L_full) 
                all_paired_cones.append(cone_R_full)

        return np.array(midpoints), np.array(all_paired_cones)

    def _publish_waypoints(self, waypoints):
        """Publica a lista final de waypoints (x, z) no tópico ROS2."""
        msg = Float32MultiArray()
        msg.data = waypoints.flatten().tolist()
        self.waypoints_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    import threading 
    node = PathPlanner()
    
    # O Matplotlib Interativo deve ser rodado no main thread do ROS
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Nó encerrado pelo usuário.')
    finally:
        # Garante que a figura do Matplotlib seja fechada corretamente
        plt.close(node.fig)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()