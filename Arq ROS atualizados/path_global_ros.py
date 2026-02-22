import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
import threading 
import time

LADO_ESQUERDO = 0.0  #Azul
LADO_DIREITO = 1.0   #Amarelo

#distância máxima que olhamos para frente
HORIZONTE_PLANEJAMENTO = 30.0  
#largura máxima entre um par de cones
LARGURA_MAX_PISTA = 12.0       
#diferença máxima de profundidade (Z) entre um cone azul e um amarelo
DELTA_Z_MAX = 5.0

#vizualização do gráfico
PLOT_VIEW_X_RADIUS = 20.0 
PLOT_VIEW_Z_FWD = 40.0    
PLOT_VIEW_Z_BWD = 10.0    

class PathPlanner(Node):
    def __init__(self):
        super().__init__('global_path_planner') 
        
        #posição global vinda do GPS
        self.car_x = 0.0 
        self.car_z = 0.0
        
        #GPS e câmera chegam em momentos diferentes, impede ler a posição X enquanto ela está sendo atualizada
        self.lock = threading.Lock() 

        #variáveis para o gráfico
        self.current_paired_cones = np.array([])
        self.current_midpoints = np.array([])

        self.cones_sub = self.create_subscription(
            Float32MultiArray, 
            '/cones', 
            self.conesCallback, 
            10
        )
        self.gps_sub = self.create_subscription(
            Float32MultiArray, 
            '/carGPS', 
            self.gpsCallback,
            10
        )
        
        self.waypoints_pub = self.create_publisher(
            Float32MultiArray, 
            '/waypoints', 
            10
        )
        
        #gráfico matplotlib
        plt.ion() 
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.configurar_plot()

    def configurar_plot(self):
        self.ax.set_title('Glbal')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Z (m)') 
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True)

    def gpsCallback(self, msg):
        #chamar toda vez que o GPS manda uma nova posição
        
        with self.lock:
            #se a mensagem tem(X, Y, Z)
            if len(msg.data) >= 3:
                #salva posição global de agora
                self.car_x = msg.data[0]
                self.car_z = msg.data[2] #índice 2 como Z global

    def _update_plot(self):
        
        with self.lock:
            cones_data = self.current_paired_cones
            midpoints_data = self.current_midpoints
            plot_center_x = self.car_x 
            plot_center_z = self.car_z
    
        self.ax.clear()
        self.configurar_plot()
        
        self.ax.set_xlim([plot_center_x - PLOT_VIEW_X_RADIUS, plot_center_x + PLOT_VIEW_X_RADIUS]) 
        self.ax.set_ylim([plot_center_z - PLOT_VIEW_Z_BWD, plot_center_z + PLOT_VIEW_Z_FWD]) 
        self.ax.scatter(plot_center_x, plot_center_z, c='green', marker='^', s=150, label='Carro', zorder=5)

        if len(cones_data) > 0:
            cones_left = cones_data[cones_data[:, 2] == LADO_ESQUERDO] 
            cones_right = cones_data[cones_data[:, 2] == LADO_DIREITO]

            if len(cones_left) > 0:
                self.ax.scatter(cones_left[:, 0], cones_left[:, 1], c='blue', marker='o', s=50, label='Esq')
            if len(cones_right) > 0:
                self.ax.scatter(cones_right[:, 0], cones_right[:, 1], c='yellow', marker='o', s=50, label='Dir')

        if len(midpoints_data) > 0:
            midpoints_sorted = midpoints_data[np.argsort(midpoints_data[:, 1])]
            self.ax.plot(midpoints_sorted[:, 0], midpoints_sorted[:, 1], 'r-x', markersize=10, label='Waypoints', zorder=3)

        self.ax.legend(loc='upper right')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def conesCallback(self, msg):
        #recebe a lista de cones locais [x, z, id] e transforma em globais
        try:
            #converte a lista plana em matriz [N x 3]
            data_raw = np.array(msg.data, dtype=np.float32).reshape(-1, 3) 
        except:
            return

        if len(data_raw) == 0: 
            return

        #posição do carro no momento da leitura
        with self.lock: 
            car_x_atual = self.car_x
            car_z_atual = self.car_z

        working_data_global = []

        #TRANSFORMAÇÃO LOCAL PARA GLOBAL 
        for cone in data_raw:
            x_local = cone[0] #distância lateral em relação ao carro
            z_local = cone[1] # distância frontal em relação ao carro
            label_id = cone[2] # Cor (0 ou 1)
            
            #se o cone está muito longe ou atrás do carro (z < 0)
            if z_local > HORIZONTE_PLANEJAMENTO:
                continue #ignora o cone
            
            if z_local <= 0.0:
                continue # ignora o cone

            #Global: Posição Mundo = Posição Carro + Posição Relativa
            X_map = car_x_atual + x_local
            Z_map = car_z_atual + z_local 
            
            working_data_global.append([X_map, Z_map, label_id])
           
        working_data = np.array(working_data_global)
        
        #se todos os cones foram filtrados (nenhum válido)
        if len(working_data) == 0:
             return

        # Separa esquerda (Azul) direita (Amarelo)
        cones_left_data = working_data[working_data[:, 2] == LADO_ESQUERDO] 
        cones_right_data = working_data[working_data[:, 2] == LADO_DIREITO]
        
        #pareamento inteligente
        paired_midpoints, paired_cones_full = self._pair_and_calculate(cones_left_data, cones_right_data)

        if len(paired_midpoints) > 0:
            with self.lock:
                    self.current_paired_cones = paired_cones_full
                    self.current_midpoints = paired_midpoints
                    
            local_waypoints = np.copy(paired_midpoints)
            local_waypoints[:, 0] -= car_x_atual
            local_waypoints[:, 1] -= car_z_atual
            local_waypoints = local_waypoints[np.argsort(local_waypoints[:, 1])]
                    
            self._publish_waypoints(local_waypoints)
            self._update_plot()
           
    def _pair_and_calculate(self, cones_left_data, cones_right_data):
        #encontra pares de cones e calcula o ponto médio
        
        #faltou cones de um dos lados
        if len(cones_left_data) == 0:
             return np.array([]), np.array([])
        
        if len(cones_right_data) == 0:
             return np.array([]), np.array([])

        #ordena os cones da esquerda pela profundidade (Z)
        cones_left_data = cones_left_data[np.argsort(cones_left_data[:, 1])]
        midpoints = []
        all_paired_cones = []
        
        #para cada cone Azul
        for cone_L_full in cones_left_data:
            cone_L_x = cone_L_full[0]
            cone_L_z = cone_L_full[1]
            
            #vamos procurar o cone Amarelo mais próximo em Z
            #calcula a diferença de profundidade para TODOS os amarelos
            delta_z = np.abs(cones_right_data[:, 1] - cone_L_z)
            
            #índice do menor valor (o mais alinhado em Z)
            closest_z_idx = np.argmin(delta_z)
            
            #cone amarelo candidato
            cone_R_full = cones_right_data[closest_z_idx]
            cone_R_x = cone_R_full[0]
            cone_R_z = cone_R_full[1]
            
            #distância Lateral (X)
            distancia_x = np.abs(cone_L_x - cone_R_x)
            
            #distância Profundidade (Z)
            distancia_z = np.abs(cone_L_z - cone_R_z)
            
            #validade do par
            if distancia_x < LARGURA_MAX_PISTA:
                if distancia_z < DELTA_Z_MAX:
                    
                    #ponto médio
                    mid_x = (cone_L_x + cone_R_x) / 2.0
                    mid_z = (cone_L_z + cone_R_z) / 2.0
                    midpoints.append([mid_x, mid_z])
                    all_paired_cones.append(cone_L_full) 
                    all_paired_cones.append(cone_R_full)

        if len(midpoints) > 0:
            return np.array(midpoints), np.array(all_paired_cones)
        else:
            return np.array([]), np.array([])

    def _publish_waypoints(self, waypoints):
        msg = Float32MultiArray()
        #transforma [[x1,z1], [x2,z2]] em [x1, z1, x2, z2] e converte para lista ROS aceita
        msg.data = waypoints.flatten().tolist()
        
        self.waypoints_pub.publish(msg)

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