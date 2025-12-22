import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
import threading 
import time

fx = 960.0
fy = 540.0
cx = 960.0
cy = 540.0

LADO_ESQUERDO = -1.0
LADO_DIREITO = 1.0
HORIZONTE_PLANEJAMENTO = 30.0  
LARGURA_MAX_PISTA = 12.0       

PLOT_VIEW_X_RADIUS = 20.0 
PLOT_VIEW_Z_FWD = 40.0    
PLOT_VIEW_Z_BWD = 10.0    


ID_TO_SIDE_MAP = {
    0: LADO_ESQUERDO,  # AZUL
    1: LADO_DIREITO,   # AMARELO
    2: 0.0,            # Ignorar
}

# FUNÇÃO DE TRANSFORMAÇÃO PARA COORDENADAS GLOBAIS
def positionMap(pixelU:int, pixelV:int, distance:float):
    """Converte coordenadas do pixel + distância para coordenadas 3D no frame do carro (X_carro, Y_carro, Z_carro)."""
    bottomPart = np.sqrt(1 + ((pixelU-cx)**2)/(fx**2) + ((pixelV-cy)**2)/(fy**2))
        
    z = distance / bottomPart # profundidade do carro (Z_w)
    x = (pixelU - cx) * (z / fx) # lateral do carro (X_w)
    y = (pixelV - cy) * (z / fy) 
    return x, y, z

class PathPlanner(Node):
    def __init__(self):
        super().__init__('global_path_planner') 
        
        # Variáveis de Estado globais
        self.current_paired_cones = np.array([])
        self.current_midpoints = np.array([])
        self.car_x = 0.0 
        self.car_z = 0.0
        self.lock = threading.Lock() 

        self.cones_sub = self.create_subscription(Float32MultiArray,'/cones',self.conesCallback, 10)
        self.gps_sub = self.create_subscription(Float32MultiArray,'/carGPS', self.gpsCallback,10)
        self.waypoints_pub = self.create_publisher(Float32MultiArray, '/trajectory_waypoints', 10)
        
        #matplotlib global
        plt.ion() 
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_title('Planejamento')
        self.ax.set_xlabel('Posição Global X (m)')
        self.ax.set_ylabel('Posição Global Z (m)') 
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True)

    def gpsCallback(self, msg):
        """Recebe e armazena a posição global do carro."""
        with self.lock:
            len(msg.data) >= 3
            self.car_x = msg.data[0]
            self.car_z = msg.data[2] 

    def _update_plot(self):
        """Limpa o gráfico e refaz para o frame"""
        
        with self.lock:
            cones_data = self.current_paired_cones
            midpoints_data = self.current_midpoints
            plot_center_x = self.car_x #centro do gráfico é o carro
            plot_center_z = self.car_z
        
        self.ax.clear()
        self.ax.set_title('Planejamento')
        self.ax.set_xlabel('Posição Global X (m)')
        self.ax.set_ylabel('Posição Global Z (m)')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True)
        
        #limites dinâmicos para centralizar no carro
        self.ax.set_xlim([plot_center_x - PLOT_VIEW_X_RADIUS, plot_center_x + PLOT_VIEW_X_RADIUS]) 
        self.ax.set_ylim([plot_center_z - PLOT_VIEW_Z_BWD, plot_center_z + PLOT_VIEW_Z_FWD]) 
        
        #plota o carro
        self.ax.scatter(plot_center_x, plot_center_z, c='green', marker='^', s=150, label='Carro', zorder=5)

        if len(cones_data) > 0:
            cones_left = cones_data[cones_data[:, 2] == LADO_ESQUERDO] 
            cones_right = cones_data[cones_data[:, 2] == LADO_DIREITO]

            if len(cones_left) > 0:
                self.ax.scatter(cones_left[:, 0], cones_left[:, 1], c='blue', marker='o', s=50, label='Cones Esquerdo')
            if len(cones_right) > 0:
                self.ax.scatter(cones_right[:, 0], cones_right[:, 1], c='yellow', marker='o', s=50, label='Cones Direito')

        if len(midpoints_data) > 0:
            midpoints_sorted = midpoints_data[np.argsort(midpoints_data[:, 1])]
            #plota linha e marcadores
            self.ax.plot(midpoints_sorted[:, 0], midpoints_sorted[:, 1], 'r-x', markersize=10, label='Waypoints (Global)', zorder=3)

        self.ax.legend(loc='upper right')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def conesCallback(self, msg):
        """recebe [u, v, Z, ID], converte para GLOBAIS."""
    
        data_raw = np.array(msg.data, dtype=np.float32).reshape(-1, 4) 
        
        if len(data_raw) == 0: 
            with self.lock:
                self.current_paired_cones = np.array([])
                self.current_midpoints = np.array([])
            self._update_plot()
            return

        # Pega a posição ATUAL do carro
        with self.lock: 
            car_x_atual = self.car_x
            car_z_atual = self.car_z

        #converte de Pixel para Coordenadas GLOBAIS
        working_data_global = []
        working_data_local = [] # Criamos uma lista para os locais

        for cone_raw in data_raw:
            u, v, z_depth, label_id = cone_raw
            
            #Pixel -> Coordenadas Locais (X, Z)
            X_w, Y_c, Z_w = positionMap(u, v, z_depth)
            
            #filtro de horizonte local
            if Z_w > HORIZONTE_PLANEJAMENTO or Z_w <= 0.0:
                continue

            # Local (Para o controle)
            side = ID_TO_SIDE_MAP.get(int(label_id), 0.0)
            if side != 0.0:
                working_data_local.append([X_w, Z_w, side]) # Guarda o local
                
            # Local -> Coordenadas Globais
            X_map = car_x_atual + X_w
            Z_map = car_z_atual + Z_w 
            
            side = ID_TO_SIDE_MAP.get(int(label_id), 0.0)
            if side == 0.0:
                continue
                
            working_data_global.append([X_map, Z_map, side])
            
        working_data = np.array(working_data_global)
        
        #separar lados GLOBAIS
        cones_left_data = working_data[working_data[:, 2] == LADO_ESQUERDO] 
        cones_right_data = working_data[working_data[:, 2] == LADO_DIREITO]
        

        #pareamento GLOBAL
        paired_midpoints, paired_cones_full = self._pair_and_calculate(cones_left_data, cones_right_data)

        if len(paired_midpoints) > 0:
            with self.lock:
                 self.current_paired_cones = paired_cones_full
                 self.current_midpoints = paired_midpoints
                 
            self._publish_waypoints(paired_midpoints)
            self._update_plot()
            
    def _pair_and_calculate(self, cones_left_data, cones_right_data):
        """Pareia os cones e calcula os midpoints (em coordenadas globais)."""
        
        #ordena por Z GLOBAL
        cones_left_data = cones_left_data[np.argsort(cones_left_data[:, 1])]
        midpoints = []
        all_paired_cones = []
        
        if len(cones_right_data) == 0: 
            return np.array(midpoints), np.array(all_paired_cones)
            
        for cone_L_full in cones_left_data:
            cone_L = cone_L_full[:2] 
            
            delta_z = np.abs(cones_right_data[:, 1] - cone_L[1])
            closest_z_idx = np.argmin(delta_z)
            cone_R_full = cones_right_data[closest_z_idx]
            cone_R = cone_R_full[:2] 
            
            distance_x = np.abs(cone_L[0] - cone_R[0])
            
            if distance_x < LARGURA_MAX_PISTA and np.abs(cone_L[1] - cone_R[1]) < 10.0: 
                midpoint = (cone_L + cone_R) / 2.0
                midpoints.append(midpoint)
                all_paired_cones.append(cone_L_full) 
                all_paired_cones.append(cone_R_full)

        return np.array(midpoints), np.array(all_paired_cones)

    def _publish_waypoints(self, waypoints):
        """Publica a lista final de waypoints (X_map, Z_map) no tópico ROS2."""
        msg = Float32MultiArray()
        if waypoints.shape[0] > 0:
            waypoints_sorted = waypoints[np.argsort(waypoints[:, 1])]
            msg.data = waypoints_sorted.flatten().tolist()
        else:
            msg.data = []
        self.waypoints_pub.publish(msg)

def main():
    rclpy.init()
    node = PathPlanner()
    rclpy.spin(node)
    plt.close(node.fig)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()