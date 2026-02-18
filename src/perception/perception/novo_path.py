import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
import threading 
import time

# --- Configurações ---
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
    0: LADO_ESQUERDO,
    1: LADO_DIREITO,
    2: 0.0,
}

def positionMap(pixelU, pixelV, distance):
    bottomPart = np.sqrt(1 + ((pixelU-cx)**2)/(fx**2) + ((pixelV-cy)**2)/(fy**2))
    z = distance / bottomPart 
    x = (pixelU - cx) * (z / fx)
    y = (pixelV - cy) * (z / fy) 
    return x, y, z

class PathPlanner(Node):
    def __init__(self):
        super().__init__('global_path_planner') 
        
        self.current_paired_cones = np.array([])
        self.current_midpoints = np.array([])
        self.car_x = 0.0 
        self.car_z = 0.0
        self.lock = threading.Lock() 

        self.cones_sub = self.create_subscription(Float32MultiArray,'/cones',self.conesCallback, 10)
        self.gps_sub = self.create_subscription(Float32MultiArray,'/carGPS', self.gpsCallback,10)
        self.waypoints_pub = self.create_publisher(Float32MultiArray, '/trajectory_waypoints', 10)
        
        plt.ion() 
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.configurar_plot()

    def configuring_plot(self): # Corrigido nome para evitar conflito
        self.ax.set_title('Planejamento Global')
        self.ax.set_xlabel('Posição Global X (m)')
        self.ax.set_ylabel('Posição Global Z (m)') 
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True)
    
    # ... (Seu método gpsCallback e conesCallback ficam aqui dentro da classe) ...
    # Vou resumir para focar na correção do MAIN
    
    def gpsCallback(self, msg):
        with self.lock:
            if len(msg.data) >= 3:
                self.car_x = msg.data[0]
                self.car_z = msg.data[2] 

    def conesCallback(self, msg):
        # ... (sua lógica completa aqui) ...
        pass # Placeholder para não ficar gigante, mantenha sua lógica

# --- AQUI ESTÁ A CORREÇÃO: Fora da classe ---

def main(args=None): # Adicione args=None para compatibilidade total
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