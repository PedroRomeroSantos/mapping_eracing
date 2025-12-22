import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

class PIDController(Node):
    def __init__(self):
        super().__init__('pid_controller_node')

        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/trajectory_waypoints',
            self.waypoint_callback,
            10
        )

        self.bridge_pub = self.create_publisher(
            Float32MultiArray, 
            '/cmd_bridge', 
            10
        )

        # --- PARÂMETROS PARA MODO LENTO E SUAVE ---
        self.kp = 0.6           # Muito mais baixo para evitar viradas bruscas
        self.kd = 0.8           # Mais alto para amortecer o movimento do volante
        self.ki = 0.0           # Desligado para simplificar o teste inicial
        
        # Velocidade mínima para movimento (ajuste conforme o torque do carro)
        self.throttle_const = 0.26 
        self.steer_limit = 0.6  # Limita o volante a 60% para não travar as rodas

        self.target_steering = 0.0
        self.target_throttle = 0.0
        self.prev_error = 0.0
        self.last_waypoint_time = self.get_clock().now()

        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info('Modo de Segurança: Baixa Velocidade e Direção Suave.')

    def waypoint_callback(self, msg):
        if len(msg.data) < 2:
            return

        # Waypoint local (X, Z)
        tx = msg.data[0]
        tz = msg.data[1]
        
        # Erro angular - Certifique-se que tx positivo = Direita
        error = -np.arctan2(tx, tz) 
        
        now = self.get_clock().now()
        dt = (now - self.last_waypoint_time).nanoseconds / 1e9
        self.last_waypoint_time = now

        if dt > 0:
            derivative = (error - self.prev_error) / dt
            
            # Cálculo PID
            raw_steer = (self.kp * error) + (self.kd * derivative)
            self.target_steering = float(np.clip(raw_steer, -self.steer_limit, self.steer_limit))
            
            # Mantemos a velocidade no mínimo necessário
            self.target_throttle = self.throttle_const
            
            # --- LOG DE DIAGNÓSTICO ---
            # Se tx > 0 (Direita), Steer deve ser Positivo.
            self.get_logger().info(
                f"WP_X: {tx:5.2f} | Erro: {error:5.2f} | Steer: {self.target_steering:5.2f}"
            )
            
            self.prev_error = error

    def timer_callback(self):
        now = self.get_clock().now()
        time_since_last = (now - self.last_waypoint_time).nanoseconds / 1e9
        
        msg = Float32MultiArray()
        
        # Watchdog: Se não vir cones, freia (1.0 no brake)
        if time_since_last > 0.8:
            msg.data = [0.0, 0.0, 1.0] 
        else:
            msg.data = [float(self.target_steering), float(self.target_throttle), 0.0]
            
        self.bridge_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PIDController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()