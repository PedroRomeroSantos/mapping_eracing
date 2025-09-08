import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt

class ConeMapper(Node):
    def __init__(self):
        super().__init__('cone_mapper')

        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/cones_depth',
            self.listener_callback,
            10
        )

        self.cones = []  # Lista de cones (x, z, lado)

        # Configuração do matplotlib
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Mapa de Cones em Tempo Real")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Z (m)")

    def listener_callback(self, msg):
        data = msg.data
        cones_detectados = []
        for i in range(0, len(data), 3):
            x = data[i]
            z = data[i+1]
            lado = data[i+2]
            cones_detectados.append((x, z, lado))

        # Atualiza lista com os novos cones
        self.cones.extend(cones_detectados)
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        self.ax.set_title("Mapa de Cones em Tempo Real")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Z (m)")
        self.ax.grid(True)

        # Separar cones por lado e ordenar por distância (z)
        cones_esq = sorted([(x, z) for (x, z, lado) in self.cones if lado < 0], key=lambda c: c[1])
        cones_dir = sorted([(x, z) for (x, z, lado) in self.cones if lado > 0], key=lambda c: c[1])

        # Plotar cones
        if cones_esq:
            xs_esq, zs_esq = zip(*cones_esq)
            self.ax.scatter(xs_esq, zs_esq, c='blue', label='Esquerda (-1)')
        if cones_dir:
            xs_dir, zs_dir = zip(*cones_dir)
            self.ax.scatter(xs_dir, zs_dir, c='yellow', label='Direita (+1)')

        # Desenhar linha central conectando pares
        mids_x, mids_z = [], []
        for c_esq, c_dir in zip(cones_esq, cones_dir):
            mid_x = (c_esq[0] + c_dir[0]) / 2
            mid_z = (c_esq[1] + c_dir[1]) / 2
            mids_x.append(mid_x)
            mids_z.append(mid_z)

        if mids_x:
            self.ax.plot(mids_x, mids_z, 'g--', label='Linha Central')

        self.ax.legend()
        plt.draw()
        plt.pause(0.01)

def main(args=None):
    rclpy.init(args=args)
    node = ConeMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()
