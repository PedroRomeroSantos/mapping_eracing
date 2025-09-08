import rclpy
from rclpy.node import Node
from fsds_msgs.msg import ControlCommand  


from pynput import keyboard

class KeyboardControl(Node):
    def __init__(self):
        super().__init__('keyboard_control')

        self.publisher = self.create_publisher(ControlCommand, '/fsds/control_command', 10)
        self.timer = self.create_timer(0.05, self.publish_command)

        # Estado atual
        self.throttle = 0.0
        self.brake = 0.0
        self.steering = 0.0

        # Listener de teclado
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        self.get_logger().info("Keyboard control started. Use W/A/S/D to drive, SPACE to brake.")

    def publish_command(self):
        msg = ControlCommand()
        msg.throttle = float(self.throttle)
        msg.brake = float(self.brake)
        msg.steering = float(self.steering)
        self.publisher.publish(msg)

    def on_press(self, key):
        try:
            if key.char == 'w':
                self.throttle = 0.5
            elif key.char == 's':
                self.brake = 0.5
            elif key.char == 'a':
                self.steering = -0.5
            elif key.char == 'd':
                self.steering = 0.5
        except AttributeError:
            if key == keyboard.Key.space:
                self.brake = 1.0

    def on_release(self, key):
        try:
            if key.char in ['w', 's']:
                self.throttle = 0.0
                self.brake = 0.0
            elif key.char in ['a', 'd']:
                self.steering = 0.0
        except AttributeError:
            if key == keyboard.Key.space:
                self.brake = 0.0

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
