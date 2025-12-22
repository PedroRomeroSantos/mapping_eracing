#!/usr/bin/env python3
import rospy
from fs_msgs.msg import ControlCommand
from pynput import keyboard

class KeyboardControlROS1:
    def __init__(self):
        self.pub = rospy.Publisher('/fsds/control_command', ControlCommand, queue_size=10)
        self.rate = rospy.Rate(20)  # 20 Hz

        self.throttle = 0.0
        self.brake = 0.0
        self.steering = 0.0
        
        self.keys_pressed = set()

        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        rospy.loginfo("Keyboard control (ROS1) iniciado. W=acelera, S=freia, A=esquerda, D=direita, Espaço=freio total, Q=sair")

    def on_press(self, key):
        try:
            self.keys_pressed.add(key.char)
        except AttributeError:
            if key == keyboard.Key.space:
                self.keys_pressed.add('space')

    def on_release(self, key):
        try:
            if key.char in self.keys_pressed:
                self.keys_pressed.remove(key.char)
        except AttributeError:
            if key == keyboard.Key.space and 'space' in self.keys_pressed:
                self.keys_pressed.remove('space')

        if key == keyboard.Key.esc or (hasattr(key, 'char') and key.char == 'q'):
            rospy.signal_shutdown("Encerrado pelo usuário")

    def update_controls(self):
        self.throttle = 0.0
        self.brake = 0.0
        self.steering = 0.0

        if 'w' in self.keys_pressed:
            self.throttle = 0.4   # intensidade fixa (pode ajustar)
        if 's' in self.keys_pressed:
            self.brake = 0.6
        if 'a' in self.keys_pressed:
            self.steering = -0.4
        if 'd' in self.keys_pressed:
            self.steering = 0.4
        if 'space' in self.keys_pressed:
            self.brake = 1.0
            self.throttle = 0.0

    def run(self):
        while not rospy.is_shutdown():
            self.update_controls()

            msg = ControlCommand()
            msg.throttle = float(self.throttle)
            msg.brake = float(self.brake)
            msg.steering = float(self.steering)
            self.pub.publish(msg)

            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('keyboard_control_ros1')
    node = KeyboardControlROS1()
    node.run()
