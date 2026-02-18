import sys
import os
import time
from pynput import keyboard

fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
sys.path.insert(0, fsds_lib_path)
import fsds

class KeyboardControlFSDS:
    def __init__(self):
        self.client = fsds.FSDSClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        self.throttle = 0.0
        self.brake = 0.0
        self.steering = 0.0
        
        self.keys_pressed = set()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        print("Controle Manual FSDS Iniciado.")
        print("W=Acelera, S=Freia, A=Esquerda, D=Direita, Espaço=Handbrake, R=Reset, Q=Sair")

    def on_press(self, key):
        try:
            k = key.char
            self.keys_pressed.add(k)
            if k == 'r': self.client.reset()
        except AttributeError:
            if key == keyboard.Key.space:
                self.keys_pressed.add('space')

    def on_release(self, key):
        try:
            k = key.char
            if k in self.keys_pressed:
                self.keys_pressed.remove(k)
            if k == 'q':
                return False
        except AttributeError:
            if key == keyboard.Key.space and 'space' in self.keys_pressed:
                self.keys_pressed.remove('space')

    def update_logic(self):
        self.throttle = 0.0
        self.brake = 0.0
        self.steering = 0.0
        handbrake = False

        if 'w' in self.keys_pressed:
            self.throttle = 0.5
        if 's' in self.keys_pressed:
            self.brake = 0.7
        if 'a' in self.keys_pressed:
            self.steering = -0.5
        if 'd' in self.keys_pressed:
            self.steering = 0.5
        if 'space' in self.keys_pressed:
            handbrake = True

        controls = fsds.CarControls()
        controls.throttle = float(self.throttle)
        controls.steering = float(self.steering)
        controls.brake = float(self.brake)
        controls.handbrake = handbrake
        
        self.client.setCarControls(controls)

    def run(self):
        try:
            while self.listener.running:
                self.update_logic()
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            self.client.setCarControls(fsds.CarControls())
            print("\nControle encerrado.")

if __name__ == '__main__':
    node = KeyboardControlFSDS()
    node.run()
