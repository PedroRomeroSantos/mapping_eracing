import sys
import os
import time
from pynput import keyboard

# --- SETUP DO CAMINHO FSDS ---
fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
sys.path.insert(0, fsds_lib_path)
import fsds

class KeyboardControlFSDS:
    def __init__(self):
        # Conexão direta com o cliente FSDS
        self.client = fsds.FSDSClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True) # Habilita controle via API

        # Estados dos comandos
        self.throttle = 0.0
        self.brake = 0.0
        self.steering = 0.0
        
        self.keys_pressed = set()

        # Listener do teclado
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        print("Controle Manual FSDS Iniciado.")
        print("W=Acelera, S=Freia, A=Esquerda, D=Direita, Espaço=Handbrake, R=Reset, Q=Sair")

    def on_press(self, key):
        try:
            k = key.char
            self.keys_pressed.add(k)
            if k == 'r': self.client.reset() # Reset rápido da pista
        except AttributeError:
            if key == keyboard.Key.space:
                self.keys_pressed.add('space')

    def on_release(self, key):
        try:
            k = key.char
            if k in self.keys_pressed:
                self.keys_pressed.remove(k)
            if k == 'q':
                return False # Para o listener e encerra
        except AttributeError:
            if key == keyboard.Key.space and 'space' in self.keys_pressed:
                self.keys_pressed.remove('space')

    def update_logic(self):
        # Reset dos valores a cada ciclo
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

        # Cria o objeto de controle nativo do FSDS
        controls = fsds.CarControls()
        controls.throttle = float(self.throttle)
        controls.steering = float(self.steering)
        controls.brake = float(self.brake)
        controls.handbrake = handbrake
        
        # Envia para o simulador
        self.client.setCarControls(controls)

    def run(self):
        try:
            while self.listener.running:
                self.update_logic()
                time.sleep(0.05) # 20Hz como no ROS
        except KeyboardInterrupt:
            pass
        finally:
            # Para o carro antes de sair
            self.client.setCarControls(fsds.CarControls())
            print("\nControle encerrado.")

if __name__ == '__main__':
    node = KeyboardControlFSDS()
    node.run()