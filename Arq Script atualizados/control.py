"""
Pure Pursuit Controller — FSDS Formula Student
================================================
Frame do veículo
    X  →  lateral  (esquerda negativo, direita positivo)
    Z  →  profundidade / frente (positivo para frente)
Fluxo:
    waypoints (PathPlanner) → seleciona lookahead → calcula curvatura
    → steering normalizado → throttle/brake por speed controller
    → client.setCarControls()
"""
import math
import numpy as np

WHEELBASE          = 1.53    # [m] distância entre eixos
MAX_STEER_ANGLE_RAD = math.radians(25)   # ângulo de esterço máximo das rodas
MAX_SPEED_MS        = 8.0    # [m/s] velocidade alvo máxima
MIN_SPEED_MS        = 2.5    # [m/s] velocidade mínima em curvas apertadas

#Lookahead dinâmico  →  L = k_ld * v  (clampado entre os limites)
K_LD               = 0.6    # ganho do lookahead (s), se o carro oscilar aumenta, não respondeu em curvas diminui
LOOKAHEAD_MIN      = 2.0    
LOOKAHEAD_MAX      = 8.0  

#Controlador de velocidade
K_THROTTLE         = 0.4    # ganho proporcional para aceleração
K_BRAKE            = 0.6    # ganho proporcional para frenagem
SPEED_DEADBAND     = 0.3    # [m/s] erro abaixo do qual não corrige

#Redução de velocidade em curvas
CURV_SPEED_GAIN    = 2.5    #quanto a curvatura reduz a velocidade alvo

class PurePursuitController:
    def __init__(self, client, vehicle_name: str = "FSCar"):
        self.client       = client
        self.vehicle      = vehicle_name
        self._prev_steer  = 0.0
    
    def step(self, waypoints: np.ndarray, speed_mps: float):
        import fsds

        controls        = fsds.CarControls()
        controls.handbrake = False

        if len(waypoints) == 0:
            #Sem waypoints = para suavemente
            controls.throttle = 0.0
            controls.brake    = 0.3
            controls.steering = 0.0
            self.client.setCarControls(controls, self.vehicle)
            return controls

        #Lookahead dinâmico
        lookahead = float(np.clip(K_LD * speed_mps, LOOKAHEAD_MIN, LOOKAHEAD_MAX))

        #Seleciona ponto de lookahead
        target = self._selecionar_lookahead(waypoints, lookahead)

        #Pure Pursuit = ângulo de esterço
        steer_angle_rad = self._calcular_steering(target)

        #Suavização do esterço
        alpha = 0.35          # quanto do novo valor aceitar por ciclo
        steer_angle_rad = (1 - alpha) * self._prev_steer + alpha * steer_angle_rad
        self._prev_steer = steer_angle_rad

        #Normaliza para [-1, 1]
        steer_norm = float(np.clip(steer_angle_rad / MAX_STEER_ANGLE_RAD, -1.0, 1.0))
      
        curvature = abs(math.tan(steer_angle_rad) / WHEELBASE)
        target_speed  = MAX_SPEED_MS / (1.0 + CURV_SPEED_GAIN * curvature)
        target_speed  = float(np.clip(target_speed, MIN_SPEED_MS, MAX_SPEED_MS))

        #Controlador de velocidade
        throttle, brake = self._speed_controller(speed_mps, target_speed)

        controls.steering = steer_norm
        controls.throttle = throttle
        controls.brake = brake

        self.client.setCarControls(controls, self.vehicle)
        return controls
      
    def _selecionar_lookahead(self, waypoints: np.ndarray, lookahead: float) -> np.ndarray:
        """
        Retorna o waypoint mais próximo que esteja além da distância de lookahead.
        Se nenhum ultrapassar, usa o mais distante disponível
        """
        #distâncias euclidianas a partir da origem (posição do veículo)
        dists = np.linalg.norm(waypoints, axis=1)

        #candidatos além do lookahead
        mask = dists >= lookahead
        if mask.any():
            candidates = waypoints[mask]
            dists_cand = dists[mask]
            return candidates[np.argmin(dists_cand)]     #o mais próximo acima do lookahead

        #fallback: waypoint mais à frente
        return waypoints[np.argmax(waypoints[:, 1])]

    def _calcular_steering(self, target: np.ndarray) -> float:
        """
            alpha  = atan2(X_lateral, Z_frente)
            kappa  = 2 * sin(alpha) / L
            delta  = atan(kappa * wheelbase)
        """
        x, z = float(target[0]), float(target[1])
        if z <= 0.0:
            z = 0.01          #evita divisão por zero / inversão de sinal

        L = math.sqrt(x**2 + z**2)
        alpha = math.atan2(x, z)                    #ângulo lateral do alvo
        kappa = 2.0 * math.sin(alpha) / L           #curvatura de Ackermann
        delta = math.atan(kappa * WHEELBASE)        #ângulo de esterço das rodas

        return float(np.clip(delta, -MAX_STEER_ANGLE_RAD, MAX_STEER_ANGLE_RAD))

    @staticmethod
    def _speed_controller(current: float, target: float):
        """Controlador P simples de velocidade → (throttle, brake)."""
        error = target - current

        if abs(error) < SPEED_DEADBAND:
            return 0.15, 0.0 #manutenção leve

        if error > 0: #precisa acelerar
            throttle = float(np.clip(K_THROTTLE * error, 0.0, 1.0))
            brake    = 0.0
        else: #precisa frear
            throttle = 0.0
            brake    = float(np.clip(K_BRAKE * abs(error), 0.0, 1.0))

        return throttle, brake
