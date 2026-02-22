import numpy as np

class PurePursuitController:
    def __init__(self):
        print("Inicializando Controlador Pure Pursuit...")
        
        self.K_P_STEER = 0.45   #ganho proporcional KP
        self.LOOK_AHEAD_DIST = 6.0 #distância que considera na frente
        self.MAX_STEER = 1.0    #limite de esterço do simulador (-1 a 1)
        
        #vel alvo
        self.TARGET_THROTTLE = 0.25
        self.MIN_THROTTLE = 0.12

    def calculate_controls(self, waypoints):
        #Recebe waypoints: [[X_lateral, Z_frente], ...]
        #Retorna: (throttle, steering, brake)
        if len(waypoints) == 0:
            return 0.0, 0.0, 0.5 #para o carro se não tem caminho
          
        #waypoint que está mais perto da distância alvo
        target_pt = self._get_lookahead_point(waypoints)
        
        tx, tz = target_pt[0], target_pt[1]

        #esterço, erro angular
        dist_sq = tx**2 + tz**2
        if dist_sq > 0:
            steering = (2 * tx) / dist_sq
        else:
            steering = 0.0
          
        steering_cmd = np.clip(steering * self.K_P_STEER, -self.MAX_STEER, self.MAX_STEER)

        #acelerador pra frente
        steer_factor = abs(steering_cmd)
        throttle_cmd = self.TARGET_THROTTLE * (1.0 - 0.5 * steer_factor)
        throttle_cmd = max(throttle_cmd, self.MIN_THROTTLE)

        return float(throttle_cmd), float(steering_cmd), 0.0

    def _get_lookahead_point(self, waypoints):
        #waypoint mais próximo da distância LOOK_AHEAD_DIST
        pts = waypoints[waypoints[:, 1].argsort()]
        
        for pt in pts:
            dist = np.sqrt(pt[0]**2 + pt[1]**2)
            if dist >= self.LOOK_AHEAD_DIST:
                return pt
        
        return pts[-1]
