import numpy as np

class PurePursuitController:
    def __init__(self):
        self.KP_STEER = 1
        self.LOOK_AHEAD = 10
        self.MAX_THR = 0.20
        self.MIN_THR = 0.15

    def calculate_controls(self, wps_locais):
        if len(wps_locais) == 0: return 0.0, 0.0, 1
        dists = np.sqrt(wps_locais[:, 0]**2 + wps_locais[:, 1]**2)
        idx = np.argmin(np.abs(dists - self.LOOK_AHEAD))
        target = wps_locais[idx]
        tx, tz = target[0], target[1]
        dist_sq = tx**2 + tz**2
        steer = (2 * tx) / dist_sq if dist_sq > 0 else 0.0
        steer_cmd = np.clip(steer * self.KP_STEER, -1.0, 1.0)
        throttle = self.MAX_THR * (1.0 - abs(steer_cmd) * 1.25)
        throttle = max(throttle, self.MIN_THR)
        return float(throttle), float(steer_cmd), 0.0