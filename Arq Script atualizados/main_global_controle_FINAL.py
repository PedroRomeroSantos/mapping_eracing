import time
import numpy as np
import fsds
from perception_FINAL import PerceptionSystem
from path_global_FINAL import GlobalPathPlanner
from controle_FINAL import PurePursuitController

def main():
    sensor = PerceptionSystem()
    planner = GlobalPathPlanner(sensor.client)
    controller = PurePursuitController()
    sensor.client.enableApiControl(True)

    try:
        while True:
            cones_locais = sensor.detect_cones()
            waypoints_globais = planner.processar_global(cones_locais)
            
            if len(waypoints_globais) > 0:
                waypoints_locais = []
                for wp in waypoints_globais:
                    waypoints_locais.append([wp[0] - planner.car_x, wp[1] - planner.car_z])
                
                throttle, steering, brake = controller.calculate_controls(np.array(waypoints_locais))
                
                controls = fsds.CarControls()
                controls.throttle = throttle
                controls.steering = steering
                controls.brake = brake
                sensor.client.setCarControls(controls)
            else:
                sensor.client.setCarControls(fsds.CarControls(brake=0.5))
            
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        sensor.client.setCarControls(fsds.CarControls(brake=1.0, throttle=0.0))
        print("Finalizado.")

if __name__ == "__main__":
    main()