import time
import fsds
from perception_FINAL import PerceptionSystem
from path_local_FINAL import PathPlanner
from controle_FINAL import PurePursuitController

def main():
    sensor = PerceptionSystem()
    planner = PathPlanner()
    controller = PurePursuitController()
        
    sensor.client.enableApiControl(True)

    print("perception, path e controle ON")

    try:
        while True:
            cones = sensor.detect_cones()
            waypoints = planner.calcular_trajetoria(cones)
            
            if len(waypoints) > 0:
                throttle, steering, brake = controller.calculate_controls(waypoints)
                
                car_controls = fsds.CarControls()
                car_controls.throttle = throttle
                car_controls.steering = steering
                car_controls.brake = brake
                
                sensor.client.setCarControls(car_controls)
                
                print(f"Controle = STR: {steering:5.2f} | THR: {throttle:5.2f} | WPs: {len(waypoints)}")
            else:
                #se não vir cones freia um pouco
                sensor.client.setCarControls(fsds.CarControls(brake=0.5))
                print("Aguardando detecção de pares de cones...")

            time.sleep(0.02)

    except KeyboardInterrupt:
        sensor.client.setCarControls(fsds.CarControls(brake=1.0, throttle=0.0))
        print("\nOFF")

if __name__ == "__main__":
    main()
