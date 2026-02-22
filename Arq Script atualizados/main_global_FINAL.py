import time
from perception_FINAL import PerceptionSystem
from path_global_FINAL import GlobalPathPlanner

def main():
    sensor = PerceptionSystem()
    planner = GlobalPathPlanner(sensor.client)

    print("Mapeamento global ON")

    try:
        while True:
            cones = sensor.detect_cones()
            waypoints_globais = planner.processar_global(cones)

            if len(waypoints_globais) > 0:

                print(f"GPS -> X: {planner.car_x:.2f} | Z: {planner.car_z:.2f} | WPs: {len(waypoints_globais)}")
            else:
                print("esperando pares")

            time.sleep(0.02) 

    except KeyboardInterrupt:
        print("\nOFF")

if __name__ == "__main__":
    main()