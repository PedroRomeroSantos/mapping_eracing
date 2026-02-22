import time
from perception_FINAL import PerceptionSystem
from path_local_FINAL import PathPlanner

def main():
    sensor = PerceptionSystem()
    planner = PathPlanner()

    print("Mapeamento reativo sem controle ON")

    try:

        while True:

            cones = sensor.detect_cones()
            waypoints = planner.calcular_trajetoria(cones)

            if len(waypoints) > 0:

                target = waypoints[waypoints[:, 1].argsort()][0]
                print(f"Target -> X: {target[0]:.2f}m | Z: {target[1]:.2f}m | Cones: {len(cones)}")

            else:
                print("esperando pares de cones")

            time.sleep(0.02) 

    except KeyboardInterrupt:

        print("\n=OFF")


if __name__ == "__main__":

    main()