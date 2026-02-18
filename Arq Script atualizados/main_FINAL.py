import time
from perception_FINAL import PerceptionSystem
from path_FINAL import PathPlanner

def main():
    try:
        sensor = PerceptionSystem()
        planner = PathPlanner()
    except Exception as e:
        print(f"Falha na inicialização: {e}")
        return

    print("=== Sistema de Mapeamento Reativo Iniciado ===")

    try:
        while True:
            cones = sensor.detect_cones()
            waypoints = planner.calcular_trajetoria(cones)
            
            if len(waypoints) > 0:
                # O waypoint[0] = Target imediato
                target = waypoints[waypoints[:, 1].argsort()][0]
                print(f"Target -> X: {target[0]:.2f}m | Z: {target[1]:.2f}m | Cones: {len(cones)}")
            else:
                print("Aguardando detecção de pares de cones...")
            time.sleep(0.02) 

    except KeyboardInterrupt:
        print("\nDesligando sistemas...")

if __name__ == "__main__":
    main()
