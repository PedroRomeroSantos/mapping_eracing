import time
from perception_FINAL import PerceptionSystem
from path_FINAL import PathPlanner

def main():
    try:
        sensor = PerceptionSystem()
        planner = PathPlanner()
    except Exception as e:
        print("Erro ao inicializar sistemas:", e)
        return

    print("iniciou")

    try:
        while True:
            # [[Lateral, Profundidade, ID]
            cones = sensor.detect_cones()
            
            # calcula pontos médios
            waypoints = planner.calcular_trajetoria(cones)
            
            if len(waypoints) > 0:
                # Z mais próximo para mostrar o alvo imediato
                wps_ordenados = waypoints[waypoints[:, 1].argsort()]
                target = wps_ordenados[0]
                print(f"Target Imediato -> X (Lat): {target[0]:.2f} | Z (Prof): {target[1]:.2f} | Total WPs: {len(waypoints)}")
            else:
                print("Procurando cones...")

    except KeyboardInterrupt:
        print("\nEncerrando...")

if __name__ == "__main__":
    main()
