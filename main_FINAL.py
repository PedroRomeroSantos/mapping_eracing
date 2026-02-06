import time
from perception_FINAL import PerceptionSystem
from path_FINAL import PathPlanner

def main():
    # 1. Instanciar Sistemas
    try:
        sensor = PerceptionSystem()
        planner = PathPlanner()
    except Exception as e:
        print("Erro ao inicializar sistemas:", e)
        return

    print("=== Sistema Rodando (Pressione Ctrl+C para parar) ===")

    try:
        while True:
            # A. PERCEPÇÃO: Pega dados do FSDS + YOLO + LiDAR
            # Retorna: [[Lateral, Profundidade, ID], ...]
            cones = sensor.detect_cones()
            
            # B. PLANEJAMENTO: Calcula pontos médios
            waypoints = planner.calcular_trajetoria(cones)
            
            # C. PRINT DE DEBUG
            if len(waypoints) > 0:
                # Ordena pelo Z mais próximo para mostrar o alvo imediato
                wps_ordenados = waypoints[waypoints[:, 1].argsort()]
                target = wps_ordenados[0]
                print(f"Target Imediato -> X (Lat): {target[0]:.2f} | Z (Prof): {target[1]:.2f} | Total WPs: {len(waypoints)}")
            else:
                print("Procurando cones...")

            # Pequeno sleep para não travar a máquina (opcional)
            # time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nEncerrando...")

if __name__ == "__main__":
    main()