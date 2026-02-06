import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURAÇÕES ---
LADO_ESQUERDO = -1.0 # Verifique se o ID 0 do seu YOLO é Esquerda ou Direita
LADO_DIREITO = 1.0   # Verifique se o ID 1 do seu YOLO é Esquerda ou Direita

# IDs do YOLO: Mapeie o ID que sai do modelo para o Lado correto
# Exemplo: Se o YOLO diz 0=Azul e 1=Amarelo.
ID_CORES = {
    0: LADO_ESQUERDO, 
    1: LADO_DIREITO,
    2: 0.0,            
}

HORIZONTE_MAXIMO = 30.0   
LARGURA_MAXIMA = 12.0    

class PathPlanner:
    def __init__(self):
        print("Inicializando Path Planner (Standalone)...")
        
        # GRÁFICO
        self.visualizacao_ativa = True
        if self.visualizacao_ativa:
            plt.ion() 
            self.fig, self.ax = plt.subplots(figsize=(6, 8))
            self.configurar_plot()

    def configurar_plot(self):
        self.ax.set_title('Visão Local (Carro em 0,0)')
        self.ax.set_xlabel('Lateral X (m)')
        self.ax.set_ylabel('Frente Z (m)')
        self.ax.set_xlim([-10, 10]) 
        self.ax.set_ylim([0, 35])
        self.ax.grid(True)
        # Carro verde
        self.ax.plot(0, 0, 'g^', markersize=15, label='Carro')

    def atualizar_grafico(self, esq, dir, wps):
        if not self.visualizacao_ativa: return

        self.ax.clear()
        self.configurar_plot()

        if len(esq) > 0:
            self.ax.scatter(esq[:, 0], esq[:, 1], c='blue', label='Esq')
        if len(dir) > 0:
            self.ax.scatter(dir[:, 0], dir[:, 1], c='orange', label='Dir')

        if len(wps) > 0:
            self.ax.plot(wps[:, 0], wps[:, 1], 'r--', alpha=0.5) 
            self.ax.scatter(wps[:, 0], wps[:, 1], c='red', marker='x', s=80)
            self.ax.scatter(wps[0, 0], wps[0, 1], c='darkred', s=150, label='Target')

        self.ax.legend(loc='upper right')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def calcular_trajetoria(self, dados_cones):
        """
        Recebe lista de cones: [[lateral, depth, id], ...]
        """
        if len(dados_cones) == 0:
            return []

        cones_esquerda = []
        cones_direita = []

        # Separar e Filtrar
        for cone in dados_cones:
            x_local, z_local, id_cor = cone
            
            # Filtro de Horizonte (profundidade)
            if z_local > HORIZONTE_MAXIMO or z_local < 0.5: continue

            lado = ID_CORES.get(int(id_cor), 0.0)
            if lado == 0.0: continue

            # NOTA: O perception já entrega Metros, não precisa converter pixel!
            if lado == LADO_ESQUERDO:
                cones_esquerda.append([x_local, z_local])
            elif lado == LADO_DIREITO:
                cones_direita.append([x_local, z_local])

        cones_esquerda = np.array(cones_esquerda)
        cones_direita = np.array(cones_direita)

        # Calcular Waypoints
        waypoints = np.array([])
        if len(cones_esquerda) > 0 and len(cones_direita) > 0:
            waypoints = self.calcular_ponto_medio(cones_esquerda, cones_direita)

        # Atualizar Visualização
        self.atualizar_grafico(cones_esquerda, cones_direita, waypoints)
        
        return waypoints

    def calcular_ponto_medio(self, esquerda, direita):
        pontos_medios = []
        indices_usados = set()
        
        esquerda = esquerda[np.argsort(esquerda[:, 1])]
        direita = direita[np.argsort(direita[:, 1])]

        for cone_esq in esquerda:
            x_esq, z_esq = cone_esq[0], cone_esq[1]
            melhor_idx = -1
            menor_diff = float('inf')

            for i, cone_dir in enumerate(direita):
                if i in indices_usados: continue

                z_dir = cone_dir[1]
                diff_z = abs(z_esq - z_dir)

                if diff_z < 5.0 and diff_z < menor_diff:
                    menor_diff = diff_z
                    melhor_idx = i
            
            if melhor_idx != -1:
                cone_dir = direita[melhor_idx]
                x_dir, z_dir = cone_dir[0], cone_dir[1]
                
                largura = abs(x_esq - x_dir)
                if largura < LARGURA_MAXIMA:
                    meio_x = (x_esq + x_dir) / 2.0
                    meio_z = (z_esq + z_dir) / 2.0
                    pontos_medios.append([meio_x, meio_z])
                    indices_usados.add(melhor_idx)

        return np.array(pontos_medios)