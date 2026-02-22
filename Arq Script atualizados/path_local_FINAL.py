import numpy as np
import matplotlib.pyplot as plt

class PathPlanner:
    def __init__(self):
        self.LADO_ESQUERDO = -1.0
        self.LADO_DIREITO = 1.0   
        self.ID_CORES = {0: self.LADO_ESQUERDO, 1: self.LADO_DIREITO}
        
        self.HORIZONTE_MAXIMO = 30.0
        self.LARGURA_MAXIMA = 5.0
        self.LARGURA_MINIMA = 2.0
        self.MAX_DIFF_Z = 2.0

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(5, 7))

    def configurar_plot(self):
        self.ax.set_title('Mapeamento local')
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([0, 35])
        self.ax.grid(True)
        self.ax.plot(0, 0, 'g^', markersize=12)

    def calcular_trajetoria(self, cones):
        if len(cones) == 0: 
            self.renderizar(np.array([]), np.array([]), np.array([]))
            return np.array([])

        esq = cones[(cones[:, 2] == 0) & (cones[:, 1] < self.HORIZONTE_MAXIMO)]
        dire = cones[(cones[:, 2] == 1) & (cones[:, 1] < self.HORIZONTE_MAXIMO)]

        waypoints = []
        if len(esq) > 0 and len(dire) > 0:
            esq = esq[esq[:, 1].argsort()]
            dire = dire[dire[:, 1].argsort()]
            
            indices_usados = set()
            for c_esq in esq:
                best_idx = -1
                min_z_err = self.MAX_DIFF_Z
                
                for i, c_dire in enumerate(dire):
                    if i in indices_usados: continue
                    
                    dz = abs(c_esq[1] - c_dire[1])
                    dx = abs(c_esq[0] - c_dire[0])
                    
                    if dz < min_z_err and self.LARGURA_MINIMA < dx < self.LARGURA_MAXIMA:
                        min_z_err = dz
                        best_idx = i
                
                if best_idx != -1:
                    waypoints.append([(c_esq[0] + dire[best_idx, 0])/2, (c_esq[1] + dire[best_idx, 1])/2])
                    indices_usados.add(best_idx)

        wps = np.array(waypoints)
        self.renderizar(esq, dire, wps)
        return wps

    def renderizar(self, esq, dire, wps):
        self.ax.clear()
        self.configurar_plot()
        if len(esq) > 0: self.ax.scatter(esq[:, 0], esq[:, 1], c='blue', s=30)
        if len(dire) > 0: self.ax.scatter(dire[:, 0], dire[:, 1], c='yellow', s=30)
        if len(wps) > 0:
            wps_plot = wps[wps[:, 1].argsort()]
            self.ax.plot(wps_plot[:, 0], wps_plot[:, 1], 'r--', alpha=0.6)
            self.ax.scatter(wps_plot[:, 0], wps_plot[:, 1], c='red', marker='x')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)