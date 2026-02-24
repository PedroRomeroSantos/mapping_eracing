import numpy as np
import matplotlib
matplotlib.use('TkAgg')  #pro mapa aparecer mas não flodar/bloquear outros comandos
import matplotlib.pyplot as plt
import threading
import queue

class PathPlanner:
    HORIZONTE_MAXIMO = 30.0
    LARGURA_MAXIMA   = 5.0
    LARGURA_MINIMA   = 2.0
    MAX_DIFF_Z       = 2.0

    def __init__(self, visualizar: bool = True):
        self.visualizar = visualizar
        self._fila      = queue.Queue(maxsize=1)  #só guarda o frame mais recente

        if self.visualizar:
            t = threading.Thread(target=self._loop_plot, daemon=True)
            t.start()

    def calcular_trajetoria(self, cones: np.ndarray) -> np.ndarray:
        if len(cones) == 0:
            if self.visualizar:
                self._enfileirar(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2)))
            return np.empty((0, 2))

        esq  = cones[(cones[:, 2] == 0) & (cones[:, 1] < self.HORIZONTE_MAXIMO)]
        dire = cones[(cones[:, 2] == 1) & (cones[:, 1] < self.HORIZONTE_MAXIMO)]

        waypoints = self._parear_cones(esq, dire)
        wps = np.array(waypoints) if waypoints else np.empty((0, 2))

        if self.visualizar:
            self._enfileirar(esq, dire, wps)

        return wps

    def _parear_cones(self, esq, dire):
        if len(esq) == 0 or len(dire) == 0:
            return []

        esq  = esq [esq [:, 1].argsort()]
        dire = dire[dire[:, 1].argsort()]

        waypoints      = []
        indices_usados = set()

        for c_esq in esq:
            best_idx  = -1
            min_z_err = self.MAX_DIFF_Z

            for i, c_dire in enumerate(dire):
                if i in indices_usados:
                    continue
                dz = abs(c_esq[1] - c_dire[1])
                dx = abs(c_esq[0] - c_dire[0])
                if dz < min_z_err and self.LARGURA_MINIMA < dx < self.LARGURA_MAXIMA:
                    min_z_err = dz
                    best_idx  = i

            if best_idx != -1:
                waypoints.append([(c_esq[0] + dire[best_idx, 0]) / 2,
                                  (c_esq[1] + dire[best_idx, 1]) / 2])
                indices_usados.add(best_idx)

        return waypoints

    def _enfileirar(self, esq, dire, wps):
        try:
            self._fila.get_nowait()
        except queue.Empty:
            pass
        self._fila.put((esq.copy(), dire.copy(), wps.copy()))

    def _loop_plot(self):
        #roda em thread separada
        plt.ion()
        fig, ax = plt.subplots(figsize=(5, 7))
        fig.canvas.manager.set_window_title("Mapeamento local")

        while True:
            esq, dire, wps = self._fila.get()

            ax.clear()
            ax.set_title("Mapeamento local")
            ax.set_xlim([-10, 10])
            ax.set_ylim([0, 35])
            ax.set_xlabel("X lateral")
            ax.set_ylabel("Z profundidade")
            ax.grid(True)
            ax.plot(0, 0, "g^", markersize=12, label="Veículo")

            if len(esq) > 0:
                ax.scatter(esq[:, 0], esq[:, 1], c="blue", s=40, label="Cone esq")
            if len(dire) > 0:
                ax.scatter(dire[:, 0], dire[:, 1], c="gold", s=40, label="Cone dir")
            if len(wps) > 0:
                wps_ord = wps[wps[:, 1].argsort()]
                ax.plot(wps_ord[:, 0], wps_ord[:, 1], "r--", alpha=0.6)
                ax.scatter(wps_ord[:, 0], wps_ord[:, 1],
                           c="red", marker="x", s=60, label="Waypoint")

            ax.legend(loc="upper right", fontsize=8)
            fig.canvas.draw()
            fig.canvas.flush_events()
