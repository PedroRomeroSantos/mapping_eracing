import numpy as np
import matplotlib.pyplot as plt

class GlobalPathPlanner:
    def __init__(self, fsds_client):
        self.client = fsds_client
        self.LADO_ESQUERDO = 0.0
        self.LADO_DIREITO = 1.0
        self.HORIZONTE_PLANEJAMENTO = 40.0
        self.LARGURA_MAX_PISTA = 12.0
        self.DELTA_Z_MAX = 5.0
        
        self.car_x = 0.0
        self.car_z = 0.0
        
        self.current_midpoints = np.array([])
        self.current_paired_cones = np.array([])

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

    def atualizar_posicao_carro(self):
        state = self.client.getCarState()
        self.car_z = state.kinematics_estimated.position.x_val
        self.car_x = state.kinematics_estimated.position.y_val

    def processar_global(self, cones_locais):
        if len(cones_locais) == 0:
            return np.array([])

        self.atualizar_posicao_carro()

        working_data_global = []
        for cone in cones_locais:
            x_local, z_local, cid = cone
            
            if 0.0 < z_local < self.HORIZONTE_PLANEJAMENTO:
                X_map = self.car_x + x_local
                Z_map = self.car_z + z_local
                working_data_global.append([X_map, Z_map, cid])
        
        working_data = np.array(working_data_global)
        if len(working_data) == 0: return np.array([])

        esq = working_data[working_data[:, 2] == self.LADO_ESQUERDO]
        dire = working_data[working_data[:, 2] == self.LADO_DIREITO]

        waypoints, paired_cones = self._pair_and_calculate(esq, dire)

        self.current_midpoints = waypoints
        self.current_paired_cones = paired_cones
        self.renderizar()

        return waypoints

    def _pair_and_calculate(self, cones_left, cones_right):
        if len(cones_left) == 0 or len(cones_right) == 0:
            return np.array([]), np.array([])

        cones_left = cones_left[np.argsort(cones_left[:, 1])]
        midpoints = []
        all_paired_cones = []
        indices_dire_usados = set()

        for c_L in cones_left:
            x_L, z_L = c_L[0], c_L[1]
            
            melhor_idx = -1
            menor_dz = self.DELTA_Z_MAX

            for i, c_R in enumerate(cones_right):
                if i in indices_dire_usados: continue
                
                x_R, z_R = c_R[0], c_R[1]
                dz = abs(z_L - z_R)
                dx = abs(x_L - x_R)

                if dz < menor_dz and dx < self.LARGURA_MAX_PISTA:
                    menor_dz = dz
                    melhor_idx = i
            
            if melhor_idx != -1:
                c_R_par = cones_right[melhor_idx]
                midpoints.append([(x_L + c_R_par[0])/2.0, (z_L + c_R_par[1])/2.0])
                all_paired_cones.extend([c_L, c_R_par])
                indices_dire_usados.add(melhor_idx)

        return np.array(midpoints), np.array(all_paired_cones)

    def renderizar(self):
        self.ax.clear()
        self.ax.set_title('Mapeamento Global')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Z')
        
        self.ax.set_xlim([self.car_x - 20, self.car_x + 20])
        self.ax.set_ylim([self.car_z - 10, self.car_z + 40])
        self.ax.grid(True)
        
        self.ax.scatter(self.car_x, self.car_z, c='green', marker='^', s=150, label='Carro')

        if len(self.current_paired_cones) > 0:
            data = np.array(self.current_paired_cones)
            esq = data[data[:, 2] == self.LADO_ESQUERDO]
            dire = data[data[:, 2] == self.LADO_DIREITO]
            if len(esq) > 0: self.ax.scatter(esq[:, 0], esq[:, 1], c='blue', s=50)
            if len(dire) > 0: self.ax.scatter(dire[:, 0], dire[:, 1], c='yellow', s=50)

        if len(self.current_midpoints) > 0:
            wps = self.current_midpoints[self.current_midpoints[:, 1].argsort()]
            self.ax.plot(wps[:, 0], wps[:, 1], 'r-x', markersize=8, label='Waypoints')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)