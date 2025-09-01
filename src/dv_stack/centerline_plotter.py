class CenterlinePlotter:
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt

    def plot_centerline(self, points, triangles):
        for triangle in triangles:
            self.plt.fill(*zip(*triangle), alpha=0.3)
        self.plt.plot(*zip(*points), marker='o', color='red', linestyle='None')
        self.plt.title('Centerline Plot')
        self.plt.xlabel('X-axis')
        self.plt.ylabel('Y-axis')
        self.plt.axis('equal')
        self.plt.grid()

    def save_plot(self, filename):
        self.plt.savefig(filename)
        self.plt.close()