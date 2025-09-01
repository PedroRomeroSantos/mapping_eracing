class DelaunayTriangulator:
    def __init__(self, points):
        self.points = points
        self.triangles = []

    def compute_triangles(self):
        from scipy.spatial import Delaunay
        delaunay = Delaunay(self.points)
        self.triangles = delaunay.simplices

    def get_edges(self):
        edges = set()
        for triangle in self.triangles:
            for i in range(3):
                edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3])))
                edges.add(edge)
        return list(edges)