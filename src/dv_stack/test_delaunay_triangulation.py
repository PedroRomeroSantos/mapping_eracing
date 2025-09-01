import unittest
import numpy as np
from src.delaunay_triangulation import DelaunayTriangulator

class TestDelaunayTriangulation(unittest.TestCase):

    def setUp(self):
        self.triangulator = DelaunayTriangulator()

    def test_simple_triangle(self):
        points = np.array([[0, 0], [1, 0], [0, 1]])
        triangles = self.triangulator.compute_triangles(points)
        expected_triangles = [[0, 1, 2]]
        self.assertEqual(triangles.tolist(), expected_triangles)

    def test_collinear_points(self):
        points = np.array([[0, 0], [1, 1], [2, 2]])
        triangles = self.triangulator.compute_triangles(points)
        self.assertEqual(triangles, [])

    def test_multiple_points(self):
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
        triangles = self.triangulator.compute_triangles(points)
        self.assertGreater(len(triangles), 0)

    def test_get_edges(self):
        points = np.array([[0, 0], [1, 0], [0, 1]])
        triangles = self.triangulator.compute_triangles(points)
        edges = self.triangulator.get_edges(triangles)
        expected_edges = {(0, 1), (1, 2), (2, 0)}
        self.assertEqual(edges, expected_edges)

if __name__ == '__main__':
    unittest.main()