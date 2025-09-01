import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.centerline_plotter import CenterlinePlotter

class TestCenterlinePlotter(unittest.TestCase):

    def setUp(self):
        self.plotter = CenterlinePlotter()

    def test_plot_centerline(self):
        points = np.array([[0, 0], [1, 1], [2, 0], [1, -1]])
        self.plotter.plot_centerline(points)
        plt.close()  # Close the plot to avoid displaying it during tests

    def test_save_plot(self):
        points = np.array([[0, 0], [1, 1], [2, 0], [1, -1]])
        self.plotter.plot_centerline(points)
        self.plotter.save_plot('test_plot.png')
        # Check if the file was created
        self.assertTrue(os.path.exists('test_plot.png'))
        os.remove('test_plot.png')  # Clean up after test

    def test_empty_plot(self):
        points = np.array([])
        with self.assertRaises(ValueError):
            self.plotter.plot_centerline(points)

if __name__ == '__main__':
    unittest.main()