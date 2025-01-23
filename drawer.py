import numpy as np
import pyvista as pv

def draw_mesh(width, height, altitude_map, colors):
    # Création de la grille 3D
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    x, y = np.meshgrid(x, y)
    z = altitude_map  # Les altitudes définissent la hauteur

    # Transformation en maillage PyVista
    grid = pv.StructuredGrid(x, y, z)
    grid["colors"] = colors

    # Visualisation avec PyVista
    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars="colors", rgb=True)
    plotter.add_axes()
    plotter.add_text("Paysage avec bruit de Perlin", position="upper_left", font_size=10)
    plotter.show_grid()  # Afficher une grille sur la scène

    plotter.show()