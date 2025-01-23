import numpy as np
from noise import pnoise2  # Bibliothèque de bruit Perlin

def map_colored(width=200, height = 200):
    scale = 150  # Échelle du bruit (plus grand = paysage plus étendu)

    # Génération d'une carte d'altitude avec du bruit de Perlin
    altitude_map = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            altitude_map[i, j] = pnoise2(i / scale, j / scale, octaves=6, persistence=0.5, lacunarity=2.0)

    # Normalisation entre 0 et 1
    altitude_map = (altitude_map - altitude_map.min()) / (altitude_map.max() - altitude_map.min())
   

    curve_exponent = 1.5  # Higher values emphasize higher altitudes
    altitude_map = altitude_map**curve_exponent#altitude_map*((altitude_map)**curve_exponent-(altitude_map))

    # Normalisation entre 0 et 1
    altitude_map = (altitude_map - altitude_map.min()) / (altitude_map.max() - altitude_map.min())
    altitude_map = altitude_map*0.2

    # Définir les couleurs selon l'altitude
    def altitude_to_color(altitude):
        """Renvoie une couleur (R, G, B) en fonction de l'altitude."""
        rel = altitude_map.max()
        if altitude < 0.2*rel:
            return [0, 0, 1]  # Bleu pour la mer
        elif altitude < 0.25*rel:
            return [0.9, 0.7, 0.5]  # Beige pour le sable
        elif altitude < 0.4*rel:
            return [0, 1, 0]  # Vert pour la végétation
        elif altitude < 0.8*rel:
            return [0.5, 0.5, 0.5]
        else:
            return [1, 1, 1]  # Blanc pour la neige

    # Application des couleurs
    colors = np.array([altitude_to_color(a) for a in altitude_map.flatten(order="F")])

    return width, height, altitude_map, colors



