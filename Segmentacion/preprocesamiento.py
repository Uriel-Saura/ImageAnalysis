"""
Módulo de Preprocesamiento Básico
Solo incluye funciones auxiliares simples
"""

import numpy as np
import cv2


def convertir_a_grises(imagen: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen a escala de grises si es necesario.
    
    Args:
        imagen: Imagen de entrada
    
    Returns:
        Imagen en escala de grises
    """
    if len(imagen.shape) == 3:
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return imagen
