"""
Módulo de procesamiento básico de imágenes
Incluye conversión a escala de grises y binarización
"""

import cv2
import numpy as np


def rgb_a_grises(imagen):
    """
    Convierte una imagen RGB a escala de grises
    
    Args:
        imagen: Imagen BGR (formato OpenCV)
    
    Returns:
        Imagen en escala de grises
    """
    if len(imagen.shape) == 3:
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return imagen


def binarizacion_umbral_fijo(imagen, umbral=127):
    """
    Binariza una imagen usando un umbral fijo
    
    Args:
        imagen: Imagen en escala de grises
        umbral: Valor del umbral (0-255)
    
    Returns:
        Imagen binarizada y valor del umbral usado
    """
    # Convertir a grises si es necesario
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    _, img_binaria = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)
    return img_binaria, umbral


def binarizacion_umbral_otsu(imagen):
    """
    Binariza una imagen usando el método de Otsu (umbral automático)
    
    Args:
        imagen: Imagen en escala de grises
    
    Returns:
        Imagen binarizada y valor del umbral calculado
    """
    # Convertir a grises si es necesario
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    umbral_otsu, img_binaria = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_binaria, int(umbral_otsu)


def calcular_histograma(imagen):
    """
    Calcula el histograma de intensidad de una imagen
    
    Args:
        imagen: Imagen en escala de grises
    
    Returns:
        Histograma de 256 bins
    """
    # Convertir a grises si es necesario
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256])
    return histograma.flatten()
