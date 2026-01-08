"""
Técnicas de Ecualización de Histograma
Implementa diferentes métodos de ecualización para mejorar el contraste
"""

import numpy as np
import cv2
from typing import Tuple


def ecualizacion_uniforme(imagen: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ecualización uniforme: Mejora contraste global mediante distribución uniforme del histograma.
    
    Args:
        imagen: Imagen en escala de grises
    
    Returns:
        Tupla (imagen_ecualizada, histograma_original, histograma_ecualizado)
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Histograma original
    hist_original = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()
    
    # Ecualización usando OpenCV
    img_ecualizada = cv2.equalizeHist(imagen)
    
    # Histograma ecualizado
    hist_ecualizado = cv2.calcHist([img_ecualizada], [0], None, [256], [0, 256]).flatten()
    
    return img_ecualizada, hist_original, hist_ecualizado


def ecualizacion_exponencial(imagen: np.ndarray, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ecualización exponencial: Resalta tonos oscuros usando distribución exponencial.
    
    Args:
        imagen: Imagen en escala de grises
        alpha: Parámetro de la distribución exponencial (mayor valor = más resalte)
    
    Returns:
        Tupla (imagen_ecualizada, histograma_original, histograma_ecualizado)
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Histograma original
    hist_original = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()
    
    # Calcular CDF (Cumulative Distribution Function)
    hist = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()
    cdf = hist.cumsum()
    cdf_normalizado = cdf / cdf[-1]
    
    # Aplicar función de transformación exponencial
    # T(r) = -1/alpha * ln(1 - cdf(r))
    # Normalizar para evitar valores negativos
    transformacion = np.zeros(256)
    for i in range(256):
        valor = 1 - cdf_normalizado[i]
        if valor > 0:
            transformacion[i] = -1/alpha * np.log(valor)
    
    # Normalizar la transformación al rango [0, 255]
    transformacion = transformacion - transformacion.min()
    if transformacion.max() > 0:
        transformacion = 255 * transformacion / transformacion.max()
    
    # Aplicar la transformación
    img_ecualizada = cv2.LUT(imagen, transformacion.astype(np.uint8))
    
    # Histograma ecualizado
    hist_ecualizado = cv2.calcHist([img_ecualizada], [0], None, [256], [0, 256]).flatten()
    
    return img_ecualizada, hist_original, hist_ecualizado


def ecualizacion_rayleigh(imagen: np.ndarray, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ecualización Rayleigh: Favorece regiones claras usando distribución Rayleigh.
    
    Args:
        imagen: Imagen en escala de grises
        alpha: Parámetro de la distribución Rayleigh
    
    Returns:
        Tupla (imagen_ecualizada, histograma_original, histograma_ecualizado)
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Histograma original
    hist_original = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()
    
    # Calcular CDF
    hist = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()
    cdf = hist.cumsum()
    cdf_normalizado = cdf / cdf[-1]
    
    # Aplicar función de transformación Rayleigh
    # T(r) = sqrt(-2*alpha^2 * ln(1 - cdf(r)))
    transformacion = np.zeros(256)
    for i in range(256):
        valor = 1 - cdf_normalizado[i]
        if valor > 0:
            transformacion[i] = np.sqrt(-2 * alpha**2 * np.log(valor))
    
    # Normalizar la transformación al rango [0, 255]
    transformacion = transformacion - transformacion.min()
    if transformacion.max() > 0:
        transformacion = 255 * transformacion / transformacion.max()
    
    # Aplicar la transformación
    img_ecualizada = cv2.LUT(imagen, transformacion.astype(np.uint8))
    
    # Histograma ecualizado
    hist_ecualizado = cv2.calcHist([img_ecualizada], [0], None, [256], [0, 256]).flatten()
    
    return img_ecualizada, hist_original, hist_ecualizado


def ecualizacion_hipercubica(imagen: np.ndarray, alpha: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ecualización hipercúbica: Potencia diferencias extremas.
    
    Args:
        imagen: Imagen en escala de grises
        alpha: Parámetro de potencia (valores típicos: 1.5 - 3.0)
    
    Returns:
        Tupla (imagen_ecualizada, histograma_original, histograma_ecualizado)
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Histograma original
    hist_original = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()
    
    # Calcular CDF
    hist = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()
    cdf = hist.cumsum()
    cdf_normalizado = cdf / cdf[-1]
    
    # Aplicar función de transformación hipercúbica
    # T(r) = cdf(r)^alpha
    transformacion = np.power(cdf_normalizado, alpha)
    
    # Escalar al rango [0, 255]
    transformacion = (transformacion * 255).astype(np.uint8)
    
    # Aplicar la transformación
    img_ecualizada = cv2.LUT(imagen, transformacion)
    
    # Histograma ecualizado
    hist_ecualizado = cv2.calcHist([img_ecualizada], [0], None, [256], [0, 256]).flatten()
    
    return img_ecualizada, hist_original, hist_ecualizado


def ecualizacion_logaritmica_hiperbolica(imagen: np.ndarray, c: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ecualización logarítmica hiperbólica: Mejora detalles en sombras.
    
    Args:
        imagen: Imagen en escala de grises
        c: Constante de ajuste
    
    Returns:
        Tupla (imagen_ecualizada, histograma_original, histograma_ecualizado)
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Histograma original
    hist_original = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()
    
    # Calcular CDF
    hist = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()
    cdf = hist.cumsum()
    cdf_normalizado = cdf / cdf[-1]
    
    # Aplicar función de transformación logarítmica hiperbólica
    # T(r) = c * log(1 + cdf(r))
    transformacion = c * np.log(1 + cdf_normalizado)
    
    # Normalizar al rango [0, 255]
    transformacion = transformacion - transformacion.min()
    if transformacion.max() > 0:
        transformacion = 255 * transformacion / transformacion.max()
    
    # Aplicar la transformación
    img_ecualizada = cv2.LUT(imagen, transformacion.astype(np.uint8))
    
    # Histograma ecualizado
    hist_ecualizado = cv2.calcHist([img_ecualizada], [0], None, [256], [0, 256]).flatten()
    
    return img_ecualizada, hist_original, hist_ecualizado


def ecualizacion_clahe(imagen: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization): 
    Ecualización adaptativa que limita el contraste para evitar sobre-amplificación del ruido.
    
    Args:
        imagen: Imagen en escala de grises
        clip_limit: Límite de contraste
        tile_size: Tamaño de los bloques (tiles)
    
    Returns:
        Tupla (imagen_ecualizada, histograma_original, histograma_ecualizado)
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Histograma original
    hist_original = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()
    
    # Aplicar CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    img_ecualizada = clahe.apply(imagen)
    
    # Histograma ecualizado
    hist_ecualizado = cv2.calcHist([img_ecualizada], [0], None, [256], [0, 256]).flatten()
    
    return img_ecualizada, hist_original, hist_ecualizado


def ecualizacion_personalizada(imagen: np.ndarray, funcion_transformacion: callable) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Permite aplicar una función de transformación personalizada al histograma.
    
    Args:
        imagen: Imagen en escala de grises
        funcion_transformacion: Función que toma valores [0,1] y retorna valores [0,1]
    
    Returns:
        Tupla (imagen_ecualizada, histograma_original, histograma_ecualizado)
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Histograma original
    hist_original = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()
    
    # Calcular CDF
    hist = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()
    cdf = hist.cumsum()
    cdf_normalizado = cdf / cdf[-1]
    
    # Aplicar función de transformación personalizada
    transformacion = np.array([funcion_transformacion(val) for val in cdf_normalizado])
    
    # Escalar al rango [0, 255]
    transformacion = (transformacion * 255).astype(np.uint8)
    
    # Aplicar la transformación
    img_ecualizada = cv2.LUT(imagen, transformacion)
    
    # Histograma ecualizado
    hist_ecualizado = cv2.calcHist([img_ecualizada], [0], None, [256], [0, 256]).flatten()
    
    return img_ecualizada, hist_original, hist_ecualizado
