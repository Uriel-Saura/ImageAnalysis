"""
Técnicas de Ajuste de Histograma
Implementa funciones de transformación para ajustar brillo y contraste
"""

import numpy as np
import cv2
from typing import Tuple


def funcion_potencia(imagen: np.ndarray, c: float = 1.0, gamma: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Función potencia: Ajusta contraste no lineal mediante s = c * r^gamma
    
    Args:
        imagen: Imagen en escala de grises
        c: Constante de escala
        gamma: Exponente de potencia (gamma < 1 aclara, gamma > 1 oscurece)
    
    Returns:
        Tupla (imagen_ajustada, histograma_original, histograma_ajustado)
    """
    if len(imagen.shape) == 3:
        imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gray = imagen.copy()
    
    # Histograma original
    hist_original = cv2.calcHist([imagen_gray], [0], None, [256], [0, 256]).flatten()
    
    # Normalizar imagen al rango [0, 1]
    imagen_norm = imagen_gray.astype(np.float32) / 255.0
    
    # Aplicar función potencia
    imagen_ajustada = c * np.power(imagen_norm, gamma)
    
    # Recortar valores fuera del rango y escalar a [0, 255]
    imagen_ajustada = np.clip(imagen_ajustada, 0, 1)
    imagen_ajustada = (imagen_ajustada * 255).astype(np.uint8)
    
    # Histograma ajustado
    hist_ajustado = cv2.calcHist([imagen_ajustada], [0], None, [256], [0, 256]).flatten()
    
    return imagen_ajustada, hist_original, hist_ajustado


def correccion_gamma(imagen: np.ndarray, gamma: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Corrección gamma: Controla brillo global de la imagen.
    
    Args:
        imagen: Imagen en escala de grises o color
        gamma: Valor de gamma (< 1 aclara, > 1 oscurece, 1 = sin cambio)
    
    Returns:
        Tupla (imagen_corregida, histograma_original, histograma_corregido)
    """
    if len(imagen.shape) == 3:
        imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gray = imagen.copy()
    
    # Histograma original
    hist_original = cv2.calcHist([imagen_gray], [0], None, [256], [0, 256]).flatten()
    
    # Crear tabla de lookup para la corrección gamma
    tabla = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    
    # Aplicar corrección gamma
    if len(imagen.shape) == 3:
        imagen_corregida = cv2.LUT(imagen, tabla)
        imagen_corregida_gray = cv2.cvtColor(imagen_corregida, cv2.COLOR_BGR2GRAY)
    else:
        imagen_corregida = cv2.LUT(imagen, tabla)
        imagen_corregida_gray = imagen_corregida
    
    # Histograma corregido
    hist_corregido = cv2.calcHist([imagen_corregida_gray], [0], None, [256], [0, 256]).flatten()
    
    if len(imagen.shape) == 3:
        return imagen_corregida, hist_original, hist_corregido
    else:
        return imagen_corregida, hist_original, hist_corregido


def desplazamiento_histograma(imagen: np.ndarray, desplazamiento: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Desplazamiento del histograma: Aumenta o disminuye el brillo.
    Suma un valor constante a todos los píxeles.
    
    Args:
        imagen: Imagen en escala de grises
        desplazamiento: Valor a sumar (positivo aclara, negativo oscurece)
    
    Returns:
        Tupla (imagen_desplazada, histograma_original, histograma_desplazado)
    """
    if len(imagen.shape) == 3:
        imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gray = imagen.copy()
    
    # Histograma original
    hist_original = cv2.calcHist([imagen_gray], [0], None, [256], [0, 256]).flatten()
    
    # Aplicar desplazamiento con saturación
    imagen_desplazada = cv2.add(imagen_gray, desplazamiento)
    
    # Histograma desplazado
    hist_desplazado = cv2.calcHist([imagen_desplazada], [0], None, [256], [0, 256]).flatten()
    
    return imagen_desplazada, hist_original, hist_desplazado


def contraccion_histograma(imagen: np.ndarray, factor: float = 0.5, 
                           centro: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Contracción del histograma: Reduce el contraste concentrando valores hacia un punto central.
    
    Args:
        imagen: Imagen en escala de grises
        factor: Factor de contracción (0-1, donde 0 = contracción total, 1 = sin cambio)
        centro: Punto central hacia el que se contraen los valores
    
    Returns:
        Tupla (imagen_contraida, histograma_original, histograma_contraido)
    """
    if len(imagen.shape) == 3:
        imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gray = imagen.copy()
    
    # Histograma original
    hist_original = cv2.calcHist([imagen_gray], [0], None, [256], [0, 256]).flatten()
    
    # Aplicar contracción: nuevo_valor = centro + factor * (valor_original - centro)
    imagen_float = imagen_gray.astype(np.float32)
    imagen_contraida = centro + factor * (imagen_float - centro)
    
    # Recortar y convertir a uint8
    imagen_contraida = np.clip(imagen_contraida, 0, 255).astype(np.uint8)
    
    # Histograma contraído
    hist_contraido = cv2.calcHist([imagen_contraida], [0], None, [256], [0, 256]).flatten()
    
    return imagen_contraida, hist_original, hist_contraido


def expansion_histograma(imagen: np.ndarray, min_out: int = 0, 
                         max_out: int = 255) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expansión del histograma: Aumenta el contraste expandiendo el rango de intensidades.
    Estira el histograma desde [min_in, max_in] hasta [min_out, max_out].
    
    Args:
        imagen: Imagen en escala de grises
        min_out: Valor mínimo de salida (por defecto 0)
        max_out: Valor máximo de salida (por defecto 255)
    
    Returns:
        Tupla (imagen_expandida, histograma_original, histograma_expandido)
    """
    if len(imagen.shape) == 3:
        imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gray = imagen.copy()
    
    # Histograma original
    hist_original = cv2.calcHist([imagen_gray], [0], None, [256], [0, 256]).flatten()
    
    # Encontrar valores mínimo y máximo actuales
    min_in = imagen_gray.min()
    max_in = imagen_gray.max()
    
    # Evitar división por cero
    if max_in == min_in:
        return imagen_gray.copy(), hist_original, hist_original
    
    # Aplicar expansión lineal
    imagen_float = imagen_gray.astype(np.float32)
    imagen_expandida = (imagen_float - min_in) * (max_out - min_out) / (max_in - min_in) + min_out
    
    # Recortar y convertir a uint8
    imagen_expandida = np.clip(imagen_expandida, 0, 255).astype(np.uint8)
    
    # Histograma expandido
    hist_expandido = cv2.calcHist([imagen_expandida], [0], None, [256], [0, 256]).flatten()
    
    return imagen_expandida, hist_original, hist_expandido


def expansion_histograma_percentil(imagen: np.ndarray, percentil_bajo: float = 2.0, 
                                   percentil_alto: float = 98.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expansión del histograma usando percentiles: Ignora outliers para mejor contraste.
    
    Args:
        imagen: Imagen en escala de grises
        percentil_bajo: Percentil inferior (valores por debajo se mapean a 0)
        percentil_alto: Percentil superior (valores por encima se mapean a 255)
    
    Returns:
        Tupla (imagen_expandida, histograma_original, histograma_expandido)
    """
    if len(imagen.shape) == 3:
        imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gray = imagen.copy()
    
    # Histograma original
    hist_original = cv2.calcHist([imagen_gray], [0], None, [256], [0, 256]).flatten()
    
    # Calcular percentiles
    p_bajo = np.percentile(imagen_gray, percentil_bajo)
    p_alto = np.percentile(imagen_gray, percentil_alto)
    
    # Evitar división por cero
    if p_alto == p_bajo:
        return imagen_gray.copy(), hist_original, hist_original
    
    # Aplicar expansión basada en percentiles
    imagen_float = imagen_gray.astype(np.float32)
    imagen_expandida = (imagen_float - p_bajo) * 255.0 / (p_alto - p_bajo)
    
    # Recortar y convertir a uint8
    imagen_expandida = np.clip(imagen_expandida, 0, 255).astype(np.uint8)
    
    # Histograma expandido
    hist_expandido = cv2.calcHist([imagen_expandida], [0], None, [256], [0, 256]).flatten()
    
    return imagen_expandida, hist_original, hist_expandido


def normalizacion_histograma(imagen: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalización del histograma: Expande el histograma para usar todo el rango [0, 255].
    Equivalente a expansion_histograma con parámetros por defecto.
    
    Args:
        imagen: Imagen en escala de grises
    
    Returns:
        Tupla (imagen_normalizada, histograma_original, histograma_normalizado)
    """
    return expansion_histograma(imagen, 0, 255)


def ajuste_contraste_brillo(imagen: np.ndarray, alpha: float = 1.0, 
                            beta: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ajuste lineal de contraste y brillo: nuevo = alpha * original + beta
    
    Args:
        imagen: Imagen en escala de grises
        alpha: Factor de contraste (1.0 = sin cambio, >1 aumenta, <1 disminuye)
        beta: Desplazamiento de brillo (0 = sin cambio, >0 aclara, <0 oscurece)
    
    Returns:
        Tupla (imagen_ajustada, histograma_original, histograma_ajustado)
    """
    if len(imagen.shape) == 3:
        imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gray = imagen.copy()
    
    # Histograma original
    hist_original = cv2.calcHist([imagen_gray], [0], None, [256], [0, 256]).flatten()
    
    # Aplicar transformación lineal con saturación
    imagen_ajustada = cv2.convertScaleAbs(imagen_gray, alpha=alpha, beta=beta)
    
    # Histograma ajustado
    hist_ajustado = cv2.calcHist([imagen_ajustada], [0], None, [256], [0, 256]).flatten()
    
    return imagen_ajustada, hist_original, hist_ajustado


def transformacion_logaritmica(imagen: np.ndarray, c: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transformación logarítmica: s = c * log(1 + r)
    Expande valores oscuros y comprime valores claros.
    
    Args:
        imagen: Imagen en escala de grises
        c: Constante de escala
    
    Returns:
        Tupla (imagen_transformada, histograma_original, histograma_transformado)
    """
    if len(imagen.shape) == 3:
        imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gray = imagen.copy()
    
    # Histograma original
    hist_original = cv2.calcHist([imagen_gray], [0], None, [256], [0, 256]).flatten()
    
    # Aplicar transformación logarítmica
    imagen_float = imagen_gray.astype(np.float32)
    imagen_transformada = c * np.log(1 + imagen_float)
    
    # Normalizar al rango [0, 255]
    imagen_transformada = cv2.normalize(imagen_transformada, None, 0, 255, cv2.NORM_MINMAX)
    imagen_transformada = imagen_transformada.astype(np.uint8)
    
    # Histograma transformado
    hist_transformado = cv2.calcHist([imagen_transformada], [0], None, [256], [0, 256]).flatten()
    
    return imagen_transformada, hist_original, hist_transformado
