"""
Módulo de procesamiento básico de imágenes
Incluye conversión a escala de grises, binarización, separación de canales,
conversiones entre modelos de color y análisis de histogramas
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List


# ===================================================================
# 1.4 LECTURA Y OBTENCIÓN DE IMÁGENES
# ===================================================================

def leer_imagen(ruta: str, modo='color'):
    """
    Lee una imagen desde archivo
    
    Args:
        ruta: Ruta del archivo de imagen
        modo: 'color' para BGR, 'grises' para escala de grises
    
    Returns:
        Imagen cargada o None si hay error
    """
    if modo == 'grises':
        return cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(ruta, cv2.IMREAD_COLOR)


def obtener_propiedades_pixel(imagen, x, y):
    """
    Obtiene las propiedades de un píxel específico
    
    Args:
        imagen: Imagen cargada
        x, y: Coordenadas del píxel
    
    Returns:
        Diccionario con propiedades del píxel
    """
    if imagen is None:
        return None
    
    h, w = imagen.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return None
    
    propiedades = {
        'posicion': (x, y),
        'dimensiones_imagen': (w, h)
    }
    
    if len(imagen.shape) == 3:
        b, g, r = imagen[y, x]
        propiedades['color'] = 'RGB'
        propiedades['valores'] = {'R': int(r), 'G': int(g), 'B': int(b)}
    else:
        propiedades['color'] = 'Grises'
        propiedades['valores'] = {'Intensidad': int(imagen[y, x])}
    
    return propiedades


# ===================================================================
# 1.5 SEPARACIÓN DE CANALES RGB
# ===================================================================

def separar_canales_rgb(imagen):
    """
    Separa los canales R, G, B de una imagen
    
    Args:
        imagen: Imagen BGR (formato OpenCV)
    
    Returns:
        Tupla (canal_r, canal_g, canal_b)
    """
    if len(imagen.shape) != 3:
        raise ValueError("La imagen debe ser a color (3 canales)")
    
    # OpenCV usa BGR, así que separamos en orden inverso
    b, g, r = cv2.split(imagen)
    
    return r, g, b


def separar_canales_rgb_visualizar(imagen):
    """
    Separa canales RGB y crea versiones coloreadas para visualización
    
    Args:
        imagen: Imagen BGR
    
    Returns:
        Diccionario con canales en blanco y negro y coloreados
    """
    r, g, b = separar_canales_rgb(imagen)
    
    # Crear versiones coloreadas (solo el canal correspondiente activo)
    canal_r_color = cv2.merge([np.zeros_like(r), np.zeros_like(r), r])
    canal_g_color = cv2.merge([np.zeros_like(g), g, np.zeros_like(g)])
    canal_b_color = cv2.merge([b, np.zeros_like(b), np.zeros_like(b)])
    
    return {
        'r_bn': r,
        'g_bn': g,
        'b_bn': b,
        'r_color': canal_r_color,
        'g_color': canal_g_color,
        'b_color': canal_b_color
    }


# ===================================================================
# 1.6 ESCALAMIENTO A GRISES
# ===================================================================

def rgb_a_grises(imagen):
    """
    Convierte una imagen RGB a escala de grises
    Usa la fórmula estándar: Y = 0.299*R + 0.587*G + 0.114*B
    
    Args:
        imagen: Imagen BGR (formato OpenCV)
    
    Returns:
        Imagen en escala de grises
    """
    if len(imagen.shape) == 3:
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return imagen


def rgb_a_grises_promedio(imagen):
    """
    Convierte RGB a grises usando el promedio simple
    
    Args:
        imagen: Imagen BGR
    
    Returns:
        Imagen en escala de grises
    """
    if len(imagen.shape) != 3:
        return imagen
    
    return np.mean(imagen, axis=2).astype(np.uint8)


# ===================================================================
# 1.7 BINARIZACIÓN
# ===================================================================

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


# ===================================================================
# MODELOS DE COLOR
# ===================================================================

def rgb_a_cmy(imagen):
    """
    Convierte imagen RGB a modelo CMY (Cyan, Magenta, Yellow)
    CMY = 1 - RGB (normalizado)
    
    Args:
        imagen: Imagen BGR de OpenCV
    
    Returns:
        Imagen CMY (3 canales)
    """
    if len(imagen.shape) != 3:
        raise ValueError("La imagen debe ser a color")
    
    # Convertir BGR a RGB
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    
    # Normalizar a [0, 1]
    imagen_norm = imagen_rgb.astype(np.float32) / 255.0
    
    # Aplicar transformación CMY
    imagen_cmy = 1.0 - imagen_norm
    
    # Volver a [0, 255]
    return (imagen_cmy * 255).astype(np.uint8)


def rgb_a_yiq(imagen):
    """
    Convierte RGB a YIQ (usado en televisión NTSC)
    Y = luminancia, I y Q = crominancia
    
    Args:
        imagen: Imagen BGR
    
    Returns:
        Imagen YIQ (3 canales)
    """
    if len(imagen.shape) != 3:
        raise ValueError("La imagen debe ser a color")
    
    # Convertir BGR a RGB
    b, g, r = cv2.split(imagen)
    
    # Matriz de transformación RGB a YIQ
    y = 0.299 * r + 0.587 * g + 0.114 * b
    i = 0.596 * r - 0.275 * g - 0.321 * b
    q = 0.212 * r - 0.523 * g + 0.311 * b
    
    # Normalizar I y Q al rango [0, 255]
    i = ((i + 151.9) * 255 / 303.8).clip(0, 255).astype(np.uint8)
    q = ((q + 133.3) * 255 / 266.6).clip(0, 255).astype(np.uint8)
    y = y.astype(np.uint8)
    
    return cv2.merge([y, i, q])


def rgb_a_hsi(imagen):
    """
    Convierte RGB a HSI (Hue, Saturation, Intensity)
    
    Args:
        imagen: Imagen BGR
    
    Returns:
        Imagen HSI (3 canales, valores normalizados a 0-255)
    """
    if len(imagen.shape) != 3:
        raise ValueError("La imagen debe ser a color")
    
    # Convertir a HSV primero (OpenCV lo maneja bien)
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    
    # En HSV: H está en [0, 179], S y V en [0, 255]
    # Para HSI necesitamos ajustar
    h, s, v = cv2.split(imagen_hsv)
    
    # Convertir BGR a RGB normalizado
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r, g, b = cv2.split(imagen_rgb)
    
    # Calcular Intensity
    i = (r + g + b) / 3.0
    
    # Usar H y S de HSV pero ajustados
    h_rad = h.astype(np.float32) * 2 * np.pi / 179.0  # Convertir a radianes
    s_norm = s.astype(np.float32) / 255.0
    
    # Volver a escala 0-255 para visualización
    h_out = (h.astype(np.float32) * 255 / 179).astype(np.uint8)
    s_out = s
    i_out = (i * 255).astype(np.uint8)
    
    return cv2.merge([h_out, s_out, i_out])


def rgb_a_hsv(imagen):
    """
    Convierte RGB a HSV (Hue, Saturation, Value)
    
    Args:
        imagen: Imagen BGR
    
    Returns:
        Imagen HSV (3 canales)
    """
    if len(imagen.shape) != 3:
        raise ValueError("La imagen debe ser a color")
    
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)


# ===================================================================
# ANÁLISIS DE HISTOGRAMAS
# ===================================================================

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


def calcular_histogramas_rgb(imagen):
    """
    Calcula histogramas separados para cada canal RGB
    
    Args:
        imagen: Imagen BGR
    
    Returns:
        Diccionario con histogramas de R, G, B
    """
    if len(imagen.shape) != 3:
        raise ValueError("La imagen debe ser a color")
    
    # OpenCV usa BGR
    b, g, r = cv2.split(imagen)
    
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]).flatten()
    
    return {
        'R': hist_r,
        'G': hist_g,
        'B': hist_b
    }


def propiedades_histograma(histograma):
    """
    Calcula propiedades estadísticas del histograma
    
    Args:
        histograma: Array del histograma
    
    Returns:
        Diccionario con propiedades: media, mediana, moda, varianza, desviación estándar
    """
    # Crear array de valores de intensidad
    valores = np.arange(256)
    
    # Calcular media ponderada
    total_pixels = np.sum(histograma)
    if total_pixels == 0:
        return None
    
    media = np.sum(valores * histograma) / total_pixels
    
    # Calcular varianza y desviación estándar
    varianza = np.sum(((valores - media) ** 2) * histograma) / total_pixels
    desviacion = np.sqrt(varianza)
    
    # Moda (valor más frecuente)
    moda = np.argmax(histograma)
    
    # Mediana (valor acumulado hasta 50%)
    hist_acumulado = np.cumsum(histograma)
    mediana = np.where(hist_acumulado >= total_pixels / 2)[0][0]
    
    # Rango
    valores_no_cero = np.where(histograma > 0)[0]
    if len(valores_no_cero) > 0:
        minimo = valores_no_cero[0]
        maximo = valores_no_cero[-1]
        rango = maximo - minimo
    else:
        minimo = maximo = rango = 0
    
    return {
        'media': float(media),
        'mediana': int(mediana),
        'moda': int(moda),
        'varianza': float(varianza),
        'desviacion_estandar': float(desviacion),
        'minimo': int(minimo),
        'maximo': int(maximo),
        'rango': int(rango),
        'total_pixeles': int(total_pixels)
    }
