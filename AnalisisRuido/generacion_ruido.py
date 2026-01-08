"""
Módulo para generación de diferentes tipos de ruido en imágenes.
"""
import numpy as np
import cv2


def aplicar_ruido_sal_pimienta(imagen, probabilidad=0.05):
    """
    Aplica ruido sal y pimienta a una imagen.
    
    Args:
        imagen: Imagen de entrada (numpy array)
        probabilidad: Probabilidad de que un píxel sea afectado (0-1)
        
    Returns:
        Imagen con ruido sal y pimienta
    """
    imagen_ruido = imagen.copy()
    
    # Generar máscara de ruido aleatorio
    ruido = np.random.random(imagen.shape[:2])
    
    # Sal (píxeles blancos)
    imagen_ruido[ruido < probabilidad / 2] = 255
    
    # Pimienta (píxeles negros)
    imagen_ruido[ruido > 1 - probabilidad / 2] = 0
    
    return imagen_ruido


def aplicar_ruido_sal(imagen, probabilidad=0.05):
    """
    Aplica solo ruido sal (píxeles blancos) a una imagen.
    
    Args:
        imagen: Imagen de entrada (numpy array)
        probabilidad: Probabilidad de que un píxel sea afectado (0-1)
        
    Returns:
        Imagen con ruido sal
    """
    imagen_ruido = imagen.copy()
    
    # Generar máscara de ruido aleatorio
    ruido = np.random.random(imagen.shape[:2])
    
    # Sal (píxeles blancos)
    imagen_ruido[ruido < probabilidad] = 255
    
    return imagen_ruido


def aplicar_ruido_pimienta(imagen, probabilidad=0.05):
    """
    Aplica solo ruido pimienta (píxeles negros) a una imagen.
    
    Args:
        imagen: Imagen de entrada (numpy array)
        probabilidad: Probabilidad de que un píxel sea afectado (0-1)
        
    Returns:
        Imagen con ruido pimienta
    """
    imagen_ruido = imagen.copy()
    
    # Generar máscara de ruido aleatorio
    ruido = np.random.random(imagen.shape[:2])
    
    # Pimienta (píxeles negros)
    imagen_ruido[ruido < probabilidad] = 0
    
    return imagen_ruido


def aplicar_ruido_gaussiano(imagen, media=0, sigma=25):
    """
    Aplica ruido gaussiano a una imagen.
    
    Args:
        imagen: Imagen de entrada (numpy array)
        media: Media de la distribución gaussiana
        sigma: Desviación estándar de la distribución gaussiana
        
    Returns:
        Imagen con ruido gaussiano
    """
    imagen_ruido = imagen.copy().astype(np.float32)
    
    # Generar ruido gaussiano
    if len(imagen.shape) == 3:
        ruido = np.random.normal(media, sigma, imagen.shape)
    else:
        ruido = np.random.normal(media, sigma, imagen.shape)
    
    # Añadir ruido a la imagen
    imagen_ruido = imagen_ruido + ruido
    
    # Recortar valores fuera del rango [0, 255]
    imagen_ruido = np.clip(imagen_ruido, 0, 255)
    
    return imagen_ruido.astype(np.uint8)


def calcular_histograma(imagen):
    """
    Calcula el histograma de una imagen en escala de grises o de cada canal.
    
    Args:
        imagen: Imagen de entrada
        
    Returns:
        Lista de histogramas para cada canal
    """
    if len(imagen.shape) == 2:
        # Imagen en escala de grises
        hist = cv2.calcHist([imagen], [0], None, [256], [0, 256])
        return [hist]
    else:
        # Imagen en color - calcular histograma para cada canal
        histogramas = []
        for i in range(imagen.shape[2]):
            hist = cv2.calcHist([imagen], [i], None, [256], [0, 256])
            histogramas.append(hist)
        return histogramas
