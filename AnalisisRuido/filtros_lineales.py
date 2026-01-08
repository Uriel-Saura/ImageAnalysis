"""
Módulo para aplicación de filtros lineales (paso altas y paso bajas).
"""
import numpy as np
import cv2
from scipy import ndimage


# ========================= FILTROS PASO ALTAS =========================

# ----------------- Operadores de Primer Orden -----------------

def filtro_sobel(imagen):
    """
    Aplica el operador Sobel para detección de bordes.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Operadores Sobel
    sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitud del gradiente
    magnitud = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitud = np.uint8(np.clip(magnitud, 0, 255))
    
    return magnitud


def filtro_prewitt(imagen):
    """
    Aplica el operador Prewitt para detección de bordes.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Máscaras Prewitt
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=np.float32)
    
    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]], dtype=np.float32)
    
    prewitt_x = cv2.filter2D(imagen, cv2.CV_64F, kernel_x)
    prewitt_y = cv2.filter2D(imagen, cv2.CV_64F, kernel_y)
    
    magnitud = np.sqrt(prewitt_x**2 + prewitt_y**2)
    magnitud = np.uint8(np.clip(magnitud, 0, 255))
    
    return magnitud


def filtro_roberts(imagen):
    """
    Aplica el operador Roberts para detección de bordes.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Máscaras Roberts
    kernel_x = np.array([[1, 0],
                         [0, -1]], dtype=np.float32)
    
    kernel_y = np.array([[0, 1],
                         [-1, 0]], dtype=np.float32)
    
    roberts_x = cv2.filter2D(imagen, cv2.CV_64F, kernel_x)
    roberts_y = cv2.filter2D(imagen, cv2.CV_64F, kernel_y)
    
    magnitud = np.sqrt(roberts_x**2 + roberts_y**2)
    magnitud = np.uint8(np.clip(magnitud, 0, 255))
    
    return magnitud


def filtro_kirsch(imagen):
    """
    Aplica el operador Kirsch para detección de bordes.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Las 8 máscaras direccionales de Kirsch
    kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
    ]
    
    # Aplicar todas las máscaras y tomar el máximo
    resultados = []
    for kernel in kernels:
        resultado = cv2.filter2D(imagen, cv2.CV_64F, kernel)
        resultados.append(resultado)
    
    magnitud = np.maximum.reduce(resultados)
    magnitud = np.uint8(np.clip(magnitud, 0, 255))
    
    return magnitud


def filtro_canny(imagen, umbral1=100, umbral2=200):
    """
    Aplica el detector de bordes Canny.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    bordes = cv2.Canny(imagen, umbral1, umbral2)
    
    return bordes


# ----------------- Operadores de Segundo Orden -----------------

def filtro_laplaciano_clasico(imagen):
    """
    Aplica el operador Laplaciano con máscara clásica.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Máscara clásica del Laplaciano
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    
    laplaciano = cv2.filter2D(imagen, cv2.CV_64F, kernel)
    laplaciano = np.uint8(np.clip(np.abs(laplaciano), 0, 255))
    
    return laplaciano


def filtro_laplaciano_8_vecinos(imagen):
    """
    Aplica el operador Laplaciano con máscara de 8 vecinos.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Máscara de 8 vecinos
    kernel = np.array([[1, 1, 1],
                       [1, -8, 1],
                       [1, 1, 1]], dtype=np.float32)
    
    laplaciano = cv2.filter2D(imagen, cv2.CV_64F, kernel)
    laplaciano = np.uint8(np.clip(np.abs(laplaciano), 0, 255))
    
    return laplaciano


def filtro_laplaciano_horizontal(imagen):
    """
    Aplica el operador Laplaciano direccional horizontal.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    kernel = np.array([[0, 0, 0],
                       [1, -2, 1],
                       [0, 0, 0]], dtype=np.float32)
    
    laplaciano = cv2.filter2D(imagen, cv2.CV_64F, kernel)
    laplaciano = np.uint8(np.clip(np.abs(laplaciano), 0, 255))
    
    return laplaciano


def filtro_laplaciano_vertical(imagen):
    """
    Aplica el operador Laplaciano direccional vertical.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    kernel = np.array([[0, 1, 0],
                       [0, -2, 0],
                       [0, 1, 0]], dtype=np.float32)
    
    laplaciano = cv2.filter2D(imagen, cv2.CV_64F, kernel)
    laplaciano = np.uint8(np.clip(np.abs(laplaciano), 0, 255))
    
    return laplaciano


def filtro_laplaciano_diagonal_principal(imagen):
    """
    Aplica el operador Laplaciano direccional diagonal principal.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    kernel = np.array([[1, 0, 0],
                       [0, -2, 0],
                       [0, 0, 1]], dtype=np.float32)
    
    laplaciano = cv2.filter2D(imagen, cv2.CV_64F, kernel)
    laplaciano = np.uint8(np.clip(np.abs(laplaciano), 0, 255))
    
    return laplaciano


def filtro_laplaciano_diagonal_secundaria(imagen):
    """
    Aplica el operador Laplaciano direccional diagonal secundaria.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    kernel = np.array([[0, 0, 1],
                       [0, -2, 0],
                       [1, 0, 0]], dtype=np.float32)
    
    laplaciano = cv2.filter2D(imagen, cv2.CV_64F, kernel)
    laplaciano = np.uint8(np.clip(np.abs(laplaciano), 0, 255))
    
    return laplaciano


# ========================= FILTROS PASO BAJAS =========================

def filtro_promediador(imagen, tamano_kernel=5):
    """
    Aplica un filtro promediador (blur) simple.
    """
    # Asegurar que el tamaño del kernel sea positivo
    if tamano_kernel < 1:
        tamano_kernel = 1
    
    resultado = cv2.blur(imagen, (tamano_kernel, tamano_kernel))
    return resultado


def filtro_promediador_pesado(imagen):
    """
    Aplica un filtro promediador con pesos.
    """
    # Kernel con pesos hacia el centro
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32) / 16
    
    if len(imagen.shape) == 3:
        resultado = cv2.filter2D(imagen, -1, kernel)
    else:
        resultado = cv2.filter2D(imagen, -1, kernel)
    
    return resultado


def filtro_gaussiano(imagen, tamano_kernel=5, sigma=1.0):
    """
    Aplica un filtro gaussiano para suavizado.
    """
    # Asegurar que el tamaño del kernel sea impar y positivo
    if tamano_kernel < 1:
        tamano_kernel = 1
    if tamano_kernel % 2 == 0:
        tamano_kernel += 1
    
    resultado = cv2.GaussianBlur(imagen, (tamano_kernel, tamano_kernel), sigma)
    return resultado


def filtro_bilateral(imagen, d=9, sigma_color=75, sigma_space=75):
    """
    Aplica un filtro bilateral que preserva bordes.
    """
    resultado = cv2.bilateralFilter(imagen, d, sigma_color, sigma_space)
    return resultado
