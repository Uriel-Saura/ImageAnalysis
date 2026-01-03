"""
Módulo para aplicación de filtros no lineales (de orden).
"""
import numpy as np
import cv2
from scipy import ndimage
from collections import Counter


def filtro_mediana(imagen, tamano_kernel=5):
    """
    Aplica un filtro de mediana.
    Útil para eliminar ruido sal y pimienta.
    """
    resultado = cv2.medianBlur(imagen, tamano_kernel)
    return resultado


def filtro_moda(imagen, tamano_kernel=5):
    """
    Aplica un filtro de moda (valor más frecuente en la vecindad).
    """
    if len(imagen.shape) == 3:
        # Aplicar a cada canal
        resultado = np.zeros_like(imagen)
        for i in range(imagen.shape[2]):
            resultado[:, :, i] = _aplicar_moda_canal(imagen[:, :, i], tamano_kernel)
        return resultado
    else:
        return _aplicar_moda_canal(imagen, tamano_kernel)


def _aplicar_moda_canal(canal, tamano_kernel):
    """
    Aplica el filtro de moda a un solo canal.
    """
    resultado = np.zeros_like(canal)
    pad = tamano_kernel // 2
    canal_pad = np.pad(canal, pad, mode='reflect')
    
    for i in range(canal.shape[0]):
        for j in range(canal.shape[1]):
            ventana = canal_pad[i:i+tamano_kernel, j:j+tamano_kernel]
            valores = ventana.flatten()
            # Encontrar la moda
            contador = Counter(valores)
            moda = contador.most_common(1)[0][0]
            resultado[i, j] = moda
    
    return resultado


def filtro_maximo(imagen, tamano_kernel=5):
    """
    Aplica un filtro de máximo (dilación).
    """
    kernel = np.ones((tamano_kernel, tamano_kernel), np.uint8)
    
    if len(imagen.shape) == 3:
        resultado = cv2.dilate(imagen, kernel)
    else:
        resultado = cv2.dilate(imagen, kernel)
    
    return resultado


def filtro_minimo(imagen, tamano_kernel=5):
    """
    Aplica un filtro de mínimo (erosión).
    """
    kernel = np.ones((tamano_kernel, tamano_kernel), np.uint8)
    
    if len(imagen.shape) == 3:
        resultado = cv2.erode(imagen, kernel)
    else:
        resultado = cv2.erode(imagen, kernel)
    
    return resultado


# ========================= FILTROS OPCIONALES =========================

def filtro_mediana_adaptativa(imagen, tamano_max=7):
    """
    Aplica un filtro de mediana adaptativa.
    """
    resultado = imagen.copy()
    
    if len(imagen.shape) == 3:
        # Convertir a escala de grises para el procesamiento
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen
    
    for tamano in range(3, tamano_max + 1, 2):
        mediana = cv2.medianBlur(gris, tamano)
        # Aplicar donde sea necesario
        mascara = np.abs(gris.astype(np.float32) - mediana.astype(np.float32)) > 50
        if len(imagen.shape) == 3:
            for i in range(3):
                resultado[:, :, i][mascara] = cv2.medianBlur(imagen[:, :, i], tamano)[mascara]
        else:
            resultado[mascara] = mediana[mascara]
    
    return resultado


def filtro_contraharmonic_mean(imagen, tamano_kernel=5, Q=1.5):
    """
    Aplica un filtro de media contraarmónica.
    Q > 0: elimina ruido pimienta
    Q < 0: elimina ruido sal
    """
    if len(imagen.shape) == 3:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen
    
    imagen_float = imagen_gris.astype(np.float32)
    
    # Calcular numerador y denominador
    numerador = ndimage.uniform_filter(np.power(imagen_float, Q + 1), size=tamano_kernel)
    denominador = ndimage.uniform_filter(np.power(imagen_float, Q), size=tamano_kernel)
    
    # Evitar división por cero
    denominador[denominador == 0] = 1e-10
    
    resultado = numerador / denominador
    resultado = np.clip(resultado, 0, 255).astype(np.uint8)
    
    return resultado


def filtro_mediana_ponderada(imagen, tamano_kernel=5):
    """
    Aplica un filtro de mediana ponderada.
    Da más peso a los píxeles centrales.
    """
    if len(imagen.shape) == 3:
        resultado = np.zeros_like(imagen)
        for i in range(imagen.shape[2]):
            resultado[:, :, i] = _aplicar_mediana_ponderada_canal(imagen[:, :, i], tamano_kernel)
        return resultado
    else:
        return _aplicar_mediana_ponderada_canal(imagen, tamano_kernel)


def _aplicar_mediana_ponderada_canal(canal, tamano_kernel):
    """
    Aplica el filtro de mediana ponderada a un solo canal.
    """
    resultado = np.zeros_like(canal)
    pad = tamano_kernel // 2
    canal_pad = np.pad(canal, pad, mode='reflect')
    
    # Crear pesos (más peso al centro)
    centro = tamano_kernel // 2
    pesos = np.zeros((tamano_kernel, tamano_kernel))
    for i in range(tamano_kernel):
        for j in range(tamano_kernel):
            dist = abs(i - centro) + abs(j - centro)
            pesos[i, j] = tamano_kernel - dist
    
    for i in range(canal.shape[0]):
        for j in range(canal.shape[1]):
            ventana = canal_pad[i:i+tamano_kernel, j:j+tamano_kernel]
            # Crear lista ponderada
            valores_ponderados = []
            for x in range(tamano_kernel):
                for y in range(tamano_kernel):
                    valores_ponderados.extend([ventana[x, y]] * int(pesos[x, y]))
            
            # Calcular mediana ponderada
            mediana = np.median(valores_ponderados)
            resultado[i, j] = int(mediana)
    
    return resultado
