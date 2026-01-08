"""
Técnicas de Umbralización para Segmentación de Imágenes
Implementa diferentes métodos de umbralización automática
"""

import numpy as np
import cv2
from typing import Tuple, List


def metodo_otsu(imagen: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Método de Otsu: Determina un umbral óptimo basado en minimizar la varianza intra-clase.
    
    Args:
        imagen: Imagen en escala de grises
    
    Returns:
        Tupla (umbral_optimo, imagen_segmentada)
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Usar el método de Otsu de OpenCV
    umbral, img_segmentada = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return int(umbral), img_segmentada


def metodo_otsu_manual(imagen: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Implementación manual del método de Otsu.
    
    Args:
        imagen: Imagen en escala de grises
    
    Returns:
        Tupla (umbral_optimo, imagen_segmentada)
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Calcular histograma
    histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()
    histograma = histograma / histograma.sum()
    
    # Buscar el umbral óptimo
    mejor_varianza = 0
    umbral_optimo = 0
    
    for t in range(256):
        # Probabilidad de clase 0 (fondo)
        w0 = histograma[:t].sum()
        if w0 == 0:
            continue
        
        # Probabilidad de clase 1 (objeto)
        w1 = histograma[t:].sum()
        if w1 == 0:
            continue
        
        # Media de clase 0
        mu0 = (histograma[:t] * np.arange(t)).sum() / w0
        
        # Media de clase 1
        mu1 = (histograma[t:] * np.arange(t, 256)).sum() / w1
        
        # Varianza entre clases
        varianza_entre = w0 * w1 * (mu0 - mu1) ** 2
        
        if varianza_entre > mejor_varianza:
            mejor_varianza = varianza_entre
            umbral_optimo = t
    
    # Aplicar umbral
    _, img_segmentada = cv2.threshold(imagen, umbral_optimo, 255, cv2.THRESH_BINARY)
    
    return umbral_optimo, img_segmentada


def metodo_entropia_kapur(imagen: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Método de Entropía de Kapur: Selecciona un umbral basado en maximizar 
    la entropía de las regiones segmentadas.
    
    Args:
        imagen: Imagen en escala de grises
    
    Returns:
        Tupla (umbral_optimo, imagen_segmentada)
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Calcular histograma normalizado
    histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()
    histograma = histograma / histograma.sum()
    
    # Evitar log(0)
    histograma[histograma == 0] = 1e-10
    
    max_entropia = 0
    umbral_optimo = 0
    
    for t in range(1, 255):
        # Probabilidad acumulada para cada región
        P0 = histograma[:t].sum()
        P1 = histograma[t:].sum()
        
        if P0 == 0 or P1 == 0:
            continue
        
        # Entropía de la región 0 (fondo)
        H0 = -np.sum((histograma[:t] / P0) * np.log2(histograma[:t] / P0))
        
        # Entropía de la región 1 (objeto)
        H1 = -np.sum((histograma[t:] / P1) * np.log2(histograma[t:] / P1))
        
        # Entropía total
        entropia_total = H0 + H1
        
        if entropia_total > max_entropia:
            max_entropia = entropia_total
            umbral_optimo = t
    
    # Aplicar umbral
    _, img_segmentada = cv2.threshold(imagen, umbral_optimo, 255, cv2.THRESH_BINARY)
    
    return umbral_optimo, img_segmentada


def metodo_minimo_histograma(imagen: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Método basado en el mínimo de histogramas: Encuentra un umbral considerando
    el mínimo entre dos picos en el histograma.
    
    Args:
        imagen: Imagen en escala de grises
    
    Returns:
        Tupla (umbral_optimo, imagen_segmentada)
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Calcular histograma
    histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()
    
    # Suavizar el histograma para eliminar ruido
    histograma_suavizado = cv2.GaussianBlur(histograma.reshape(-1, 1), (0, 0), sigmaX=3).flatten()
    
    # Encontrar picos en el histograma
    from scipy.signal import find_peaks
    picos, propiedades = find_peaks(histograma_suavizado, distance=20, prominence=50)
    
    if len(picos) < 2:
        # Si no hay suficientes picos, usar Otsu
        return metodo_otsu(imagen)
    
    # Ordenar picos por prominencia
    indices_ordenados = np.argsort(propiedades['prominences'])[::-1]
    picos_principales = picos[indices_ordenados[:2]]
    picos_principales.sort()
    
    # Buscar el mínimo entre los dos picos principales
    inicio = picos_principales[0]
    fin = picos_principales[1]
    umbral_optimo = inicio + np.argmin(histograma_suavizado[inicio:fin])
    
    # Aplicar umbral
    _, img_segmentada = cv2.threshold(imagen, umbral_optimo, 255, cv2.THRESH_BINARY)
    
    return int(umbral_optimo), img_segmentada


def metodo_media(imagen: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Método usando la media: Utiliza la media de la imagen como umbral.
    
    Args:
        imagen: Imagen en escala de grises
    
    Returns:
        Tupla (umbral, imagen_segmentada)
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Calcular la media de la imagen
    umbral = int(np.mean(imagen))
    
    # Aplicar umbral
    _, img_segmentada = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)
    
    return umbral, img_segmentada


def metodo_multiumbral(imagen: np.ndarray, num_umbrales: int = 2) -> Tuple[List[int], np.ndarray]:
    """
    Método de multiumbralización: Segmenta la imagen en múltiples regiones.
    
    Args:
        imagen: Imagen en escala de grises
        num_umbrales: Número de umbrales a calcular
    
    Returns:
        Tupla (lista_umbrales, imagen_segmentada)
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Calcular histograma
    histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()
    histograma = histograma / histograma.sum()
    
    # Usar el método de Otsu iterativo
    umbrales = []
    img_segmentada = np.zeros_like(imagen)
    
    # Método mejorado usando el método de Otsu recursivo
    # Encontrar umbrales óptimos dividiendo el histograma
    def encontrar_umbral_otsu_en_rango(hist, inicio, fin):
        """Encuentra el umbral de Otsu en un rango específico del histograma"""
        hist_region = hist[inicio:fin]
        if hist_region.sum() == 0:
            return (inicio + fin) // 2
        
        suma_total = sum(i * hist_region[i] for i in range(len(hist_region)))
        suma_fondo = 0
        peso_fondo = 0
        varianza_max = 0
        umbral_optimo = 0
        
        for t in range(len(hist_region)):
            peso_fondo += hist_region[t]
            if peso_fondo == 0:
                continue
            
            peso_objeto = hist_region.sum() - peso_fondo
            if peso_objeto == 0:
                break
            
            suma_fondo += t * hist_region[t]
            media_fondo = suma_fondo / peso_fondo
            media_objeto = (suma_total - suma_fondo) / peso_objeto
            
            varianza_entre = peso_fondo * peso_objeto * (media_fondo - media_objeto) ** 2
            
            if varianza_entre > varianza_max:
                varianza_max = varianza_entre
                umbral_optimo = t
        
        return inicio + umbral_optimo
    
    # Dividir recursivamente usando Otsu
    umbrales = []
    rangos = [(0, 256)]
    
    for _ in range(num_umbrales):
        # Encontrar el mejor rango para dividir
        mejor_rango = max(rangos, key=lambda r: sum(histograma[r[0]:r[1]]))
        rangos.remove(mejor_rango)
        
        # Encontrar umbral óptimo en ese rango
        umbral = encontrar_umbral_otsu_en_rango(histograma, mejor_rango[0], mejor_rango[1])
        umbrales.append(umbral)
        
        # Dividir el rango en dos
        rangos.append((mejor_rango[0], umbral))
        rangos.append((umbral, mejor_rango[1]))
    
    umbrales = sorted(umbrales)
    
    # Crear imagen segmentada con niveles de gris
    img_segmentada = np.zeros_like(imagen)
    nivel_actual = 0
    
    for i, umbral in enumerate(umbrales):
        if i == 0:
            mascara = imagen <= umbral
        else:
            mascara = (imagen > umbrales[i-1]) & (imagen <= umbral)
        img_segmentada[mascara] = int(255 * (i + 1) / (num_umbrales + 1))
    
    # Última región
    mascara = imagen > umbrales[-1]
    img_segmentada[mascara] = 255
    
    return umbrales, img_segmentada


def umbral_por_banda(imagen: np.ndarray, umbral_min: int, umbral_max: int) -> np.ndarray:
    """
    Umbralización por banda: Segmenta píxeles dentro de un rango específico.
    
    Args:
        imagen: Imagen en escala de grises
        umbral_min: Umbral inferior
        umbral_max: Umbral superior
    
    Returns:
        Imagen segmentada (binaria)
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Crear máscara para píxeles dentro del rango
    mascara = cv2.inRange(imagen, umbral_min, umbral_max)
    
    return mascara


def umbral_adaptativo_media(imagen: np.ndarray, tamano_bloque: int = 11, constante: int = 2) -> np.ndarray:
    """
    Umbralización adaptativa usando la media local.
    
    Args:
        imagen: Imagen en escala de grises
        tamano_bloque: Tamaño del bloque para calcular la media local
        constante: Constante a restar de la media
    
    Returns:
        Imagen segmentada
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    img_segmentada = cv2.adaptiveThreshold(
        imagen, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, tamano_bloque, constante
    )
    
    return img_segmentada


def umbral_adaptativo_gaussiano(imagen: np.ndarray, tamano_bloque: int = 11, constante: int = 2) -> np.ndarray:
    """
    Umbralización adaptativa usando media ponderada gaussiana.
    
    Args:
        imagen: Imagen en escala de grises
        tamano_bloque: Tamaño del bloque para calcular la media local
        constante: Constante a restar de la media
    
    Returns:
        Imagen segmentada
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    img_segmentada = cv2.adaptiveThreshold(
        imagen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, tamano_bloque, constante
    )
    
    return img_segmentada
