"""
Detección y Análisis de Objetos Binarizados
Funciones para detectar contornos y calcular propiedades geométricas
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict


def detectar_objetos_binarizados(imagen_binaria: np.ndarray) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
    """
    Detecta objetos en una imagen binarizada y calcula sus propiedades geométricas.
    
    Args:
        imagen_binaria: Imagen binarizada (blanco y negro)
    
    Returns:
        Tupla que contiene:
        - Lista de contornos detectados
        - Lista de diccionarios con propiedades de cada objeto (área, perímetro, centro)
    """
    # Asegurar que la imagen esté en formato correcto
    if len(imagen_binaria.shape) == 3:
        imagen_binaria = cv2.cvtColor(imagen_binaria, cv2.COLOR_BGR2GRAY)
    
    # Asegurar que la imagen sea binaria (0 o 255)
    _, imagen_binaria = cv2.threshold(imagen_binaria, 127, 255, cv2.THRESH_BINARY)
    
    # Detectar contornos
    contornos, jerarquia = cv2.findContours(
        imagen_binaria, 
        cv2.RETR_EXTERNAL,  # Solo contornos externos
        cv2.CHAIN_APPROX_SIMPLE  # Comprime segmentos horizontales, verticales y diagonales
    )
    
    # Calcular propiedades de cada objeto
    propiedades = []
    
    for i, contorno in enumerate(contornos):
        # Calcular área
        area = cv2.contourArea(contorno)
        
        # Calcular perímetro
        perimetro = cv2.arcLength(contorno, True)
        
        # Calcular centroide
        momentos = cv2.moments(contorno)
        if momentos['m00'] != 0:
            cx = int(momentos['m10'] / momentos['m00'])
            cy = int(momentos['m01'] / momentos['m00'])
        else:
            cx, cy = 0, 0
        
        # Calcular rectángulo delimitador
        x, y, w, h = cv2.boundingRect(contorno)
        
        # Calcular circularidad (4π * área / perímetro²)
        if perimetro > 0:
            circularidad = (4 * np.pi * area) / (perimetro ** 2)
        else:
            circularidad = 0
        
        propiedades.append({
            'id': i + 1,
            'area': area,
            'perimetro': perimetro,
            'centroide': (cx, cy),
            'rectangulo': (x, y, w, h),
            'circularidad': circularidad,
            'num_vertices': len(contorno)
        })
    
    return contornos, propiedades


def dibujar_contornos_con_info(imagen_original: np.ndarray, 
                                contornos: List[np.ndarray], 
                                propiedades: List[Dict[str, float]],
                                mostrar_info: bool = True) -> np.ndarray:
    """
    Dibuja los contornos detectados sobre la imagen original con información de propiedades.
    
    Args:
        imagen_original: Imagen original (puede ser color o escala de grises)
        contornos: Lista de contornos detectados
        propiedades: Lista de propiedades de cada objeto
        mostrar_info: Si es True, muestra información textual sobre cada objeto
    
    Returns:
        Imagen con los contornos y la información dibujada
    """
    # Crear una copia de la imagen
    if len(imagen_original.shape) == 2:
        imagen_resultado = cv2.cvtColor(imagen_original, cv2.COLOR_GRAY2BGR)
    else:
        imagen_resultado = imagen_original.copy()
    
    # Colores para diferentes objetos
    colores = [
        (0, 255, 0),    # Verde
        (255, 0, 0),    # Azul
        (0, 0, 255),    # Rojo
        (255, 255, 0),  # Cian
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Amarillo
    ]
    
    # Dibujar cada contorno
    for i, (contorno, prop) in enumerate(zip(contornos, propiedades)):
        color = colores[i % len(colores)]
        
        # Dibujar el contorno
        cv2.drawContours(imagen_resultado, [contorno], -1, color, 2)
        
        if mostrar_info:
            # Dibujar el centroide
            cx, cy = prop['centroide']
            cv2.circle(imagen_resultado, (cx, cy), 5, color, -1)
            
            # Dibujar rectángulo delimitador
            x, y, w, h = prop['rectangulo']
            cv2.rectangle(imagen_resultado, (x, y), (x + w, y + h), color, 1)
            
            # Añadir texto con información
            texto_area = f"ID: {prop['id']}"
            texto_info = f"A: {prop['area']:.0f}"
            texto_per = f"P: {prop['perimetro']:.1f}"
            
            # Posición del texto (encima del objeto)
            pos_y = y - 10 if y > 40 else y + h + 20
            
            cv2.putText(imagen_resultado, texto_area, (x, pos_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(imagen_resultado, texto_info, (x, pos_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(imagen_resultado, texto_per, (x, pos_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return imagen_resultado


def filtrar_objetos_por_area(contornos: List[np.ndarray], 
                             propiedades: List[Dict[str, float]],
                             area_min: float = 0,
                             area_max: float = float('inf')) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
    """
    Filtra objetos basándose en el área.
    
    Args:
        contornos: Lista de contornos detectados
        propiedades: Lista de propiedades de cada objeto
        area_min: Área mínima para conservar el objeto
        area_max: Área máxima para conservar el objeto
    
    Returns:
        Tupla con contornos y propiedades filtrados
    """
    contornos_filtrados = []
    propiedades_filtradas = []
    
    for contorno, prop in zip(contornos, propiedades):
        if area_min <= prop['area'] <= area_max:
            contornos_filtrados.append(contorno)
            propiedades_filtradas.append(prop)
    
    return contornos_filtrados, propiedades_filtradas


def obtener_estadisticas_objetos(propiedades: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calcula estadísticas generales de todos los objetos detectados.
    
    Args:
        propiedades: Lista de propiedades de cada objeto
    
    Returns:
        Diccionario con estadísticas (promedio, mínimo, máximo, etc.)
    """
    if not propiedades:
        return {
            'num_objetos': 0,
            'area_total': 0,
            'area_promedio': 0,
            'area_min': 0,
            'area_max': 0,
            'perimetro_promedio': 0,
            'perimetro_min': 0,
            'perimetro_max': 0
        }
    
    areas = [p['area'] for p in propiedades]
    perimetros = [p['perimetro'] for p in propiedades]
    
    estadisticas = {
        'num_objetos': len(propiedades),
        'area_total': sum(areas),
        'area_promedio': np.mean(areas),
        'area_min': min(areas),
        'area_max': max(areas),
        'area_std': np.std(areas),
        'perimetro_promedio': np.mean(perimetros),
        'perimetro_min': min(perimetros),
        'perimetro_max': max(perimetros),
        'perimetro_std': np.std(perimetros)
    }
    
    return estadisticas


def crear_imagen_con_etiquetas(imagen_binaria: np.ndarray, 
                               contornos: List[np.ndarray]) -> np.ndarray:
    """
    Crea una imagen donde cada objeto tiene un valor de gris diferente (etiquetado).
    
    Args:
        imagen_binaria: Imagen binarizada original
        contornos: Lista de contornos detectados
    
    Returns:
        Imagen etiquetada donde cada objeto tiene un valor único
    """
    # Crear imagen de etiquetas
    imagen_etiquetas = np.zeros_like(imagen_binaria, dtype=np.uint8)
    
    # Asignar una etiqueta única a cada objeto
    for i, contorno in enumerate(contornos):
        # El valor de la etiqueta será proporcional al índice
        valor_etiqueta = int((i + 1) * (255 / max(len(contornos), 1)))
        cv2.drawContours(imagen_etiquetas, [contorno], -1, valor_etiqueta, -1)  # -1 para rellenar
    
    return imagen_etiquetas
