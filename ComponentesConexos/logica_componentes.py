"""
Lógica de Componentes Conexos
Implementa algoritmos de conectividad 4 y 8 para análisis de objetos en imágenes binarias
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List


def binarizar_imagen(imagen, umbral=127):
    """
    Binariza una imagen si no lo está
    
    Args:
        imagen: Imagen en escala de grises
        umbral: Umbral para binarización
    
    Returns:
        Imagen binaria
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Verificar si ya es binaria
    valores_unicos = np.unique(imagen)
    if len(valores_unicos) <= 2:
        return imagen
    
    # Binarizar
    _, img_binaria = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)
    return img_binaria


def componentes_conexos_4(imagen_binaria):
    """
    Encuentra componentes conexos usando conectividad de 4 vecinos
    (arriba, abajo, izquierda, derecha)
    
    Args:
        imagen_binaria: Imagen binaria
    
    Returns:
        Tupla (num_objetos, imagen_etiquetada, estadisticas)
    """
    # Asegurar que es binaria
    imagen_binaria = binarizar_imagen(imagen_binaria)
    
    # Aplicar algoritmo de componentes conexos con conectividad 4
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        imagen_binaria, 
        connectivity=4
    )
    
    # El primer componente (0) es el fondo, así que restamos 1
    num_objetos = num_labels - 1
    
    # Crear imagen coloreada para visualización
    imagen_coloreada = colorear_componentes(labels, num_labels)
    
    # Preparar estadísticas
    estadisticas = procesar_estadisticas(stats, centroids, num_labels)
    
    return num_objetos, labels, imagen_coloreada, estadisticas


def componentes_conexos_8(imagen_binaria):
    """
    Encuentra componentes conexos usando conectividad de 8 vecinos
    (arriba, abajo, izquierda, derecha y 4 diagonales)
    
    Args:
        imagen_binaria: Imagen binaria
    
    Returns:
        Tupla (num_objetos, imagen_etiquetada, estadisticas)
    """
    # Asegurar que es binaria
    imagen_binaria = binarizar_imagen(imagen_binaria)
    
    # Aplicar algoritmo de componentes conexos con conectividad 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        imagen_binaria, 
        connectivity=8
    )
    
    # El primer componente (0) es el fondo, así que restamos 1
    num_objetos = num_labels - 1
    
    # Crear imagen coloreada para visualización
    imagen_coloreada = colorear_componentes(labels, num_labels)
    
    # Preparar estadísticas
    estadisticas = procesar_estadisticas(stats, centroids, num_labels)
    
    return num_objetos, labels, imagen_coloreada, estadisticas


def colorear_componentes(labels, num_labels):
    """
    Crea una imagen coloreada donde cada componente tiene un color diferente
    
    Args:
        labels: Matriz de etiquetas
        num_labels: Número de etiquetas
    
    Returns:
        Imagen RGB con componentes coloreados
    """
    # Crear una paleta de colores
    np.random.seed(42)  # Para colores consistentes
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Fondo negro
    
    # Mapear etiquetas a colores
    imagen_coloreada = colors[labels]
    
    return imagen_coloreada


def procesar_estadisticas(stats, centroids, num_labels):
    """
    Procesa las estadísticas de cada componente
    
    Args:
        stats: Estadísticas de cv2.connectedComponentsWithStats
        centroids: Centroides de cada componente
        num_labels: Número de etiquetas
    
    Returns:
        Lista de diccionarios con información de cada objeto
    """
    estadisticas = []
    
    # Saltar el índice 0 (fondo)
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        
        estadisticas.append({
            'id': i,
            'area': area,
            'x': x,
            'y': y,
            'ancho': w,
            'alto': h,
            'centroide': (int(cx), int(cy))
        })
    
    return estadisticas


def dibujar_componentes_con_info(imagen_original, labels, estadisticas):
    """
    Dibuja los componentes con información sobre cada uno
    
    Args:
        imagen_original: Imagen original
        labels: Matriz de etiquetas
        estadisticas: Lista de estadísticas de objetos
    
    Returns:
        Imagen con componentes marcados
    """
    # Convertir a BGR si es necesario
    if len(imagen_original.shape) == 2:
        imagen_resultado = cv2.cvtColor(imagen_original, cv2.COLOR_GRAY2BGR)
    else:
        imagen_resultado = imagen_original.copy()
    
    # Colores para dibujar
    colores = [
        (0, 255, 0),    # Verde
        (255, 0, 0),    # Azul
        (0, 0, 255),    # Rojo
        (255, 255, 0),  # Cian
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Amarillo
    ]
    
    for i, obj in enumerate(estadisticas):
        color = colores[i % len(colores)]
        
        # Dibujar rectángulo delimitador
        x, y, w, h = obj['x'], obj['y'], obj['ancho'], obj['alto']
        cv2.rectangle(imagen_resultado, (x, y), (x + w, y + h), color, 2)
        
        # Dibujar centroide
        cx, cy = obj['centroide']
        cv2.circle(imagen_resultado, (cx, cy), 5, color, -1)
        
        # Añadir texto con ID y área
        texto = f"#{obj['id']} A:{obj['area']}"
        cv2.putText(imagen_resultado, texto, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return imagen_resultado


def comparar_conectividades(imagen_binaria):
    """
    Compara los resultados de conectividad 4 y 8
    
    Args:
        imagen_binaria: Imagen binaria
    
    Returns:
        Diccionario con resultados de ambas conectividades
    """
    # Aplicar ambos métodos
    num_obj_4, labels_4, img_col_4, stats_4 = componentes_conexos_4(imagen_binaria)
    num_obj_8, labels_8, img_col_8, stats_8 = componentes_conexos_8(imagen_binaria)
    
    return {
        'conectividad_4': {
            'num_objetos': num_obj_4,
            'labels': labels_4,
            'imagen_coloreada': img_col_4,
            'estadisticas': stats_4
        },
        'conectividad_8': {
            'num_objetos': num_obj_8,
            'labels': labels_8,
            'imagen_coloreada': img_col_8,
            'estadisticas': stats_8
        },
        'diferencia': num_obj_4 - num_obj_8
    }


def filtrar_por_area(estadisticas, area_min=0, area_max=float('inf')):
    """
    Filtra componentes por área
    
    Args:
        estadisticas: Lista de estadísticas
        area_min: Área mínima
        area_max: Área máxima
    
    Returns:
        Lista filtrada de estadísticas
    """
    return [obj for obj in estadisticas if area_min <= obj['area'] <= area_max]


def obtener_resumen_estadistico(estadisticas):
    """
    Obtiene un resumen estadístico de los componentes
    
    Args:
        estadisticas: Lista de estadísticas
    
    Returns:
        Diccionario con resumen
    """
    if not estadisticas:
        return {
            'num_objetos': 0,
            'area_total': 0,
            'area_promedio': 0,
            'area_min': 0,
            'area_max': 0,
            'area_std': 0
        }
    
    areas = [obj['area'] for obj in estadisticas]
    
    return {
        'num_objetos': len(estadisticas),
        'area_total': sum(areas),
        'area_promedio': np.mean(areas),
        'area_min': min(areas),
        'area_max': max(areas),
        'area_std': np.std(areas)
    }
