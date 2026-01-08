"""
Inicialización del módulo de Segmentación
"""

__version__ = "1.0.0"
__author__ = "Sistema de Análisis de Imágenes"

# Importar las funciones principales para acceso directo
from .tecnicas_umbralizacion import (
    metodo_otsu,
    metodo_entropia_kapur,
    metodo_minimo_histograma,
    metodo_media,
    metodo_multiumbral,
    umbral_por_banda
)

from .tecnicas_ecualizacion import (
    ecualizacion_uniforme,
    ecualizacion_exponencial,
    ecualizacion_rayleigh,
    ecualizacion_hipercubica,
    ecualizacion_logaritmica_hiperbolica,
    ecualizacion_clahe
)

from .tecnicas_ajuste import (
    funcion_potencia,
    correccion_gamma,
    desplazamiento_histograma,
    contraccion_histograma,
    expansion_histograma
)

__all__ = [
    # Umbralización
    'metodo_otsu',
    'metodo_entropia_kapur',
    'metodo_minimo_histograma',
    'metodo_media',
    'metodo_multiumbral',
    'umbral_por_banda',
    # Ecualización
    'ecualizacion_uniforme',
    'ecualizacion_exponencial',
    'ecualizacion_rayleigh',
    'ecualizacion_hipercubica',
    'ecualizacion_logaritmica_hiperbolica',
    'ecualizacion_clahe',
    # Ajuste
    'funcion_potencia',
    'correccion_gamma',
    'desplazamiento_histograma',
    'contraccion_histograma',
    'expansion_histograma',
]
