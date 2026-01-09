"""
Módulo de Procesamiento Básico de Imágenes Digitales

Este módulo proporciona funcionalidades para:
- Conversión de imágenes RGB a escala de grises
- Binarización con umbral fijo y automático (Otsu)
- Visualización de histogramas de intensidad
"""

from .procesamiento_basico import (
    rgb_a_grises,
    binarizacion_umbral_fijo,
    binarizacion_umbral_otsu,
    calcular_histograma
)

__all__ = [
    'rgb_a_grises',
    'binarizacion_umbral_fijo',
    'binarizacion_umbral_otsu',
    'calcular_histograma'
]
