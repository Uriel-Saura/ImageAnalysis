"""
Práctica 2 - Generación de Ruido y Aplicación de Filtros

Este paquete contiene todas las funcionalidades para la práctica 2,
incluyendo generación de ruido, filtros lineales y no lineales, y visualización.
"""

__version__ = '1.0.0'
__author__ = 'ImageAnalysis'

# Importaciones para facilitar el uso del paquete
from .generacion_ruido import (
    aplicar_ruido_sal_pimienta,
    aplicar_ruido_sal,
    aplicar_ruido_pimienta,
    aplicar_ruido_gaussiano,
    calcular_histograma
)

from .filtros_lineales import (
    # Paso altas
    filtro_sobel,
    filtro_prewitt,
    filtro_roberts,
    filtro_kirsch,
    filtro_canny,
    filtro_laplaciano_clasico,
    filtro_laplaciano_8_vecinos,
    filtro_laplaciano_horizontal,
    filtro_laplaciano_vertical,
    filtro_laplaciano_diagonal_principal,
    filtro_laplaciano_diagonal_secundaria,
    # Paso bajas
    filtro_promediador,
    filtro_promediador_pesado,
    filtro_gaussiano,
    filtro_bilateral
)

from .filtros_no_lineales import (
    filtro_mediana,
    filtro_moda,
    filtro_maximo,
    filtro_minimo,
    filtro_mediana_adaptativa,
    filtro_contraharmonic_mean,
    filtro_mediana_ponderada
)

__all__ = [
    # Ruido
    'aplicar_ruido_sal_pimienta',
    'aplicar_ruido_sal',
    'aplicar_ruido_pimienta',
    'aplicar_ruido_gaussiano',
    'calcular_histograma',
    # Filtros paso altas
    'filtro_sobel',
    'filtro_prewitt',
    'filtro_roberts',
    'filtro_kirsch',
    'filtro_canny',
    'filtro_laplaciano_clasico',
    'filtro_laplaciano_8_vecinos',
    'filtro_laplaciano_horizontal',
    'filtro_laplaciano_vertical',
    'filtro_laplaciano_diagonal_principal',
    'filtro_laplaciano_diagonal_secundaria',
    # Filtros paso bajas
    'filtro_promediador',
    'filtro_promediador_pesado',
    'filtro_gaussiano',
    'filtro_bilateral',
    # Filtros no lineales
    'filtro_mediana',
    'filtro_moda',
    'filtro_maximo',
    'filtro_minimo',
    'filtro_mediana_adaptativa',
    'filtro_contraharmonic_mean',
    'filtro_mediana_ponderada'
]
