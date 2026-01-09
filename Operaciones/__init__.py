"""
Módulo de Operaciones sobre Imágenes

Este módulo proporciona funcionalidades para:
- Operaciones con escalares: suma, resta, multiplicación, división
- Operaciones lógicas: AND, OR, XOR, NOT
- Operaciones aritméticas entre imágenes: suma, resta, multiplicación, división, diferencia absoluta
"""

from .procesamiento_operaciones import (
    suma_escalar, resta_escalar, multiplicacion_escalar, division_escalar,
    operacion_and, operacion_or, operacion_xor, operacion_not,
    suma_imagenes, resta_imagenes, multiplicacion_imagenes, 
    division_imagenes, diferencia_absoluta
)

__all__ = [
    'suma_escalar', 'resta_escalar', 'multiplicacion_escalar', 'division_escalar',
    'operacion_and', 'operacion_or', 'operacion_xor', 'operacion_not',
    'suma_imagenes', 'resta_imagenes', 'multiplicacion_imagenes', 
    'division_imagenes', 'diferencia_absoluta'
]
