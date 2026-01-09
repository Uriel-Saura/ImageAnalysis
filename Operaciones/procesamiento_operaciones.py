"""
Módulo de operaciones sobre imágenes
Incluye operaciones con escalares, operaciones lógicas y operaciones aritméticas
"""

import cv2
import numpy as np


# ==================== OPERACIONES CON ESCALARES ====================

def suma_escalar(imagen, valor):
    """
    Suma un valor escalar a todos los píxeles de la imagen
    
    Args:
        imagen: Imagen de entrada
        valor: Valor escalar a sumar
    
    Returns:
        Imagen resultante
    """
    # Convertir a grises si es necesario
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Sumar y saturar (clip) entre 0 y 255
    resultado = np.clip(imagen.astype(np.int16) + valor, 0, 255).astype(np.uint8)
    return resultado


def resta_escalar(imagen, valor):
    """
    Resta un valor escalar a todos los píxeles de la imagen
    
    Args:
        imagen: Imagen de entrada
        valor: Valor escalar a restar
    
    Returns:
        Imagen resultante
    """
    # Convertir a grises si es necesario
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Restar y saturar entre 0 y 255
    resultado = np.clip(imagen.astype(np.int16) - valor, 0, 255).astype(np.uint8)
    return resultado


def multiplicacion_escalar(imagen, valor):
    """
    Multiplica todos los píxeles de la imagen por un valor escalar
    
    Args:
        imagen: Imagen de entrada
        valor: Valor escalar multiplicador
    
    Returns:
        Imagen resultante
    """
    # Convertir a grises si es necesario
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Multiplicar y saturar entre 0 y 255
    resultado = np.clip(imagen.astype(np.float32) * valor, 0, 255).astype(np.uint8)
    return resultado


def division_escalar(imagen, valor):
    """
    Divide todos los píxeles de la imagen por un valor escalar
    
    Args:
        imagen: Imagen de entrada
        valor: Valor escalar divisor
    
    Returns:
        Imagen resultante
    """
    # Convertir a grises si es necesario
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    if valor == 0:
        return imagen.copy()
    
    # Dividir y saturar entre 0 y 255
    resultado = np.clip(imagen.astype(np.float32) / valor, 0, 255).astype(np.uint8)
    return resultado


# ==================== OPERACIONES LÓGICAS ====================

def operacion_and(imagen1, imagen2):
    """
    Operación lógica AND bit a bit entre dos imágenes
    
    Args:
        imagen1: Primera imagen
        imagen2: Segunda imagen
    
    Returns:
        Imagen resultante de la operación AND
    """
    # Convertir ambas a grises
    if len(imagen1.shape) == 3:
        imagen1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
    if len(imagen2.shape) == 3:
        imagen2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
    
    # Asegurar mismo tamaño
    if imagen1.shape != imagen2.shape:
        imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
    
    return cv2.bitwise_and(imagen1, imagen2)


def operacion_or(imagen1, imagen2):
    """
    Operación lógica OR bit a bit entre dos imágenes
    
    Args:
        imagen1: Primera imagen
        imagen2: Segunda imagen
    
    Returns:
        Imagen resultante de la operación OR
    """
    # Convertir ambas a grises
    if len(imagen1.shape) == 3:
        imagen1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
    if len(imagen2.shape) == 3:
        imagen2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
    
    # Asegurar mismo tamaño
    if imagen1.shape != imagen2.shape:
        imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
    
    return cv2.bitwise_or(imagen1, imagen2)


def operacion_xor(imagen1, imagen2):
    """
    Operación lógica XOR bit a bit entre dos imágenes
    
    Args:
        imagen1: Primera imagen
        imagen2: Segunda imagen
    
    Returns:
        Imagen resultante de la operación XOR
    """
    # Convertir ambas a grises
    if len(imagen1.shape) == 3:
        imagen1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
    if len(imagen2.shape) == 3:
        imagen2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
    
    # Asegurar mismo tamaño
    if imagen1.shape != imagen2.shape:
        imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
    
    return cv2.bitwise_xor(imagen1, imagen2)


def operacion_not(imagen):
    """
    Operación lógica NOT (inversión) de una imagen
    
    Args:
        imagen: Imagen de entrada
    
    Returns:
        Imagen resultante de la operación NOT
    """
    # Convertir a grises si es necesario
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    return cv2.bitwise_not(imagen)


# ==================== OPERACIONES ARITMÉTICAS ====================

def suma_imagenes(imagen1, imagen2, peso1=0.5, peso2=0.5):
    """
    Suma aritmética de dos imágenes con pesos opcionales
    
    Args:
        imagen1: Primera imagen
        imagen2: Segunda imagen
        peso1: Peso para imagen1 (default 0.5)
        peso2: Peso para imagen2 (default 0.5)
    
    Returns:
        Imagen resultante de la suma ponderada
    """
    # Convertir ambas a grises
    if len(imagen1.shape) == 3:
        imagen1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
    if len(imagen2.shape) == 3:
        imagen2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
    
    # Asegurar mismo tamaño
    if imagen1.shape != imagen2.shape:
        imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
    
    # Suma ponderada
    resultado = cv2.addWeighted(imagen1, peso1, imagen2, peso2, 0)
    return resultado


def resta_imagenes(imagen1, imagen2):
    """
    Resta aritmética de dos imágenes (imagen1 - imagen2)
    
    Args:
        imagen1: Primera imagen (minuendo)
        imagen2: Segunda imagen (sustraendo)
    
    Returns:
        Imagen resultante de la resta
    """
    # Convertir ambas a grises
    if len(imagen1.shape) == 3:
        imagen1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
    if len(imagen2.shape) == 3:
        imagen2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
    
    # Asegurar mismo tamaño
    if imagen1.shape != imagen2.shape:
        imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
    
    # Resta con saturación
    resultado = cv2.subtract(imagen1, imagen2)
    return resultado


def multiplicacion_imagenes(imagen1, imagen2):
    """
    Multiplicación aritmética de dos imágenes (píxel a píxel)
    
    Args:
        imagen1: Primera imagen
        imagen2: Segunda imagen
    
    Returns:
        Imagen resultante de la multiplicación
    """
    # Convertir ambas a grises
    if len(imagen1.shape) == 3:
        imagen1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
    if len(imagen2.shape) == 3:
        imagen2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
    
    # Asegurar mismo tamaño
    if imagen1.shape != imagen2.shape:
        imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
    
    # Multiplicar y normalizar
    resultado = cv2.multiply(imagen1, imagen2, scale=1.0/255.0)
    return resultado


def division_imagenes(imagen1, imagen2):
    """
    División aritmética de dos imágenes (imagen1 / imagen2)
    
    Args:
        imagen1: Primera imagen (dividendo)
        imagen2: Segunda imagen (divisor)
    
    Returns:
        Imagen resultante de la división
    """
    # Convertir ambas a grises
    if len(imagen1.shape) == 3:
        imagen1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
    if len(imagen2.shape) == 3:
        imagen2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
    
    # Asegurar mismo tamaño
    if imagen1.shape != imagen2.shape:
        imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
    
    # División con protección contra división por cero
    imagen2_safe = np.where(imagen2 == 0, 1, imagen2)
    resultado = np.clip((imagen1.astype(np.float32) / imagen2_safe.astype(np.float32)) * 255, 0, 255).astype(np.uint8)
    return resultado


def diferencia_absoluta(imagen1, imagen2):
    """
    Diferencia absoluta entre dos imágenes |imagen1 - imagen2|
    
    Args:
        imagen1: Primera imagen
        imagen2: Segunda imagen
    
    Returns:
        Imagen resultante de la diferencia absoluta
    """
    # Convertir ambas a grises
    if len(imagen1.shape) == 3:
        imagen1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
    if len(imagen2.shape) == 3:
        imagen2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
    
    # Asegurar mismo tamaño
    if imagen1.shape != imagen2.shape:
        imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
    
    return cv2.absdiff(imagen1, imagen2)
