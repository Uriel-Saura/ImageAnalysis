"""
Pipeline de Preprocesamiento para OCR
Pipeline: Grises → CLAHE → Umbralización Adaptativa → Filtro Mediana
"""

import numpy as np
import cv2
import sys
import os
from typing import Dict, List, Tuple

# Importar módulos del proyecto
directorio_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if directorio_raiz not in sys.path:
    sys.path.insert(0, directorio_raiz)

# Importar técnicas de los módulos existentes
from ImagenDigital.procesamiento_basico import rgb_a_grises
from AnalisisRuido.filtros_no_lineales import filtro_mediana


class PipelinePreprocesamientoOCR:
    """Pipeline simplificado de preprocesamiento de imágenes para OCR"""
    
    def __init__(self):
        """Inicializa el pipeline"""
        self.historial_pasos = []
        self.imagen_original = None
        self.imagen_procesada = None
    
    def procesar(self, imagen: np.ndarray, 
                mostrar_pasos: bool = False) -> Tuple[np.ndarray, List[Dict]]:
        """
        Procesa una imagen aplicando el pipeline optimizado para OCR:
        1. Convertir a escala de grises
        2. Limpieza de ruido inicial (Filtro Mediana kernel 5x5)
        3. Reducción de ruido preservando bordes (Bilateral)
        4. Mejora de contraste agresiva (CLAHE con parámetros optimizados)
        5. Umbralización adaptativa optimizada (GAUSSIAN con mejor separación)
        6. Operaciones morfológicas (cierre para conectar caracteres)
        7. Limpieza de ruido final agresiva (Filtro Mediana kernel 5x5)
        
        Args:
            imagen: Imagen de entrada
            mostrar_pasos: Si True, retorna imágenes intermedias
        
        Returns:
            Tupla (imagen_procesada, historial_pasos)
        """
        self.imagen_original = imagen.copy()
        self.historial_pasos = []
        img_actual = imagen.copy()
        
        # Paso 1: Convertir a escala de grises
        img_actual = self._convertir_grises(img_actual)
        self._registrar_paso("Conversión a escala de grises", img_actual)
        
        # Paso 2: Limpieza de ruido inicial con filtro de mediana
        img_actual = self._limpieza_ruido_inicial(img_actual)
        self._registrar_paso("Limpieza de ruido inicial (Mediana 5x5)", img_actual)
        
        # Paso 3: Reducción de ruido preservando bordes
        img_actual = self._reducir_ruido_bilateral(img_actual)
        self._registrar_paso("Reducción de ruido (Bilateral)", img_actual)
        
        # Paso 4: Mejora de contraste agresiva (CLAHE optimizado)
        img_actual = self._mejorar_contraste(img_actual)
        self._registrar_paso("Mejora de contraste (CLAHE mejorado)", img_actual)
        
        # Paso 5: Umbralización adaptativa optimizada
        img_actual = self._umbralizar_adaptativo(img_actual)
        self._registrar_paso("Umbralización adaptativa (GAUSSIAN optimizado)", img_actual)
        
        # Paso 6: Operaciones morfológicas para mejorar la estructura del texto
        img_actual = self._operacion_morfologica_cierre(img_actual)
        self._registrar_paso("Cierre morfológico (conectar caracteres)", img_actual)
        
        # Paso 7: Limpieza de ruido agresiva final con filtro de mediana
        img_actual = self._limpieza_ruido_final(img_actual)
        self._registrar_paso("Limpieza de ruido final (Mediana 5x5)", img_actual)
        
        self.imagen_procesada = img_actual
        return img_actual, self.historial_pasos
    
    def _convertir_grises(self, imagen: np.ndarray) -> np.ndarray:
        """Convierte la imagen a escala de grises"""
        if len(imagen.shape) == 3:
            return rgb_a_grises(imagen)
        return imagen
    
    def _limpieza_ruido_inicial(self, imagen: np.ndarray) -> np.ndarray:
        """
        Limpieza inicial de ruido con filtro de mediana 5x5
        Elimina ruido sal y pimienta antes de otros procesamientos
        """
        return cv2.medianBlur(imagen, 5)
    
    def _reducir_ruido_bilateral(self, imagen: np.ndarray) -> np.ndarray:
        """
        Aplica filtro bilateral suave para reducir ruido preservando bordes
        Parámetros suaves para no perder detalles del texto
        """
        return cv2.bilateralFilter(imagen, d=5, sigmaColor=50, sigmaSpace=50)
    
    def _mejorar_contraste(self, imagen: np.ndarray) -> np.ndarray:
        """
        Mejora el contraste usando CLAHE balanceado para OCR
        clipLimit moderado (2.5) = contraste mejorado sin distorsión
        tileGridSize (6,6) = adaptación local equilibrada
        """
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
        return clahe.apply(imagen)
    
    def _umbralizar_adaptativo(self, imagen: np.ndarray) -> np.ndarray:
        """
        Aplica umbralización adaptativa balanceada para OCR
        blockSize 13 = ventana balanceada
        C = 3 = ajuste moderado para separar texto de fondo
        """
        return cv2.adaptiveThreshold(
            imagen, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            13,  # Tamaño de bloque balanceado
            3    # Constante C moderada
        )
    
    def _operacion_morfologica_cierre(self, imagen: np.ndarray) -> np.ndarray:
        """
        Aplica operación de cierre morfológico muy ligero
        Kernel mínimo (1,1) para conectar pixeles sin distorsionar caracteres
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        return cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    def _limpieza_ruido_final(self, imagen: np.ndarray) -> np.ndarray:
        """
        Limpieza final agresiva de ruido con filtro de mediana 5x5
        Elimina ruido residual después de todos los procesamientos
        Deja la imagen limpia para una detección óptima
        """
        return cv2.medianBlur(imagen, 5)
    
    def _registrar_paso(self, nombre: str, imagen: np.ndarray):
        """Registra un paso del pipeline en el historial"""
        self.historial_pasos.append({
            'nombre': nombre,
            'imagen': imagen.copy()
        })
