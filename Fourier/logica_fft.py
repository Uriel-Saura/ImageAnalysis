# ===================================================================
# LÓGICA DE LA TRANSFORMADA DE FOURIER
# Funciones para cálculo y procesamiento de FFT
# ===================================================================

import numpy as np
import cv2


class TransformadaFourier:
    """Clase para manejar la Transformada de Fourier de una imagen"""
    
    def __init__(self):
        self.imagen = None
        self.fft = None
        self.fft_shift = None
        self.magnitud = None
        self.fase = None
    
    def cargar_imagen(self, imagen):
        """Carga una imagen y calcula la FFT automáticamente"""
        self.imagen = imagen
        self._calcular_fft()
    
    def _calcular_fft(self):
        """Calcula la Transformada de Fourier 2D"""
        if self.imagen is None:
            return
        
        # Convertir a float32 para mayor precisión
        imagen_float = np.float32(self.imagen)
        
        # Calcular FFT 2D
        self.fft = np.fft.fft2(imagen_float)
        
        # Desplazar componente DC al centro
        self.fft_shift = np.fft.fftshift(self.fft)
        
        # Calcular magnitud (escala logarítmica para mejor visualización)
        self.magnitud = np.log1p(np.abs(self.fft_shift))
        
        # Calcular fase
        self.fase = np.angle(self.fft_shift)
    
    def obtener_magnitud(self):
        """Retorna el espectro de magnitud"""
        return self.magnitud
    
    def obtener_fase(self):
        """Retorna el espectro de fase"""
        return self.fase
    
    def obtener_magnitud_normalizada(self):
        """Retorna magnitud normalizada a 8 bits para guardar"""
        if self.magnitud is None:
            return None
        magnitud_norm = cv2.normalize(self.magnitud, None, 0, 255, cv2.NORM_MINMAX)
        return magnitud_norm.astype(np.uint8)
    
    def obtener_fase_normalizada(self):
        """Retorna fase normalizada a 8 bits para guardar"""
        if self.fase is None:
            return None
        fase_norm = cv2.normalize(self.fase, None, 0, 255, cv2.NORM_MINMAX)
        return fase_norm.astype(np.uint8)
    
    def reconstruir_imagen(self):
        """Reconstruye la imagen desde la FFT (IFFT)"""
        if self.fft_shift is None:
            return None
        
        # Invertir el shift y aplicar IFFT
        fft_ishift = np.fft.ifftshift(self.fft_shift)
        imagen_reconstruida = np.fft.ifft2(fft_ishift)
        return np.abs(imagen_reconstruida)
    
    def obtener_info(self):
        """Retorna información sobre la FFT calculada"""
        if self.fft_shift is None:
            return None
        
        return {
            'tamaño': self.fft_shift.shape,
            'magnitud_min': self.magnitud.min(),
            'magnitud_max': self.magnitud.max(),
            'fase_min': self.fase.min(),
            'fase_max': self.fase.max()
        }
    
    def esta_calculada(self):
        """Verifica si la FFT ha sido calculada"""
        return self.fft_shift is not None
