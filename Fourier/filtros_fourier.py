# ===================================================================
# LÓGICA DE FILTROS EN EL DOMINIO DE FOURIER
# Implementa filtros pasa-bajas y pasa-altas
# ===================================================================

import numpy as np
import cv2


class FiltrosFourier:
    """Clase para aplicar filtros en el dominio de Fourier"""
    
    def __init__(self, fft_shift):
        """
        Inicializa el filtro con la FFT desplazada
        
        Args:
            fft_shift: FFT 2D con componente DC centrado
        """
        self.fft_shift = fft_shift
        self.filas, self.columnas = fft_shift.shape
        self.centro_fila = self.filas // 2
        self.centro_col = self.columnas // 2
    
    def _crear_mascara_distancia(self):
        """Crea una matriz con las distancias desde el centro"""
        y = np.arange(self.filas)
        x = np.arange(self.columnas)
        Y, X = np.meshgrid(y, x, indexing='ij')
        
        # Calcular distancia euclidiana desde el centro
        distancia = np.sqrt((Y - self.centro_fila)**2 + (X - self.centro_col)**2)
        return distancia
    
    # ===== FILTROS PASA-BAJAS =====
    
    def ideal_pasabajas(self, radio):
        """
        Filtro ideal pasa-bajas
        Deja pasar frecuencias dentro del radio, bloquea el resto
        
        Args:
            radio: Radio de corte del filtro
        """
        distancia = self._crear_mascara_distancia()
        mascara = np.zeros((self.filas, self.columnas))
        mascara[distancia <= radio] = 1
        
        # Aplicar filtro
        fft_filtrada = self.fft_shift * mascara
        
        # Reconstruir imagen
        fft_ishift = np.fft.ifftshift(fft_filtrada)
        imagen_filtrada = np.fft.ifft2(fft_ishift)
        return np.abs(imagen_filtrada), mascara
    
    def gaussiano_pasabajas(self, sigma):
        """
        Filtro Gaussiano pasa-bajas
        Transición suave entre frecuencias pasadas y bloqueadas
        
        Args:
            sigma: Desviación estándar (controla el ancho del filtro)
        """
        distancia = self._crear_mascara_distancia()
        
        # Crear máscara Gaussiana
        mascara = np.exp(-(distancia**2) / (2 * (sigma**2)))
        
        # Aplicar filtro
        fft_filtrada = self.fft_shift * mascara
        
        # Reconstruir imagen
        fft_ishift = np.fft.ifftshift(fft_filtrada)
        imagen_filtrada = np.fft.ifft2(fft_ishift)
        return np.abs(imagen_filtrada), mascara
    
    def butterworth_pasabajas(self, radio, orden=2):
        """
        Filtro Butterworth pasa-bajas
        Transición controlada por el orden del filtro
        
        Args:
            radio: Radio de corte (frecuencia de corte)
            orden: Orden del filtro (mayor = más pronunciado)
        """
        distancia = self._crear_mascara_distancia()
        
        # Evitar división por cero
        distancia[distancia == 0] = 0.01
        
        # Crear máscara Butterworth
        mascara = 1 / (1 + (distancia / radio)**(2 * orden))
        
        # Aplicar filtro
        fft_filtrada = self.fft_shift * mascara
        
        # Reconstruir imagen
        fft_ishift = np.fft.ifftshift(fft_filtrada)
        imagen_filtrada = np.fft.ifft2(fft_ishift)
        return np.abs(imagen_filtrada), mascara
    
    # ===== FILTROS PASA-ALTAS =====
    
    def ideal_pasaaltas(self, radio):
        """
        Filtro ideal pasa-altas
        Bloquea frecuencias dentro del radio, deja pasar el resto
        
        Args:
            radio: Radio de corte del filtro
        """
        distancia = self._crear_mascara_distancia()
        mascara = np.ones((self.filas, self.columnas))
        mascara[distancia <= radio] = 0
        
        # Aplicar filtro
        fft_filtrada = self.fft_shift * mascara
        
        # Reconstruir imagen
        fft_ishift = np.fft.ifftshift(fft_filtrada)
        imagen_filtrada = np.fft.ifft2(fft_ishift)
        return np.abs(imagen_filtrada), mascara
    
    def gaussiano_pasaaltas(self, sigma):
        """
        Filtro Gaussiano pasa-altas
        Transición suave, bloquea bajas frecuencias
        
        Args:
            sigma: Desviación estándar (controla el ancho del filtro)
        """
        distancia = self._crear_mascara_distancia()
        
        # Crear máscara Gaussiana invertida
        mascara = 1 - np.exp(-(distancia**2) / (2 * (sigma**2)))
        
        # Aplicar filtro
        fft_filtrada = self.fft_shift * mascara
        
        # Reconstruir imagen
        fft_ishift = np.fft.ifftshift(fft_filtrada)
        imagen_filtrada = np.fft.ifft2(fft_ishift)
        return np.abs(imagen_filtrada), mascara
    
    def butterworth_pasaaltas(self, radio, orden=2):
        """
        Filtro Butterworth pasa-altas
        Transición controlada, bloquea bajas frecuencias
        
        Args:
            radio: Radio de corte (frecuencia de corte)
            orden: Orden del filtro (mayor = más pronunciado)
        """
        distancia = self._crear_mascara_distancia()
        
        # Evitar división por cero
        distancia[distancia == 0] = 0.01
        
        # Crear máscara Butterworth invertida
        mascara = 1 / (1 + (radio / distancia)**(2 * orden))
        
        # Aplicar filtro
        fft_filtrada = self.fft_shift * mascara
        
        # Reconstruir imagen
        fft_ishift = np.fft.ifftshift(fft_filtrada)
        imagen_filtrada = np.fft.ifft2(fft_ishift)
        return np.abs(imagen_filtrada), mascara


def aplicar_filtro(imagen, tipo_filtro, parametro1, parametro2=None):
    """
    Función auxiliar para aplicar un filtro a una imagen
    
    Args:
        imagen: Imagen en escala de grises
        tipo_filtro: Nombre del filtro ('ideal_pb', 'gaussiano_pb', etc.)
        parametro1: Radio o sigma
        parametro2: Orden (solo para Butterworth)
    
    Returns:
        imagen_filtrada, mascara
    """
    # Convertir a float32
    imagen_float = np.float32(imagen)
    
    # Calcular FFT
    fft = np.fft.fft2(imagen_float)
    fft_shift = np.fft.fftshift(fft)
    
    # Crear filtro
    filtro = FiltrosFourier(fft_shift)
    
    # Aplicar según el tipo
    if tipo_filtro == 'ideal_pb':
        return filtro.ideal_pasabajas(parametro1)
    elif tipo_filtro == 'gaussiano_pb':
        return filtro.gaussiano_pasabajas(parametro1)
    elif tipo_filtro == 'butterworth_pb':
        orden = parametro2 if parametro2 else 2
        return filtro.butterworth_pasabajas(parametro1, orden)
    elif tipo_filtro == 'ideal_pa':
        return filtro.ideal_pasaaltas(parametro1)
    elif tipo_filtro == 'gaussiano_pa':
        return filtro.gaussiano_pasaaltas(parametro1)
    elif tipo_filtro == 'butterworth_pa':
        orden = parametro2 if parametro2 else 2
        return filtro.butterworth_pasaaltas(parametro1, orden)
    else:
        raise ValueError(f"Tipo de filtro desconocido: {tipo_filtro}")
