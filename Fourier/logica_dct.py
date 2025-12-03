# ===================================================================
# LÓGICA DE LA TRANSFORMADA DEL COSENO DISCRETA (DCT)
# Implementa DCT y compresión por bloques
# ===================================================================

import numpy as np
import cv2


class TransformadaDCT:
    """Clase para manejar la Transformada del Coseno Discreta"""
    
    def __init__(self):
        self.imagen = None
        self.dct_completa = None
        self.imagen_reconstruida = None
    
    def cargar_imagen(self, imagen):
        """Carga una imagen y la prepara para DCT"""
        self.imagen = imagen.astype(np.float32)
    
    def aplicar_dct_completa(self):
        """Aplica DCT a la imagen completa"""
        if self.imagen is None:
            return None
        
        # Aplicar DCT 2D a toda la imagen
        self.dct_completa = cv2.dct(self.imagen)
        return self.dct_completa
    
    def aplicar_idct_completa(self, dct_data=None):
        """Aplica DCT inversa a la imagen completa"""
        if dct_data is None:
            dct_data = self.dct_completa
        
        if dct_data is None:
            return None
        
        # Aplicar DCT inversa
        self.imagen_reconstruida = cv2.idct(dct_data)
        return self.imagen_reconstruida
    
    def obtener_magnitud_log(self):
        """Retorna la magnitud logarítmica de DCT para visualización"""
        if self.dct_completa is None:
            return None
        
        # Usar log para mejor visualización (similar a FFT)
        magnitud_log = np.log1p(np.abs(self.dct_completa))
        return magnitud_log
    
    def obtener_info(self):
        """Retorna información sobre la DCT calculada"""
        if self.dct_completa is None:
            return None
        
        return {
            'tamaño': self.dct_completa.shape,
            'min': self.dct_completa.min(),
            'max': self.dct_completa.max(),
            'media': self.dct_completa.mean(),
            'std': self.dct_completa.std()
        }


class CompresorDCT:
    """Clase para comprimir imágenes usando DCT por bloques 8x8"""
    
    def __init__(self, tamaño_bloque=8):
        """
        Inicializa el compresor
        
        Args:
            tamaño_bloque: Tamaño de los bloques (por defecto 8x8)
        """
        self.tamaño_bloque = tamaño_bloque
        self.imagen_original = None
        self.dct_bloques = None
        self.imagen_comprimida = None
    
    def cargar_imagen(self, imagen):
        """Carga y prepara la imagen para compresión"""
        self.imagen_original = imagen.astype(np.float32)
        
        # Ajustar dimensiones para que sean múltiplos del tamaño de bloque
        altura, ancho = self.imagen_original.shape
        
        # Calcular padding necesario
        pad_altura = (self.tamaño_bloque - altura % self.tamaño_bloque) % self.tamaño_bloque
        pad_ancho = (self.tamaño_bloque - ancho % self.tamaño_bloque) % self.tamaño_bloque
        
        # Aplicar padding si es necesario
        if pad_altura > 0 or pad_ancho > 0:
            self.imagen_original = np.pad(
                self.imagen_original,
                ((0, pad_altura), (0, pad_ancho)),
                mode='edge'
            )
        
        return self.imagen_original
    
    def aplicar_dct_por_bloques(self):
        """Aplica DCT a cada bloque de 8x8"""
        if self.imagen_original is None:
            return None
        
        altura, ancho = self.imagen_original.shape
        self.dct_bloques = np.zeros_like(self.imagen_original)
        
        # Procesar cada bloque
        for i in range(0, altura, self.tamaño_bloque):
            for j in range(0, ancho, self.tamaño_bloque):
                # Extraer bloque
                bloque = self.imagen_original[i:i+self.tamaño_bloque, j:j+self.tamaño_bloque]
                
                # Aplicar DCT al bloque
                dct_bloque = cv2.dct(bloque)
                
                # Guardar en la imagen de DCT
                self.dct_bloques[i:i+self.tamaño_bloque, j:j+self.tamaño_bloque] = dct_bloque
        
        return self.dct_bloques
    
    def comprimir_por_umbral(self, umbral_porcentaje=50):
        """
        Comprime eliminando coeficientes DCT menores a un umbral
        
        Args:
            umbral_porcentaje: Porcentaje de coeficientes a mantener (0-100)
        """
        if self.dct_bloques is None:
            self.aplicar_dct_por_bloques()
        
        # Crear copia de DCT
        dct_umbralizada = self.dct_bloques.copy()
        
        # Calcular umbral basado en el porcentaje
        valores_abs = np.abs(dct_umbralizada.flatten())
        umbral = np.percentile(valores_abs, 100 - umbral_porcentaje)
        
        # Eliminar coeficientes pequeños
        dct_umbralizada[np.abs(dct_umbralizada) < umbral] = 0
        
        # Reconstruir imagen
        self.imagen_comprimida = self._reconstruir_desde_dct(dct_umbralizada)
        
        # Calcular tasa de compresión
        coef_originales = dct_umbralizada.size
        coef_no_cero = np.count_nonzero(dct_umbralizada)
        tasa_compresion = (1 - coef_no_cero / coef_originales) * 100
        
        return self.imagen_comprimida, tasa_compresion, dct_umbralizada
    
    def comprimir_por_frecuencias(self, num_coeficientes=15):
        """
        Comprime manteniendo solo los primeros N coeficientes DCT en cada bloque
        (esquina superior izquierda = bajas frecuencias)
        
        Args:
            num_coeficientes: Número de coeficientes a mantener por bloque
        """
        if self.dct_bloques is None:
            self.aplicar_dct_por_bloques()
        
        # Crear copia de DCT
        dct_comprimida = self.dct_bloques.copy()
        
        altura, ancho = dct_comprimida.shape
        
        # Procesar cada bloque
        for i in range(0, altura, self.tamaño_bloque):
            for j in range(0, ancho, self.tamaño_bloque):
                # Extraer bloque DCT
                bloque = dct_comprimida[i:i+self.tamaño_bloque, j:j+self.tamaño_bloque]
                
                # Crear máscara en zig-zag (bajas frecuencias primero)
                mascara = self._crear_mascara_zigzag(self.tamaño_bloque, num_coeficientes)
                
                # Aplicar máscara
                bloque_comprimido = bloque * mascara
                
                # Guardar bloque comprimido
                dct_comprimida[i:i+self.tamaño_bloque, j:j+self.tamaño_bloque] = bloque_comprimido
        
        # Reconstruir imagen
        self.imagen_comprimida = self._reconstruir_desde_dct(dct_comprimida)
        
        # Calcular tasa de compresión
        tasa_compresion = (1 - num_coeficientes / (self.tamaño_bloque ** 2)) * 100
        
        return self.imagen_comprimida, tasa_compresion, dct_comprimida
    
    def _crear_mascara_zigzag(self, tamaño, num_coefs):
        """Crea una máscara que mantiene los primeros N coeficientes en orden zig-zag"""
        mascara = np.zeros((tamaño, tamaño))
        
        # Orden zig-zag simplificado (triangular superior)
        # Mantener coeficientes en la esquina superior izquierda
        n = int(np.ceil(np.sqrt(num_coefs * 2)))
        
        for i in range(tamaño):
            for j in range(tamaño):
                if i + j < n:
                    mascara[i, j] = 1
        
        return mascara
    
    def _reconstruir_desde_dct(self, dct_data):
        """Reconstruye la imagen aplicando IDCT a cada bloque"""
        altura, ancho = dct_data.shape
        imagen_reconstruida = np.zeros_like(dct_data)
        
        # Procesar cada bloque
        for i in range(0, altura, self.tamaño_bloque):
            for j in range(0, ancho, self.tamaño_bloque):
                # Extraer bloque DCT
                bloque_dct = dct_data[i:i+self.tamaño_bloque, j:j+self.tamaño_bloque]
                
                # Aplicar IDCT
                bloque_reconstruido = cv2.idct(bloque_dct)
                
                # Guardar en imagen reconstruida
                imagen_reconstruida[i:i+self.tamaño_bloque, j:j+self.tamaño_bloque] = bloque_reconstruido
        
        # Clip valores al rango válido
        imagen_reconstruida = np.clip(imagen_reconstruida, 0, 255)
        
        return imagen_reconstruida
    
    def obtener_estadisticas_compresion(self, imagen_original, imagen_comprimida):
        """Calcula estadísticas de la compresión"""
        # MSE
        mse = np.mean((imagen_original - imagen_comprimida) ** 2)
        
        # PSNR
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        return {
            'mse': mse,
            'psnr': psnr
        }
