"""
Lógica para procesar y aplicar mapas de calor a imágenes
"""
import cv2
import numpy as np


class HeatMapProcessor:
    def __init__(self):
        # Mapeo de nombres a constantes de OpenCV
        self.opencv_colormaps = {
            "AUTUMN": cv2.COLORMAP_AUTUMN,
            "BONE": cv2.COLORMAP_BONE,
            "JET": cv2.COLORMAP_JET,
            "WINTER": cv2.COLORMAP_WINTER,
            "RAINBOW": cv2.COLORMAP_RAINBOW,
            "OCEAN": cv2.COLORMAP_OCEAN,
            "SUMMER": cv2.COLORMAP_SUMMER,
            "SPRING": cv2.COLORMAP_SPRING,
            "COOL": cv2.COLORMAP_COOL,
            "HSV": cv2.COLORMAP_HSV,
            "PINK": cv2.COLORMAP_PINK,
            "HOT": cv2.COLORMAP_HOT,
            "PARULA": cv2.COLORMAP_PARULA,
            "MAGMA": cv2.COLORMAP_MAGMA,
            "INFERNO": cv2.COLORMAP_INFERNO,
            "PLASMA": cv2.COLORMAP_PLASMA,
            "VIRIDIS": cv2.COLORMAP_VIRIDIS,
            "CIVIDIS": cv2.COLORMAP_CIVIDIS,
            "TWILIGHT": cv2.COLORMAP_TWILIGHT,
            "TURBO": cv2.COLORMAP_TURBO
        }
    
    def aplicar_mapa_calor(self, imagen, mapa_tipo):
        """
        Aplicar un mapa de calor a una imagen
        
        Args:
            imagen: Imagen en formato BGR (OpenCV)
            mapa_tipo: Tipo de mapa de calor a aplicar
        
        Returns:
            Imagen con mapa de calor aplicado
        """
        # Convertir imagen a escala de grises
        if len(imagen.shape) == 3:
            imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gray = imagen.copy()
        
        # Aplicar mapa de calor
        if mapa_tipo == "PASTEL":
            # Mapa personalizado tipo pastel
            imagen_coloreada = self.aplicar_mapa_pastel(imagen_gray)
        elif mapa_tipo in self.opencv_colormaps:
            # Mapas de OpenCV
            imagen_coloreada = cv2.applyColorMap(imagen_gray, self.opencv_colormaps[mapa_tipo])
        else:
            raise ValueError(f"Mapa de calor no reconocido: {mapa_tipo}")
        
        return imagen_coloreada
    
    def aplicar_mapa_pastel(self, imagen_gray):
        """
        Crear y aplicar un mapa de colores personalizado con tonos pastel
        
        Args:
            imagen_gray: Imagen en escala de grises
        
        Returns:
            Imagen con colores pastel aplicados
        """
        # Crear imagen de salida en color (3 canales)
        altura, ancho = imagen_gray.shape
        imagen_coloreada = np.zeros((altura, ancho, 3), dtype=np.uint8)
        
        # Crear una tabla de colores (LUT - Look Up Table) de 256 entradas
        lut = np.zeros((256, 3), dtype=np.uint8)
        
        # Definir colores pastel en BGR para diferentes rangos de intensidad
        # Dividimos en 6 segmentos para transiciones suaves
        
        for i in range(256):
            # Normalizar el valor (0-1)
            t = i / 255.0
            
            if t < 0.16:  # Rosa pastel claro
                # Interpolación de blanco a rosa pastel
                factor = t / 0.16
                r = int(255 - factor * 30)
                g = int(255 - factor * 70)
                b = int(255 - factor * 50)
            elif t < 0.33:  # Rosa a melocotón
                factor = (t - 0.16) / 0.17
                r = int(225 - factor * 20)
                g = int(185 - factor * 15)
                b = int(205 + factor * 20)
            elif t < 0.50:  # Melocotón a amarillo pastel
                factor = (t - 0.33) / 0.17
                r = int(205 + factor * 40)
                g = int(170 + factor * 70)
                b = int(225 - factor * 45)
            elif t < 0.66:  # Amarillo pastel a verde menta
                factor = (t - 0.50) / 0.16
                r = int(245 - factor * 75)
                g = int(240 - factor * 25)
                b = int(180 + factor * 30)
            elif t < 0.83:  # Verde menta a azul cielo
                factor = (t - 0.66) / 0.17
                r = int(170 - factor * 30)
                g = int(215 - factor * 20)
                b = int(210 + factor * 35)
            else:  # Azul cielo a lavanda
                factor = (t - 0.83) / 0.17
                r = int(140 + factor * 50)
                g = int(195 - factor * 40)
                b = int(245 - factor * 15)
            
            # Asignar colores en formato BGR
            lut[i, 0] = b  # Blue
            lut[i, 1] = g  # Green
            lut[i, 2] = r  # Red
        
        # Aplicar LUT a cada píxel de la imagen
        for canal in range(3):
            imagen_coloreada[:, :, canal] = lut[imagen_gray, canal]
        
        return imagen_coloreada
    
    def crear_mapa_pastel_alternativo(self):
        """
        Crear un mapa de colores pastel alternativo con diferentes tonalidades
        
        Returns:
            LUT (Look Up Table) de 256x1x3 con colores pastel
        """
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        
        # Colores pastel inspirados en dulces y postres
        for i in range(256):
            t = i / 255.0
            
            # Usar funciones sinusoidales para transiciones suaves
            r = int(200 + 55 * np.sin(t * np.pi))
            g = int(180 + 75 * np.sin(t * np.pi + np.pi / 3))
            b = int(220 + 35 * np.sin(t * np.pi + 2 * np.pi / 3))
            
            # Asegurar que los valores estén en el rango válido
            lut[i, 0, 0] = np.clip(b, 0, 255)
            lut[i, 0, 1] = np.clip(g, 0, 255)
            lut[i, 0, 2] = np.clip(r, 0, 255)
        
        return lut
    
    def obtener_informacion_imagen(self, imagen):
        """
        Obtener información básica de una imagen
        
        Args:
            imagen: Imagen en formato OpenCV
        
        Returns:
            Diccionario con información de la imagen
        """
        info = {
            "dimensiones": imagen.shape,
            "tipo": imagen.dtype,
            "canales": len(imagen.shape) if len(imagen.shape) == 2 else imagen.shape[2],
            "min": imagen.min(),
            "max": imagen.max(),
            "media": imagen.mean()
        }
        return info
