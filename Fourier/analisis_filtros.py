# ===================================================================
# ANÁLISIS DE EFECTOS DE FILTRADO
# Compara estadísticas antes y después del filtrado
# ===================================================================

import numpy as np
import cv2


class AnalizadorFiltros:
    """Clase para analizar los efectos de los filtros en las imágenes"""
    
    def __init__(self, imagen_original, imagen_filtrada):
        """
        Inicializa el analizador con las imágenes
        
        Args:
            imagen_original: Imagen antes del filtrado
            imagen_filtrada: Imagen después del filtrado
        """
        self.original = imagen_original.astype(np.float64)
        self.filtrada = imagen_filtrada.astype(np.float64)
    
    def calcular_estadisticas(self):
        """Calcula estadísticas básicas de ambas imágenes"""
        stats = {
            'original': {
                'media': np.mean(self.original),
                'std': np.std(self.original),
                'min': np.min(self.original),
                'max': np.max(self.original),
                'varianza': np.var(self.original),
                'mediana': np.median(self.original)
            },
            'filtrada': {
                'media': np.mean(self.filtrada),
                'std': np.std(self.filtrada),
                'min': np.min(self.filtrada),
                'max': np.max(self.filtrada),
                'varianza': np.var(self.filtrada),
                'mediana': np.median(self.filtrada)
            }
        }
        return stats
    
    def calcular_diferencia(self):
        """Calcula la diferencia absoluta entre las imágenes"""
        diferencia = np.abs(self.original - self.filtrada)
        return diferencia
    
    def calcular_mse(self):
        """Calcula el Error Cuadrático Medio (MSE)"""
        mse = np.mean((self.original - self.filtrada) ** 2)
        return mse
    
    def calcular_psnr(self):
        """Calcula la Relación Señal-Ruido de Pico (PSNR)"""
        mse = self.calcular_mse()
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def calcular_ssim_simple(self):
        """
        Calcula una métrica simple de similitud estructural
        (versión simplificada del SSIM)
        """
        # Constantes para estabilidad
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        # Medias
        mu1 = np.mean(self.original)
        mu2 = np.mean(self.filtrada)
        
        # Varianzas
        sigma1_sq = np.var(self.original)
        sigma2_sq = np.var(self.filtrada)
        
        # Covarianza
        sigma12 = np.mean((self.original - mu1) * (self.filtrada - mu2))
        
        # Calcular SSIM
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return ssim
    
    def analizar_frecuencias(self):
        """Analiza el contenido frecuencial de ambas imágenes"""
        # Calcular FFT de ambas imágenes
        fft_orig = np.fft.fft2(self.original)
        fft_filt = np.fft.fft2(self.filtrada)
        
        # Calcular magnitudes
        mag_orig = np.abs(fft_orig)
        mag_filt = np.abs(fft_filt)
        
        # Energía total en frecuencia
        energia_orig = np.sum(mag_orig)
        energia_filt = np.sum(mag_filt)
        
        # Porcentaje de energía conservada
        porcentaje_energia = (energia_filt / energia_orig) * 100
        
        return {
            'energia_original': energia_orig,
            'energia_filtrada': energia_filt,
            'porcentaje_conservado': porcentaje_energia
        }
    
    def obtener_reporte_completo(self):
        """Genera un reporte completo del análisis"""
        stats = self.calcular_estadisticas()
        mse = self.calcular_mse()
        psnr = self.calcular_psnr()
        ssim = self.calcular_ssim_simple()
        freq_info = self.analizar_frecuencias()
        
        reporte = {
            'estadisticas': stats,
            'metricas': {
                'mse': mse,
                'psnr': psnr,
                'ssim': ssim
            },
            'frecuencias': freq_info
        }
        
        return reporte
    
    def generar_texto_reporte(self):
        """Genera un texto formateado del reporte"""
        reporte = self.obtener_reporte_completo()
        
        texto = "="*60 + "\n"
        texto += "ANÁLISIS DE EFECTOS DEL FILTRADO\n"
        texto += "="*60 + "\n\n"
        
        texto += "ESTADÍSTICAS COMPARATIVAS:\n"
        texto += "-"*60 + "\n"
        texto += f"{'Métrica':<20} {'Original':<18} {'Filtrada':<18}\n"
        texto += "-"*60 + "\n"
        
        stats = reporte['estadisticas']
        metricas_nombres = {
            'media': 'Media',
            'std': 'Desv. Estándar',
            'min': 'Mínimo',
            'max': 'Máximo',
            'varianza': 'Varianza',
            'mediana': 'Mediana'
        }
        
        for key, nombre in metricas_nombres.items():
            orig = stats['original'][key]
            filt = stats['filtrada'][key]
            texto += f"{nombre:<20} {orig:<18.2f} {filt:<18.2f}\n"
        
        texto += "\n"
        texto += "MÉTRICAS DE SIMILITUD:\n"
        texto += "-"*60 + "\n"
        texto += f"MSE (Error Cuadrático Medio): {reporte['metricas']['mse']:.4f}\n"
        texto += f"PSNR (Relación Señal-Ruido):  {reporte['metricas']['psnr']:.2f} dB\n"
        texto += f"SSIM (Similitud Estructural):  {reporte['metricas']['ssim']:.4f}\n"
        
        texto += "\n"
        texto += "ANÁLISIS DE FRECUENCIAS:\n"
        texto += "-"*60 + "\n"
        freq = reporte['frecuencias']
        texto += f"Energía Original:    {freq['energia_original']:.2e}\n"
        texto += f"Energía Filtrada:    {freq['energia_filtrada']:.2e}\n"
        texto += f"Energía Conservada:  {freq['porcentaje_conservado']:.2f}%\n"
        
        texto += "\n"
        texto += "INTERPRETACIÓN:\n"
        texto += "-"*60 + "\n"
        
        # Interpretación de PSNR
        psnr = reporte['metricas']['psnr']
        if psnr > 40:
            texto += "• PSNR: Excelente calidad, cambios mínimos\n"
        elif psnr > 30:
            texto += "• PSNR: Buena calidad, cambios moderados\n"
        elif psnr > 20:
            texto += "• PSNR: Calidad aceptable, cambios notorios\n"
        else:
            texto += "• PSNR: Baja calidad, cambios significativos\n"
        
        # Interpretación de SSIM
        ssim = reporte['metricas']['ssim']
        if ssim > 0.95:
            texto += "• SSIM: Muy similar estructuralmente\n"
        elif ssim > 0.85:
            texto += "• SSIM: Similar estructuralmente\n"
        elif ssim > 0.70:
            texto += "• SSIM: Moderadamente similar\n"
        else:
            texto += "• SSIM: Diferencias estructurales significativas\n"
        
        texto += "="*60 + "\n"
        
        return texto
