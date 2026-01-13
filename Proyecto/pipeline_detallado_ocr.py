"""
Pipeline Detallado OCR - Paso a Paso
1. Preprocesamiento → Imagen binaria
2. Detección de texto (CRAFT) → Áreas de texto
3. Recorte de regiones detectadas
4. Reconocimiento (CRNN) → Extracción de texto
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import sys
import os

# Importar módulos del proyecto
directorio_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if directorio_raiz not in sys.path:
    sys.path.insert(0, directorio_raiz)

from Proyecto.preprocesamiento_ocr import PipelinePreprocesamientoOCR


class PipelineDetalladoOCR:
    """Pipeline paso a paso para OCR con visualización de cada etapa"""
    
    def __init__(self):
        """Inicializa el pipeline detallado"""
        self.preprocesador = PipelinePreprocesamientoOCR()
        self.reader = None
        self.pasos_ejecutados = []
        
    def cargar_motor_ocr(self, idiomas=['en']):
        """Carga EasyOCR para detección y reconocimiento"""
        try:
            import easyocr
            import ssl
            
            try:
                ssl._create_default_https_context = ssl._create_unverified_context
            except:
                pass
            
            print("Cargando EasyOCR (CRAFT + CRNN) - Idioma: English...")
            self.reader = easyocr.Reader(idiomas, gpu=False, download_enabled=True)
            print("✓ EasyOCR cargado correctamente")
            return True
        except Exception as e:
            print(f"✗ Error al cargar EasyOCR: {str(e)}")
            return False
    
    def ejecutar_pipeline_completo(self, imagen: np.ndarray, 
                                   ver_pasos: bool = True) -> Dict:
        """
        Ejecuta el pipeline completo paso a paso
        
        Args:
            imagen: Imagen de entrada
            ver_pasos: Si True, retorna imágenes de cada paso
            
        Returns:
            Diccionario con todos los pasos y resultados
        """
        self.pasos_ejecutados = []
        
        # PASO 1: Preprocesamiento → Imagen binaria
        print("\n[PASO 1] Preprocesamiento → Imagen binaria")
        img_binaria, historial_preproceso = self.paso_1_preprocesamiento(imagen)
        
        # PASO 2: Detección de texto con CRAFT
        print("\n[PASO 2] Detección de texto con CRAFT")
        regiones_detectadas, img_con_boxes = self.paso_2_deteccion_craft(img_binaria)
        
        # PASO 3: Recorte de regiones de texto
        print("\n[PASO 3] Recorte de regiones de texto")
        regiones_recortadas = self.paso_3_recortar_regiones(imagen, regiones_detectadas)
        
        # PASO 4: Reconocimiento con CRNN
        print("\n[PASO 4] Reconocimiento de texto con CRNN")
        texto_final, detalles_reconocimiento = self.paso_4_reconocimiento_crnn(regiones_recortadas)
        
        return {
            'paso_1_imagen_binaria': img_binaria,
            'paso_1_historial': historial_preproceso,
            'paso_2_regiones_detectadas': regiones_detectadas,
            'paso_2_imagen_boxes': img_con_boxes,
            'paso_3_regiones_recortadas': regiones_recortadas,
            'paso_4_texto_final': texto_final,
            'paso_4_detalles': detalles_reconocimiento,
            'resumen': self._generar_resumen(texto_final, regiones_detectadas)
        }
    
    def paso_1_preprocesamiento(self, imagen: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        PASO 1: Preprocesamiento para obtener imagen binaria
        - Grises
        - Bilateral
        - CLAHE
        - Umbralización adaptativa
        - Morfología
        - Mediana
        
        Returns:
            (imagen_binaria, historial_pasos)
        """
        img_binaria, historial = self.preprocesador.procesar(imagen)
        
        paso_info = {
            'nombre': 'Preprocesamiento',
            'descripcion': 'Convertir a imagen binaria con texto resaltado',
            'imagen': img_binaria,
            'subpasos': historial
        }
        self.pasos_ejecutados.append(paso_info)
        
        print(f"  ✓ Imagen binaria generada ({len(historial)} subpasos)")
        return img_binaria, historial
    
    def paso_2_deteccion_craft(self, imagen_binaria: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        PASO 2: Detección de áreas de texto con CRAFT
        Usa EasyOCR en modo detección optimizado con filtrado y ordenamiento
        
        Returns:
            (lista_regiones, imagen_con_boxes)
        """
        if self.reader is None:
            print("  ✗ Motor OCR no cargado")
            return [], imagen_binaria
        
        try:
            # Detectar regiones de texto con parámetros optimizados
            resultados = self.reader.readtext(
                imagen_binaria, 
                detail=1,
                paragraph=False,  # Detectar líneas individuales, no párrafos
                min_size=10,      # Tamaño mínimo de texto (px)
                text_threshold=0.7,  # Umbral de confianza para detección
                low_text=0.4      # Umbral para regiones de texto débil
            )
            
            # Dimensiones de la imagen para validación
            h, w = imagen_binaria.shape[:2]
            area_imagen = h * w
            
            regiones = []
            for idx, (bbox, texto, conf) in enumerate(resultados):
                # Convertir bbox a formato estándar
                puntos = np.array(bbox, dtype=np.int32)
                x_min, y_min = puntos.min(axis=0)
                x_max, y_max = puntos.max(axis=0)
                
                # Calcular área y dimensiones
                ancho = x_max - x_min
                alto = y_max - y_min
                area = ancho * alto
                
                # Filtrar regiones muy pequeñas (ruido) o muy grandes (false positives)
                if area < 50 or area > area_imagen * 0.8:
                    continue
                
                # Filtrar regiones con proporciones inválidas
                aspect_ratio = ancho / alto if alto > 0 else 0
                if aspect_ratio < 0.1 or aspect_ratio > 50:  # Demasiado estrecho o ancho
                    continue
                
                # Expandir bounding box ligeramente (padding)
                padding = 3
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                region = {
                    'id': idx + 1,
                    'bbox': (x_min, y_min, x_max, y_max),
                    'puntos': puntos,
                    'area': (x_max - x_min) * (y_max - y_min),
                    'centro': ((x_min + x_max) // 2, (y_min + y_max) // 2),
                    'confianza_deteccion': conf
                }
                regiones.append(region)
            
            # Ordenar regiones por posición (arriba a abajo, izquierda a derecha)
            regiones = self._ordenar_regiones(regiones)
            
            # Reasignar IDs después del ordenamiento
            for idx, region in enumerate(regiones):
                region['id'] = idx + 1
            
            # Dibujar boxes en la imagen
            img_con_boxes = self._dibujar_boxes(imagen_binaria, regiones)
            
            paso_info = {
                'nombre': 'Detección CRAFT',
                'descripcion': f'{len(regiones)} regiones de texto detectadas (filtradas y ordenadas)',
                'imagen': img_con_boxes,
                'regiones': regiones
            }
            self.pasos_ejecutados.append(paso_info)
            
            print(f"  ✓ {len(regiones)} regiones detectadas con CRAFT")
            if len(regiones) > 0:
                print(f"     (Filtradas por tamaño y ordenadas por posición)")
            return regiones, img_con_boxes
            
        except Exception as e:
            print(f"  ✗ Error en detección: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], imagen_binaria
    
    def _ordenar_regiones(self, regiones: List[Dict]) -> List[Dict]:
        """
        Ordena regiones de texto en orden de lectura natural
        (de arriba a abajo, de izquierda a derecha)
        """
        if not regiones:
            return regiones
        
        # Ordenar por coordenada Y (arriba a abajo) con tolerancia para líneas
        # Luego por coordenada X (izquierda a derecha)
        def clave_ordenamiento(region):
            centro_x, centro_y = region['centro']
            # Agrupar por líneas con tolerancia de 20 píxeles
            linea = centro_y // 20
            return (linea, centro_x)
        
        return sorted(regiones, key=clave_ordenamiento)
    
    def paso_3_recortar_regiones(self, imagen_original: np.ndarray, 
                                 regiones: List[Dict]) -> List[Dict]:
        """
        PASO 3: Recortar cada región de texto detectada
        
        Returns:
            Lista de diccionarios con imágenes recortadas
        """
        regiones_recortadas = []
        
        for region in regiones:
            x_min, y_min, x_max, y_max = region['bbox']
            
            # Recortar región
            img_recortada = imagen_original[y_min:y_max, x_min:x_max]
            
            region_info = {
                'id': region['id'],
                'bbox': region['bbox'],
                'imagen': img_recortada,
                'tamaño': (x_max - x_min, y_max - y_min)
            }
            regiones_recortadas.append(region_info)
        
        paso_info = {
            'nombre': 'Recorte de Regiones',
            'descripcion': f'{len(regiones_recortadas)} regiones recortadas',
            'regiones': regiones_recortadas
        }
        self.pasos_ejecutados.append(paso_info)
        
        print(f"  ✓ {len(regiones_recortadas)} regiones recortadas")
        return regiones_recortadas
    
    def paso_4_reconocimiento_crnn(self, regiones_recortadas: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        PASO 4: Reconocimiento de texto con CRNN (CNN + RNN)
        Procesa cada región recortada individualmente
        
        Returns:
            (texto_completo, detalles_por_region)
        """
        if self.reader is None:
            print("  ✗ Motor OCR no cargado")
            return "", []
        
        texto_completo = []
        detalles = []
        
        for region in regiones_recortadas:
            try:
                # Reconocer texto en la región recortada
                resultado = self.reader.readtext(region['imagen'], detail=1)
                
                if resultado:
                    # Tomar el resultado con mayor confianza
                    mejor_resultado = max(resultado, key=lambda x: x[2])
                    bbox, texto, confianza = mejor_resultado
                    
                    detalle = {
                        'id': region['id'],
                        'texto': texto,
                        'confianza': confianza * 100,
                        'bbox_original': region['bbox'],
                        'tamaño': region['tamaño']
                    }
                    detalles.append(detalle)
                    texto_completo.append(texto)
                    
                    print(f"    Región {region['id']}: '{texto}' (confianza: {confianza*100:.1f}%)")
                else:
                    print(f"    Región {region['id']}: [Sin texto detectado]")
                    
            except Exception as e:
                print(f"    Región {region['id']}: Error - {str(e)}")
        
        texto_final = ' '.join(texto_completo)
        
        paso_info = {
            'nombre': 'Reconocimiento CRNN',
            'descripcion': f'Texto extraído de {len(detalles)} regiones',
            'texto': texto_final,
            'detalles': detalles
        }
        self.pasos_ejecutados.append(paso_info)
        
        print(f"  ✓ Texto final: '{texto_final}'")
        return texto_final, detalles
    
    def _dibujar_boxes(self, imagen: np.ndarray, regiones: List[Dict]) -> np.ndarray:
        """Dibuja los bounding boxes en la imagen"""
        # Convertir a BGR si es necesario
        if len(imagen.shape) == 2:
            img_color = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        else:
            img_color = imagen.copy()
        
        for region in regiones:
            # Dibujar rectángulo
            x_min, y_min, x_max, y_max = region['bbox']
            cv2.rectangle(img_color, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Dibujar ID de región
            cv2.putText(img_color, f"#{region['id']}", 
                       (x_min, y_min - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 1)
        
        return img_color
    
    def _generar_resumen(self, texto_final: str, regiones: List[Dict]) -> Dict:
        """Genera un resumen del pipeline ejecutado"""
        return {
            'num_regiones_detectadas': len(regiones),
            'texto_extraido': texto_final,
            'longitud_texto': len(texto_final),
            'num_pasos_ejecutados': len(self.pasos_ejecutados)
        }
    
    def obtener_paso(self, numero_paso: int) -> Optional[Dict]:
        """Obtiene la información de un paso específico"""
        if 0 <= numero_paso - 1 < len(self.pasos_ejecutados):
            return self.pasos_ejecutados[numero_paso - 1]
        return None
    
    def listar_pasos(self) -> List[str]:
        """Lista los nombres de todos los pasos ejecutados"""
        return [f"Paso {i+1}: {paso['nombre']}" for i, paso in enumerate(self.pasos_ejecutados)]
