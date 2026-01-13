"""
Interfaz gráfica para Pipeline Detallado OCR
Muestra cada paso del pipeline con sus imágenes y resultados
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import sys
import os

# Importar módulos del proyecto
directorio_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if directorio_raiz not in sys.path:
    sys.path.insert(0, directorio_raiz)

from Proyecto.pipeline_detallado_ocr import PipelineDetalladoOCR


class InterfazPipelineDetallado:
    """Interfaz para visualizar el pipeline OCR paso a paso"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Pipeline Detallado OCR - Paso a Paso")
        self.root.geometry("1400x900")
        
        self.pipeline = PipelineDetalladoOCR()
        self.imagen_original = None
        self.resultados_pipeline = None
        self.paso_actual = 0
        self.subpaso_actual = 0
        self.num_subpasos = 0
        
        self._crear_interfaz()
    
    def _crear_interfaz(self):
        """Crea la interfaz gráfica"""
        # Frame superior - Controles
        frame_controles = ttk.Frame(self.root, padding="10")
        frame_controles.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Button(frame_controles, text="📁 Cargar Imagen", 
                  command=self.cargar_imagen).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(frame_controles, text="▶ Ejecutar Pipeline Completo", 
                  command=self.ejecutar_pipeline).pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(frame_controles, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Label(frame_controles, text="Ver paso:").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(frame_controles, text="⬅ Anterior", 
                  command=self.paso_anterior).pack(side=tk.LEFT, padx=2)
        
        self.label_paso = ttk.Label(frame_controles, text="Paso 0/4", 
                                    font=('Arial', 10, 'bold'))
        self.label_paso.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(frame_controles, text="➡ Siguiente", 
                  command=self.paso_siguiente).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(frame_controles, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Label(frame_controles, text="Subpaso:").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(frame_controles, text="◀", 
                  command=self.subpaso_anterior).pack(side=tk.LEFT, padx=2)
        
        self.label_subpaso = ttk.Label(frame_controles, text="-", 
                                       font=('Arial', 9))
        self.label_subpaso.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(frame_controles, text="▶", 
                  command=self.subpaso_siguiente).pack(side=tk.LEFT, padx=2)
        
        # Frame principal con 2 paneles
        frame_principal = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        frame_principal.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Panel izquierdo - Visualización
        frame_izq = ttk.Frame(frame_principal)
        frame_principal.add(frame_izq, weight=7)
        
        # Canvas para imagen
        self.canvas_frame = ttk.Frame(frame_izq)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg='gray20')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Panel derecho - Información
        frame_der = ttk.Frame(frame_principal)
        frame_principal.add(frame_der, weight=3)
        
        # Información del paso actual
        ttk.Label(frame_der, text="Información del Paso", 
                 font=('Arial', 12, 'bold')).pack(pady=5)
        
        self.info_paso = scrolledtext.ScrolledText(frame_der, height=15, 
                                                   font=('Consolas', 9))
        self.info_paso.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Separator(frame_der, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Texto extraído final
        ttk.Label(frame_der, text="Texto Extraído Final", 
                 font=('Arial', 12, 'bold')).pack(pady=5)
        
        self.texto_final = scrolledtext.ScrolledText(frame_der, height=10, 
                                                     font=('Arial', 10))
        self.texto_final.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Barra de estado
        self.status_bar = ttk.Label(self.root, text="Listo", 
                                    relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def cargar_imagen(self):
        """Carga una imagen desde el disco"""
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), 
                      ("Todos", "*.*")]
        )
        
        if not ruta:
            return
        
        try:
            self.imagen_original = cv2.imread(ruta)
            if self.imagen_original is None:
                raise Exception("No se pudo cargar la imagen")
            
            self.imagen_original = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2RGB)
            self.mostrar_imagen(self.imagen_original)
            self.status_bar.config(text=f"Imagen cargada: {os.path.basename(ruta)}")
            
            # Resetear pipeline
            self.resultados_pipeline = None
            self.paso_actual = 0
            self.actualizar_info_paso()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar imagen:\n{str(e)}")
    
    def ejecutar_pipeline(self):
        """Ejecuta el pipeline completo"""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            # Cargar motor OCR si no está cargado
            if self.pipeline.reader is None:
                self.status_bar.config(text="Cargando EasyOCR (English)...")
                self.root.update()
                if not self.pipeline.cargar_motor_ocr(['en']):
                    messagebox.showerror("Error", "No se pudo cargar EasyOCR")
                    return
            
            self.status_bar.config(text="Ejecutando pipeline...")
            self.root.update()
            
            # Ejecutar pipeline
            self.resultados_pipeline = self.pipeline.ejecutar_pipeline_completo(
                self.imagen_original, ver_pasos=True
            )
            
            # Mostrar primer paso
            self.paso_actual = 1
            self.subpaso_actual = 0
            self.mostrar_paso(1)
            
            self.status_bar.config(text="Pipeline completado exitosamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en pipeline:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def paso_anterior(self):
        """Muestra el paso anterior"""
        if self.paso_actual > 1:
            self.paso_actual -= 1
            self.subpaso_actual = 0
            self.mostrar_paso(self.paso_actual)
    
    def paso_siguiente(self):
        """Muestra el paso siguiente"""
        if self.resultados_pipeline and self.paso_actual < 4:
            self.paso_actual += 1
            self.subpaso_actual = 0
            self.mostrar_paso(self.paso_actual)
    
    def subpaso_anterior(self):
        """Muestra el subpaso anterior"""
        if self.subpaso_actual > 0:
            self.subpaso_actual -= 1
            self.mostrar_paso(self.paso_actual)
    
    def subpaso_siguiente(self):
        """Muestra el subpaso siguiente"""
        if self.subpaso_actual < self.num_subpasos - 1:
            self.subpaso_actual += 1
            self.mostrar_paso(self.paso_actual)
    
    def mostrar_paso(self, num_paso):
        """Muestra un paso específico del pipeline"""
        if not self.resultados_pipeline:
            return
        
        self.label_paso.config(text=f"Paso {num_paso}/4")
        
        # Mostrar imagen del paso
        if num_paso == 1:
            # Preprocesamiento - Mostrar subpasos
            historial = self.resultados_pipeline['paso_1_historial']
            self.num_subpasos = len(historial)
            
            if self.subpaso_actual >= self.num_subpasos:
                self.subpaso_actual = self.num_subpasos - 1
            
            subpaso = historial[self.subpaso_actual]
            img = subpaso['imagen']
            titulo = f"PASO 1: Preprocesamiento - {subpaso['nombre']}"
            info = self._generar_info_paso_1()
            
            self.label_subpaso.config(text=f"{self.subpaso_actual + 1}/{self.num_subpasos}")
            
        elif num_paso == 2:
            # Detección CRAFT - Mostrar imagen con boxes
            self.num_subpasos = 1
            self.subpaso_actual = 0
            img = self.resultados_pipeline['paso_2_imagen_boxes']
            titulo = "PASO 2: Detección de Texto con CRAFT"
            info = self._generar_info_paso_2()
            self.label_subpaso.config(text="-")
            
        elif num_paso == 3:
            # Recorte de regiones - Mostrar mosaico de regiones
            self.num_subpasos = 1
            self.subpaso_actual = 0
            img = self._crear_mosaico_regiones()
            titulo = "PASO 3: Regiones de Texto Recortadas"
            info = self._generar_info_paso_3()
            self.label_subpaso.config(text="-")
            
        elif num_paso == 4:
            # Reconocimiento CRNN - Mostrar resultado final
            self.num_subpasos = 1
            self.subpaso_actual = 0
            img = self.resultados_pipeline['paso_2_imagen_boxes']
            titulo = "PASO 4: Reconocimiento de Texto con CRNN"
            info = self._generar_info_paso_4()
            self.label_subpaso.config(text="-")
        
        self.mostrar_imagen(img, titulo)
        self.actualizar_info_paso(info)
        
        # Actualizar texto final
        if 'paso_4_texto_final' in self.resultados_pipeline:
            self.texto_final.delete(1.0, tk.END)
            self.texto_final.insert(1.0, self.resultados_pipeline['paso_4_texto_final'])
    
    def _generar_info_paso_1(self):
        """Genera información del paso 1"""
        historial = self.resultados_pipeline['paso_1_historial']
        info = "═══ PASO 1: PREPROCESAMIENTO ═══\n\n"
        info += "Objetivo: Obtener imagen binaria con texto resaltado\n\n"
        
        # Mostrar información del subpaso actual
        subpaso_actual = historial[self.subpaso_actual]
        info += f">>> MOSTRANDO: {subpaso_actual['nombre']} <<<\n"
        info += f"    (Subpaso {self.subpaso_actual + 1} de {len(historial)})\n\n"
        
        info += "Todos los subpasos del preprocesamiento:\n\n"
        
        # Explicaciones detalladas por subpaso
        explicaciones = {
            "Conversión a escala de grises": 
                "Reduce dimensionalidad de color a intensidad.\n"
                "Facilita procesamiento posterior.",
            
            "Reducción de ruido (Bilateral)":
                "Elimina ruido preservando bordes nítidos.\n"
                "Parámetros: d=5, sigmaColor=50, sigmaSpace=50\n"
                "Prepara imagen para mejora de contraste.",
            
            "Mejora de contraste (CLAHE mejorado)":
                "CLAHE = Contrast Limited Adaptive Histogram Equalization\n"
                "Mejora contraste local sin amplificar ruido.\n"
                "Parámetros: clipLimit=2.5, tileGrid=(6,6)\n"
                "Resalta texto débil o con baja iluminación.",
            
            "Umbralización adaptativa (GAUSSIAN optimizado)":
                "Convierte a blanco/negro según umbral local.\n"
                "Método GAUSSIAN: mejor para iluminación no uniforme.\n"
                "Parámetros: blockSize=13, C=3\n"
                "Separa texto del fondo efectivamente.",
            
            "Cierre morfológico (conectar caracteres)":
                "Operación morfológica para conectar píxeles.\n"
                "Kernel: 1x1 (mínima distorsión)\n"
                "Rellena pequeños huecos en caracteres.",
            
            "Eliminación de ruido residual (Filtro de Mediana)":
                "Elimina ruido sal y pimienta restante.\n"
                "Kernel: 3x3 (preserva detalles finos)\n"
                "Limpieza final antes de OCR."
        }
        
        for i, subpaso in enumerate(historial, 1):
            nombre = subpaso['nombre']
            marcador = "→" if i - 1 == self.subpaso_actual else " "
            info += f"{marcador} {i}. {nombre}\n"
            
            # Buscar explicación que coincida
            for clave, explicacion in explicaciones.items():
                if clave in nombre:
                    info += f"   {explicacion}\n"
                    break
            info += "\n"
        
        info += "Resultado: Imagen binaria optimizada para detección de texto\n"
        info += "\nUsa los botones ◀ ▶ para ver cada subpaso\n"
        return info
    
    def _generar_info_paso_2(self):
        """Genera información del paso 2"""
        regiones = self.resultados_pipeline['paso_2_regiones_detectadas']
        info = "═══ PASO 2: DETECCIÓN CON CRAFT ═══\n\n"
        info += "Objetivo: Detectar áreas donde hay texto\n\n"
        info += "CRAFT = Character Region Awareness For Text detection\n"
        info += "• Algoritmo de detección basado en deep learning\n"
        info += "• Detecta regiones de texto sin reconocerlo\n"
        info += "• Genera bounding boxes alrededor de cada área\n\n"
        info += f"Regiones detectadas: {len(regiones)}\n\n"
        for region in regiones:
            x, y, x2, y2 = region['bbox']
            info += f"  Región #{region['id']}\n"
            info += f"    Posición: ({x}, {y}) → ({x2}, {y2})\n"
            info += f"    Tamaño: {x2-x} × {y2-y} px\n"
            info += f"    Área: {region['area']} px²\n\n"
        info += "Las regiones se muestran con rectángulos verdes\n"
        return info
    
    def _generar_info_paso_3(self):
        """Genera información del paso 3"""
        regiones = self.resultados_pipeline['paso_3_regiones_recortadas']
        info = "═══ PASO 3: RECORTE DE REGIONES ═══\n\n"
        info += "Objetivo: Extraer cada región detectada\n\n"
        info += "Proceso:\n"
        info += "• Se toman las coordenadas del bounding box\n"
        info += "• Se recorta esa área de la imagen original\n"
        info += "• Cada región se procesa independientemente\n"
        info += "• Facilita el reconocimiento individual\n\n"
        info += f"Regiones recortadas: {len(regiones)}\n\n"
        for region in regiones:
            w, h = region['tamaño']
            info += f"  Región #{region['id']}\n"
            info += f"    Tamaño: {w} × {h} px\n"
            info += f"    Área: {w * h} px²\n\n"
        info += "Las regiones se muestran en mosaico para visualización\n"
        return info
    
    def _generar_info_paso_4(self):
        """Genera información del paso 4"""
        detalles = self.resultados_pipeline['paso_4_detalles']
        texto = self.resultados_pipeline['paso_4_texto_final']
        info = "═══ PASO 4: RECONOCIMIENTO CRNN ═══\n\n"
        info += "Objetivo: Reconocer texto en cada región\n\n"
        info += "CRNN = Convolutional Recurrent Neural Network\n"
        info += "Arquitectura:\n"
        info += "  1. CNN (Convolutional): Extrae características visuales\n"
        info += "     • Detecta formas, líneas, curvas de caracteres\n"
        info += "     • Genera mapa de características\n\n"
        info += "  2. RNN (Recurrent): Procesa secuencia de características\n"
        info += "     • LSTM/GRU para contexto temporal\n"
        info += "     • Entiende relación entre caracteres\n\n"
        info += "  3. CTC (Connectionist Temporal Classification)\n"
        info += "     • Alinea salida con texto sin segmentación\n"
        info += "     • Maneja longitudes variables\n\n"
        info += f"Texto extraído de {len(detalles)} regiones:\n\n"
        
        if detalles:
            for detalle in detalles:
                info += f"  Región #{detalle['id']}\n"
                info += f"    Texto: '{detalle['texto']}'\n"
                info += f"    Confianza: {detalle['confianza']:.1f}%\n"
                info += f"    Longitud: {len(detalle['texto'])} caracteres\n\n"
        else:
            info += "  [No se detectó texto]\n\n"
        
        info += f"\n{'='*40}\n"
        info += f"TEXTO FINAL COMPLETO:\n"
        info += f"{'='*40}\n"
        info += f"'{texto}'\n\n"
        info += f"Longitud total: {len(texto)} caracteres\n"
        info += f"Palabras: {len(texto.split())}\n"
        return info
    
    def _crear_mosaico_regiones(self):
        """Crea un mosaico con todas las regiones recortadas"""
        regiones = self.resultados_pipeline['paso_3_regiones_recortadas']
        
        if not regiones:
            return np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Calcular disposición del mosaico
        num_regiones = len(regiones)
        cols = min(3, num_regiones)
        rows = (num_regiones + cols - 1) // cols
        
        # Tamaño de cada celda
        cell_h = 150
        cell_w = 300
        
        # Crear canvas
        mosaico = np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) * 240
        
        for idx, region in enumerate(regiones):
            row = idx // cols
            col = idx % cols
            
            img_region = region['imagen']
            if len(img_region.shape) == 2:
                img_region = cv2.cvtColor(img_region, cv2.COLOR_GRAY2RGB)
            
            # Redimensionar para que quepa en la celda
            h, w = img_region.shape[:2]
            scale = min((cell_w - 20) / w, (cell_h - 40) / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_resized = cv2.resize(img_region, (new_w, new_h))
            
            # Posición en el mosaico
            y_start = row * cell_h + 30
            x_start = col * cell_w + 10
            
            # Colocar imagen
            mosaico[y_start:y_start+new_h, x_start:x_start+new_w] = img_resized
            
            # Agregar etiqueta
            cv2.putText(mosaico, f"Region #{region['id']}", 
                       (x_start, row * cell_h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return mosaico
    
    def actualizar_info_paso(self, texto=""):
        """Actualiza el panel de información"""
        self.info_paso.delete(1.0, tk.END)
        self.info_paso.insert(1.0, texto)
    
    def mostrar_imagen(self, imagen, titulo=""):
        """Muestra una imagen en el canvas"""
        if imagen is None:
            return
        
        # Convertir a PIL
        img = Image.fromarray(imagen)
        
        # Redimensionar para ajustar al canvas
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w > 1 and canvas_h > 1:
            img.thumbnail((canvas_w - 20, canvas_h - 40), Image.Resampling.LANCZOS)
        
        # Convertir a PhotoImage
        self.photo = ImageTk.PhotoImage(img)
        
        # Mostrar en canvas
        self.canvas.delete("all")
        x = (canvas_w - self.photo.width()) // 2
        y = (canvas_h - self.photo.height()) // 2 + 20
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
        
        # Título
        if titulo:
            self.canvas.create_text(canvas_w // 2, 10, 
                                   text=titulo, 
                                   font=('Arial', 14, 'bold'),
                                   fill='white')


def main():
    root = tk.Tk()
    app = InterfazPipelineDetallado(root)
    root.mainloop()


if __name__ == "__main__":
    main()
