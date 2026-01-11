"""
Interfaz Gráfica para Procesamiento Básico de Imágenes Digitales
- Conversión RGB a escala de grises
- Binarización con umbral fijo y automático
- Separación de canales RGB
- Conversión entre modelos de color (RGB, CMY, YIQ, HSI, HSV)
- Visualización de histogramas con propiedades estadísticas
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from procesamiento_basico import (
    rgb_a_grises,
    binarizacion_umbral_fijo,
    binarizacion_umbral_otsu,
    calcular_histograma,
    separar_canales_rgb,
    separar_canales_rgb_visualizar,
    calcular_histogramas_rgb,
    propiedades_histograma,
    rgb_a_cmy,
    rgb_a_yiq,
    rgb_a_hsi,
    rgb_a_hsv
)


class InterfazImagenDigital:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesamiento Básico de Imágenes Digitales")
        self.root.geometry("1600x900")
        
        self.imagen_original = None
        self.imagen_grises = None
        self.imagen_procesada = None
        
        self.crear_interfaz()
    
    def crear_interfaz(self):
        """Crea la interfaz gráfica principal"""
        
        # Frame principal
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel superior - Controles
        panel_superior = ttk.LabelFrame(main_container, text="Controles", padding=10)
        panel_superior.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(panel_superior, text="Cargar Imagen", 
                  command=self.cargar_imagen).pack(side=tk.LEFT, padx=5)
        ttk.Button(panel_superior, text="Guardar Resultado", 
                  command=self.guardar_imagen).pack(side=tk.LEFT, padx=5)
        ttk.Button(panel_superior, text="Reiniciar", 
                  command=self.reiniciar).pack(side=tk.LEFT, padx=5)
        
        # Frame contenedor para panel izquierdo y derecho
        frame_contenido = ttk.Frame(main_container)
        frame_contenido.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo con scroll - Operaciones
        frame_izq_container = ttk.Frame(frame_contenido)
        frame_izq_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Canvas con scrollbar
        canvas_izq = tk.Canvas(frame_izq_container, width=380, bg='#f0f0f0')
        scrollbar_izq = ttk.Scrollbar(frame_izq_container, orient="vertical", command=canvas_izq.yview)
        
        panel_izquierdo = ttk.Frame(canvas_izq)
        
        panel_izquierdo.bind(
            "<Configure>",
            lambda e: canvas_izq.configure(scrollregion=canvas_izq.bbox("all"))
        )
        
        canvas_izq.create_window((0, 0), window=panel_izquierdo, anchor="nw")
        canvas_izq.configure(yscrollcommand=scrollbar_izq.set)
        
        canvas_izq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_izq.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Scroll con rueda del mouse
        def _on_mousewheel(event):
            canvas_izq.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas_izq.bind_all("<MouseWheel>", _on_mousewheel)
        
        # === SECCIÓN 1: CONVERSIÓN A GRISES ===
        ttk.Label(panel_izquierdo, text="1. Conversión a Escala de Grises", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        ttk.Button(panel_izquierdo, text="Convertir a Grises",
                  command=self.convertir_a_grises, width=35).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Separator(panel_izquierdo, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # === SECCIÓN 2: SEPARACIÓN DE CANALES ===
        ttk.Label(panel_izquierdo, text="2. Separación de Canales RGB", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        ttk.Button(panel_izquierdo, text="Separar Canales RGB",
                  command=self.separar_canales, width=35).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Separator(panel_izquierdo, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # === SECCIÓN 3: MODELOS DE COLOR ===
        ttk.Label(panel_izquierdo, text="3. Modelos de Color", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        frame_modelos = ttk.Frame(panel_izquierdo)
        frame_modelos.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(frame_modelos, text="CMY", command=lambda: self.convertir_modelo('CMY'),
                  width=11).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(frame_modelos, text="YIQ", command=lambda: self.convertir_modelo('YIQ'),
                  width=11).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(frame_modelos, text="HSI", command=lambda: self.convertir_modelo('HSI'),
                  width=11).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(frame_modelos, text="HSV", command=lambda: self.convertir_modelo('HSV'),
                  width=11).grid(row=1, column=1, padx=2, pady=2)
        
        ttk.Separator(panel_izquierdo, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # === SECCIÓN 4: BINARIZACIÓN ===
        ttk.Label(panel_izquierdo, text="4. Binarización", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        # Umbral fijo
        ttk.Label(panel_izquierdo, text="Umbral Fijo:").pack(pady=(10, 2))
        
        frame_umbral = ttk.Frame(panel_izquierdo)
        frame_umbral.pack(fill=tk.X, padx=10, pady=5)
        
        self.var_umbral = tk.IntVar(value=127)
        self.label_valor_umbral = ttk.Label(frame_umbral, text="127")
        self.label_valor_umbral.pack(side=tk.RIGHT)
        
        scale_umbral = ttk.Scale(frame_umbral, from_=0, to=255, 
                                variable=self.var_umbral, orient=tk.HORIZONTAL,
                                command=self.actualizar_label_umbral)
        scale_umbral.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(panel_izquierdo, text="Aplicar Umbral Fijo",
                  command=self.aplicar_umbral_fijo, width=35).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Separator(panel_izquierdo, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Umbral automático (Otsu)
        ttk.Label(panel_izquierdo, text="Umbral Automático (Otsu):").pack(pady=(5, 2))
        
        ttk.Button(panel_izquierdo, text="Aplicar Umbral Automático",
                  command=self.aplicar_umbral_otsu, width=35).pack(fill=tk.X, padx=10, pady=5)
        
        self.label_umbral_calculado = ttk.Label(panel_izquierdo, text="", 
                                               font=('Arial', 9, 'italic'))
        self.label_umbral_calculado.pack(pady=5)
        
        ttk.Separator(panel_izquierdo, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # === SECCIÓN 5: HISTOGRAMAS ===
        ttk.Label(panel_izquierdo, text="5. Análisis de Histogramas", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        ttk.Button(panel_izquierdo, text="Histograma de Grises",
                  command=self.mostrar_histograma_grises, width=35).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(panel_izquierdo, text="Histogramas RGB",
                  command=self.mostrar_histogramas_rgb, width=35).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(panel_izquierdo, text="Propiedades del Histograma",
                  command=self.mostrar_propiedades_histograma, width=35).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Separator(panel_izquierdo, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # Panel derecho - Visualización
        panel_derecho = ttk.LabelFrame(frame_contenido, text="Visualización", padding=10)
        panel_derecho.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.frame_visualizacion = ttk.Frame(panel_derecho)
        self.frame_visualizacion.pack(fill=tk.BOTH, expand=True)
        
        # Labels para imágenes
        self.label_original = ttk.Label(self.frame_visualizacion)
        self.label_procesada = ttk.Label(self.frame_visualizacion)
        self.frame_histograma = ttk.Frame(self.frame_visualizacion)
        
        self.actualizar_layout('simple')
    
    def actualizar_layout(self, modo='simple'):
        """Actualiza el layout de visualización"""
        for widget in self.frame_visualizacion.winfo_children():
            widget.grid_forget()
        
        if modo == 'simple':
            self.label_procesada.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
            self.frame_visualizacion.grid_rowconfigure(0, weight=1)
            self.frame_visualizacion.grid_columnconfigure(0, weight=1)
        
        elif modo == 'comparacion':
            ttk.Label(self.frame_visualizacion, text="Original", 
                     font=('Arial', 11, 'bold')).grid(row=0, column=0, pady=5)
            self.label_original.grid(row=1, column=0, padx=10, pady=5)
            
            ttk.Label(self.frame_visualizacion, text="Procesada", 
                     font=('Arial', 11, 'bold')).grid(row=0, column=1, pady=5)
            self.label_procesada.grid(row=1, column=1, padx=10, pady=5)
            
            self.frame_visualizacion.grid_rowconfigure(1, weight=1)
            self.frame_visualizacion.grid_columnconfigure(0, weight=1)
            self.frame_visualizacion.grid_columnconfigure(1, weight=1)
        
        elif modo == 'con_histograma':
            ttk.Label(self.frame_visualizacion, text="Original", 
                     font=('Arial', 11, 'bold')).grid(row=0, column=0, pady=5)
            self.label_original.grid(row=1, column=0, padx=10, pady=5)
            
            ttk.Label(self.frame_visualizacion, text="Procesada", 
                     font=('Arial', 11, 'bold')).grid(row=0, column=1, pady=5)
            self.label_procesada.grid(row=1, column=1, padx=10, pady=5)
            
            ttk.Label(self.frame_visualizacion, text="Histogramas", 
                     font=('Arial', 11, 'bold')).grid(row=2, column=0, columnspan=2, pady=(15, 5))
            self.frame_histograma.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky='nsew')
            
            self.frame_visualizacion.grid_rowconfigure(1, weight=2)
            self.frame_visualizacion.grid_rowconfigure(3, weight=1)
            self.frame_visualizacion.grid_columnconfigure(0, weight=1)
            self.frame_visualizacion.grid_columnconfigure(1, weight=1)
    
    def actualizar_label_umbral(self, valor):
        """Actualiza el label con el valor del umbral"""
        self.label_valor_umbral.config(text=str(int(float(valor))))
    
    def cargar_imagen(self):
        """Carga una imagen desde el disco"""
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), 
                      ("Todos los archivos", "*.*")]
        )
        
        if ruta:
            self.imagen_original = cv2.imread(ruta)
            if self.imagen_original is None:
                messagebox.showerror("Error", "No se pudo cargar la imagen")
                return
            
            # Resetear estado
            self.imagen_grises = None
            self.imagen_procesada = None
            self.label_umbral_calculado.config(text="")
            
            # Limpiar frame de histogramas
            for widget in self.frame_histograma.winfo_children():
                widget.destroy()
            
            # Actualizar layout y mostrar imagen original
            self.actualizar_layout('simple')
            self.mostrar_imagen(self.imagen_original, self.label_procesada)
            
            # Forzar actualización de la interfaz
            self.root.update_idletasks()
            
            messagebox.showinfo("Éxito", "Imagen cargada correctamente")
    
    def mostrar_imagen(self, imagen, label, tamano_max=(600, 500)):
        """Muestra una imagen en un label de Tkinter"""
        if imagen is None:
            return
        
        # Convertir a RGB para mostrar
        if len(imagen.shape) == 3:
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        else:
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
        
        # Redimensionar si es necesario
        h, w = imagen_rgb.shape[:2]
        factor = min(tamano_max[0]/w, tamano_max[1]/h)
        if factor < 1:
            nuevo_w, nuevo_h = int(w*factor), int(h*factor)
            imagen_rgb = cv2.resize(imagen_rgb, (nuevo_w, nuevo_h))
        
        # Convertir a formato Tkinter
        imagen_pil = Image.fromarray(imagen_rgb)
        imagen_tk = ImageTk.PhotoImage(imagen_pil)
        
        label.config(image=imagen_tk)
        label.image = imagen_tk
    
    def convertir_a_grises(self):
        """Convierte la imagen cargada a escala de grises"""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            self.imagen_grises = rgb_a_grises(self.imagen_original)
            self.imagen_procesada = self.imagen_grises
            
            self.actualizar_layout('comparacion')
            self.mostrar_imagen(self.imagen_original, self.label_original)
            self.mostrar_imagen(self.imagen_grises, self.label_procesada)
            
            messagebox.showinfo("Éxito", "Imagen convertida a escala de grises")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al convertir: {str(e)}")
    
    def aplicar_umbral_fijo(self):
        """Aplica binarización con umbral fijo"""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            # Usar imagen en grises si existe, sino convertir
            img_entrada = self.imagen_grises if self.imagen_grises is not None else self.imagen_original
            
            umbral = self.var_umbral.get()
            img_binaria, umbral_usado = binarizacion_umbral_fijo(img_entrada, umbral)
            
            self.imagen_procesada = img_binaria
            
            self.actualizar_layout('comparacion')
            self.mostrar_imagen(img_entrada, self.label_original)
            self.mostrar_imagen(img_binaria, self.label_procesada)
            
            self.label_umbral_calculado.config(text=f"Umbral aplicado: {umbral_usado}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en binarización: {str(e)}")
    
    def aplicar_umbral_otsu(self):
        """Aplica binarización con umbral automático (Otsu)"""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            # Usar imagen en grises si existe, sino convertir
            img_entrada = self.imagen_grises if self.imagen_grises is not None else self.imagen_original
            
            img_binaria, umbral_calculado = binarizacion_umbral_otsu(img_entrada)
            
            self.imagen_procesada = img_binaria
            
            self.actualizar_layout('comparacion')
            self.mostrar_imagen(img_entrada, self.label_original)
            self.mostrar_imagen(img_binaria, self.label_procesada)
            
            self.label_umbral_calculado.config(
                text=f"Umbral automático calculado: {umbral_calculado}",
                foreground="green"
            )
            
            # Actualizar el slider al valor calculado
            self.var_umbral.set(umbral_calculado)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en binarización automática: {str(e)}")
    
    def mostrar_histogramas(self):
        """Muestra los histogramas de las imágenes"""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            # Limpiar frame de histograma
            for widget in self.frame_histograma.winfo_children():
                widget.destroy()
            
            # Para calcular histograma, usar grises si existe, sino convertir temporalmente
            img_para_hist = self.imagen_grises if self.imagen_grises is not None else self.imagen_original
            
            # Calcular histogramas
            hist_original = calcular_histograma(img_para_hist)
            
            # Determinar qué imagen mostrar en label_original
            img_mostrar_original = self.imagen_grises if self.imagen_grises is not None else self.imagen_original
            
            # Crear figura
            if self.imagen_procesada is not None:
                # Dos histogramas
                hist_proc = calcular_histograma(self.imagen_procesada)
                
                fig = Figure(figsize=(10, 3), dpi=80)
                
                # Histograma original/grises
                ax1 = fig.add_subplot(121)
                ax1.plot(hist_original, color='blue')
                titulo_orig = 'Histograma Grises' if self.imagen_grises is not None else 'Histograma Original'
                ax1.set_title(titulo_orig)
                ax1.set_xlabel('Intensidad')
                ax1.set_ylabel('Frecuencia')
                ax1.grid(True, alpha=0.3)
                
                # Histograma procesado
                ax2 = fig.add_subplot(122)
                ax2.plot(hist_proc, color='red')
                ax2.set_title('Histograma Procesado')
                ax2.set_xlabel('Intensidad')
                ax2.set_ylabel('Frecuencia')
                ax2.grid(True, alpha=0.3)
                
                self.actualizar_layout('con_histograma')
                self.mostrar_imagen(img_mostrar_original, self.label_original)
                self.mostrar_imagen(self.imagen_procesada, self.label_procesada)
            else:
                # Solo un histograma
                fig = Figure(figsize=(10, 3), dpi=80)
                
                ax = fig.add_subplot(111)
                ax.plot(hist_original, color='blue')
                titulo = 'Histograma de Intensidad (Grises)' if self.imagen_grises is not None else 'Histograma de Intensidad'
                ax.set_title(titulo)
                ax.set_xlabel('Intensidad (0-255)')
                ax.set_ylabel('Frecuencia (Número de píxeles)')
                ax.grid(True, alpha=0.3)
                
                self.actualizar_layout('con_histograma')
                self.mostrar_imagen(img_mostrar_original, self.label_original)
                self.mostrar_imagen(img_mostrar_original, self.label_procesada)
            
            fig.tight_layout()
            
            # Mostrar en el frame
            canvas = FigureCanvasTkAgg(fig, master=self.frame_histograma)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar histogramas: {str(e)}")
    
    def guardar_imagen(self):
        """Guarda la imagen procesada"""
        if self.imagen_procesada is None:
            messagebox.showwarning("Advertencia", "No hay imagen procesada para guardar")
            return
        
        ruta = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos", "*.*")]
        )
        
        if ruta:
            cv2.imwrite(ruta, self.imagen_procesada)
            messagebox.showinfo("Éxito", "Imagen guardada correctamente")
    
    def reiniciar(self):
        """Reinicia el estado de la aplicación"""
        self.imagen_original = None
        self.imagen_grises = None
        self.imagen_procesada = None
        
        self.label_original.config(image='')
        self.label_procesada.config(image='')
        self.label_umbral_calculado.config(text="")
        
        for widget in self.frame_histograma.winfo_children():
            widget.destroy()
        
        self.actualizar_layout('simple')
    
    def separar_canales(self):
        """Separa y visualiza los canales RGB de la imagen"""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        if len(self.imagen_original.shape) != 3:
            messagebox.showwarning("Advertencia", "La imagen debe ser a color (RGB)")
            return
        
        try:
            canales = separar_canales_rgb_visualizar(self.imagen_original)
            
            # Crear ventana con los canales
            ventana = tk.Toplevel(self.root)
            ventana.title("Separación de Canales RGB")
            ventana.geometry("1200x800")
            
            fig = Figure(figsize=(12, 8))
            
            # Canal Rojo
            ax1 = fig.add_subplot(2, 3, 1)
            ax1.imshow(canales['r_bn'], cmap='gray')
            ax1.set_title('Canal R (Blanco y Negro)', fontweight='bold')
            ax1.axis('off')
            
            ax2 = fig.add_subplot(2, 3, 4)
            ax2.imshow(cv2.cvtColor(canales['r_color'], cv2.COLOR_BGR2RGB))
            ax2.set_title('Canal R (Coloreado)', fontweight='bold')
            ax2.axis('off')
            
            # Canal Verde
            ax3 = fig.add_subplot(2, 3, 2)
            ax3.imshow(canales['g_bn'], cmap='gray')
            ax3.set_title('Canal G (Blanco y Negro)', fontweight='bold')
            ax3.axis('off')
            
            ax4 = fig.add_subplot(2, 3, 5)
            ax4.imshow(cv2.cvtColor(canales['g_color'], cv2.COLOR_BGR2RGB))
            ax4.set_title('Canal G (Coloreado)', fontweight='bold')
            ax4.axis('off')
            
            # Canal Azul
            ax5 = fig.add_subplot(2, 3, 3)
            ax5.imshow(canales['b_bn'], cmap='gray')
            ax5.set_title('Canal B (Blanco y Negro)', fontweight='bold')
            ax5.axis('off')
            
            ax6 = fig.add_subplot(2, 3, 6)
            ax6.imshow(cv2.cvtColor(canales['b_color'], cv2.COLOR_BGR2RGB))
            ax6.set_title('Canal B (Coloreado)', fontweight='bold')
            ax6.axis('off')
            
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=ventana)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al separar canales: {str(e)}")
    
    def convertir_modelo(self, modelo):
        """Convierte la imagen a diferentes modelos de color"""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        if len(self.imagen_original.shape) != 3:
            messagebox.showwarning("Advertencia", "La imagen debe ser a color (RGB)")
            return
        
        try:
            if modelo == 'CMY':
                img_convertida = rgb_a_cmy(self.imagen_original)
                titulo = "Modelo CMY"
                nombres_canales = ['Cyan', 'Magenta', 'Yellow']
            elif modelo == 'YIQ':
                img_convertida = rgb_a_yiq(self.imagen_original)
                titulo = "Modelo YIQ"
                nombres_canales = ['Y (Luminancia)', 'I (Crominancia)', 'Q (Crominancia)']
            elif modelo == 'HSI':
                img_convertida = rgb_a_hsi(self.imagen_original)
                titulo = "Modelo HSI"
                nombres_canales = ['Hue (Matiz)', 'Saturation (Saturación)', 'Intensity (Intensidad)']
            elif modelo == 'HSV':
                img_convertida = rgb_a_hsv(self.imagen_original)
                titulo = "Modelo HSV"
                nombres_canales = ['Hue (Matiz)', 'Saturation (Saturación)', 'Value (Valor)']
            
            # Crear ventana de visualización
            ventana = tk.Toplevel(self.root)
            ventana.title(f"Conversión a {modelo}")
            ventana.geometry("1200x600")
            
            fig = Figure(figsize=(12, 6))
            
            # Imagen original
            ax1 = fig.add_subplot(2, 4, 1)
            ax1.imshow(cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2RGB))
            ax1.set_title('Original RGB', fontweight='bold')
            ax1.axis('off')
            
            # Imagen convertida
            ax2 = fig.add_subplot(2, 4, 2)
            # Para mostrar, convertir de vuelta a RGB si es necesario
            if modelo in ['CMY', 'YIQ']:
                ax2.imshow(img_convertida)
            else:
                ax2.imshow(cv2.cvtColor(img_convertida, cv2.COLOR_HSV2RGB))
            ax2.set_title(f'{titulo} (Compuesto)', fontweight='bold')
            ax2.axis('off')
            
            # Separar canales
            c1, c2, c3 = cv2.split(img_convertida)
            
            # Canal 1
            ax3 = fig.add_subplot(2, 4, 5)
            ax3.imshow(c1, cmap='gray')
            ax3.set_title(nombres_canales[0], fontweight='bold')
            ax3.axis('off')
            
            # Canal 2
            ax4 = fig.add_subplot(2, 4, 6)
            ax4.imshow(c2, cmap='gray')
            ax4.set_title(nombres_canales[1], fontweight='bold')
            ax4.axis('off')
            
            # Canal 3
            ax5 = fig.add_subplot(2, 4, 7)
            ax5.imshow(c3, cmap='gray')
            ax5.set_title(nombres_canales[2], fontweight='bold')
            ax5.axis('off')
            
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=ventana)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al convertir a {modelo}: {str(e)}")
    
    def mostrar_histograma_grises(self):
        """Muestra el histograma de la imagen en escala de grises"""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            # Convertir a grises si es necesario
            if self.imagen_grises is not None:
                img_grises = self.imagen_grises
            else:
                img_grises = rgb_a_grises(self.imagen_original)
            
            hist = calcular_histograma(img_grises)
            
            # Limpiar frame
            for widget in self.frame_histograma.winfo_children():
                widget.destroy()
            
            fig = Figure(figsize=(10, 3), dpi=80)
            ax = fig.add_subplot(111)
            ax.plot(hist, color='blue')
            ax.set_title('Histograma de Intensidad (Escala de Grises)', fontweight='bold')
            ax.set_xlabel('Intensidad (0-255)')
            ax.set_ylabel('Frecuencia (Número de píxeles)')
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            
            self.actualizar_layout('con_histograma')
            self.mostrar_imagen(img_grises, self.label_original)
            self.mostrar_imagen(img_grises, self.label_procesada)
            
            canvas = FigureCanvasTkAgg(fig, master=self.frame_histograma)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar histograma: {str(e)}")
    
    def mostrar_histogramas_rgb(self):
        """Muestra los histogramas separados de cada canal RGB"""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        if len(self.imagen_original.shape) != 3:
            messagebox.showwarning("Advertencia", "La imagen debe ser a color (RGB)")
            return
        
        try:
            hists = calcular_histogramas_rgb(self.imagen_original)
            
            # Crear ventana
            ventana = tk.Toplevel(self.root)
            ventana.title("Histogramas RGB")
            ventana.geometry("1000x700")
            
            fig = Figure(figsize=(10, 7))
            
            # Imagen original
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.imshow(cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2RGB))
            ax1.set_title('Imagen Original', fontweight='bold')
            ax1.axis('off')
            
            # Histograma R
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.plot(hists['R'], color='red', label='Canal R')
            ax2.set_title('Histograma Canal Rojo', fontweight='bold')
            ax2.set_xlabel('Intensidad')
            ax2.set_ylabel('Frecuencia')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Histograma G
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.plot(hists['G'], color='green', label='Canal G')
            ax3.set_title('Histograma Canal Verde', fontweight='bold')
            ax3.set_xlabel('Intensidad')
            ax3.set_ylabel('Frecuencia')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Histograma B
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.plot(hists['B'], color='blue', label='Canal B')
            ax4.set_title('Histograma Canal Azul', fontweight='bold')
            ax4.set_xlabel('Intensidad')
            ax4.set_ylabel('Frecuencia')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=ventana)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar histogramas RGB: {str(e)}")
    
    def mostrar_propiedades_histograma(self):
        """Muestra las propiedades estadísticas del histograma"""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            # Usar grises si existe
            if self.imagen_grises is not None:
                img_grises = self.imagen_grises
            else:
                img_grises = rgb_a_grises(self.imagen_original)
            
            hist = calcular_histograma(img_grises)
            props = propiedades_histograma(hist)
            
            if props is None:
                messagebox.showerror("Error", "No se pudieron calcular las propiedades")
                return
            
            # Crear ventana con propiedades
            ventana = tk.Toplevel(self.root)
            ventana.title("Propiedades del Histograma")
            ventana.geometry("900x700")
            
            # Frame superior - Imagen y tabla
            frame_sup = ttk.Frame(ventana, padding=10)
            frame_sup.pack(fill=tk.BOTH, expand=True)
            
            # Imagen
            frame_img = ttk.LabelFrame(frame_sup, text="Imagen", padding=10)
            frame_img.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
            
            img_rgb = cv2.cvtColor(img_grises, cv2.COLOR_GRAY2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil = img_pil.resize((300, 300), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_pil)
            label_img = ttk.Label(frame_img, image=img_tk)
            label_img.image = img_tk
            label_img.pack()
            
            # Tabla de propiedades
            frame_props = ttk.LabelFrame(frame_sup, text="Propiedades Estadísticas", padding=10)
            frame_props.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
            
            # Crear tabla
            columns = ('Propiedad', 'Valor')
            tree = ttk.Treeview(frame_props, columns=columns, show='headings', height=10)
            
            tree.heading('Propiedad', text='Propiedad')
            tree.heading('Valor', text='Valor')
            
            tree.column('Propiedad', width=200)
            tree.column('Valor', width=150)
            
            # Insertar datos
            tree.insert('', tk.END, values=('Media', f"{props['media']:.2f}"))
            tree.insert('', tk.END, values=('Mediana', f"{props['mediana']}"))
            tree.insert('', tk.END, values=('Moda', f"{props['moda']}"))
            tree.insert('', tk.END, values=('Varianza', f"{props['varianza']:.2f}"))
            tree.insert('', tk.END, values=('Desviación Estándar', f"{props['desviacion_estandar']:.2f}"))
            tree.insert('', tk.END, values=('Valor Mínimo', f"{props['minimo']}"))
            tree.insert('', tk.END, values=('Valor Máximo', f"{props['maximo']}"))
            tree.insert('', tk.END, values=('Rango', f"{props['rango']}"))
            tree.insert('', tk.END, values=('Total Píxeles', f"{props['total_pixeles']:,}"))
            
            tree.pack(fill=tk.BOTH, expand=True)
            
            # Frame inferior - Histograma
            frame_inf = ttk.LabelFrame(ventana, text="Histograma", padding=10)
            frame_inf.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            fig = Figure(figsize=(8, 3))
            ax = fig.add_subplot(111)
            ax.plot(hist, color='blue')
            ax.axvline(x=props['media'], color='red', linestyle='--', label=f"Media: {props['media']:.1f}")
            ax.axvline(x=props['mediana'], color='green', linestyle='--', label=f"Mediana: {props['mediana']}")
            ax.axvline(x=props['moda'], color='orange', linestyle='--', label=f"Moda: {props['moda']}")
            ax.set_title('Histograma con Medidas de Tendencia Central', fontweight='bold')
            ax.set_xlabel('Intensidad')
            ax.set_ylabel('Frecuencia')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=frame_inf)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar propiedades: {str(e)}")


def main():
    root = tk.Tk()
    app = InterfazImagenDigital(root)
    root.mainloop()


if __name__ == "__main__":
    main()
