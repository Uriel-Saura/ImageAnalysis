"""
Interfaz Gráfica para Técnicas de Segmentación
Permite visualizar y comparar diferentes métodos de segmentación y preprocesamiento
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Importar módulos propios
from tecnicas_umbralizacion import (
    metodo_otsu, metodo_entropia_kapur, metodo_minimo_histograma,
    metodo_media, metodo_multiumbral, umbral_por_banda,
    umbral_adaptativo_media, umbral_adaptativo_gaussiano
)
from tecnicas_ecualizacion import (
    ecualizacion_uniforme, ecualizacion_exponencial, ecualizacion_rayleigh,
    ecualizacion_hipercubica, ecualizacion_logaritmica_hiperbolica, ecualizacion_clahe
)
from tecnicas_ajuste import (
    funcion_potencia, correccion_gamma, desplazamiento_histograma,
    contraccion_histograma, expansion_histograma, expansion_histograma_percentil,
    ajuste_contraste_brillo, transformacion_logaritmica
)


class InterfazSegmentacion:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Segmentación de Imágenes")
        self.root.geometry("1600x900")
        
        self.imagen_original = None
        self.imagen_procesada = None
        
        self.crear_interfaz()
    
    def crear_interfaz(self):
        """Crea la interfaz gráfica principal"""
        
        # Frame principal con scroll
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ==================== PANEL SUPERIOR: CARGA Y CONTROLES ====================
        panel_superior = ttk.LabelFrame(main_container, text="Control Principal", padding=10)
        panel_superior.pack(fill=tk.X, pady=(0, 10))
        
        # Botones de carga
        frame_botones = ttk.Frame(panel_superior)
        frame_botones.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(frame_botones, text="📁 Cargar Imagen", 
                  command=self.cargar_imagen).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botones, text="💾 Guardar Resultado", 
                  command=self.guardar_imagen).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botones, text="🔄 Reiniciar", 
                  command=self.reiniciar).pack(side=tk.LEFT, padx=5)
        
        # ==================== PANEL IZQUIERDO: TÉCNICAS ====================
        panel_izquierdo = ttk.Frame(main_container)
        panel_izquierdo.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        # Notebook con tabs para diferentes categorías
        self.notebook = ttk.Notebook(panel_izquierdo)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Preprocesamiento
        self.creaUmbralización
        self.crear_tab_umbralizacion()
        
        # Tab 2: Ecualización
        self.crear_tab_ecualizacion()
        
        # Tab 3: Ajustes
        self.crear_tab_ajustes()
        
        # Tab 4ear_tab_comparacion()
        
        # ==================== PANEL DERECHO: VISUALIZACIÓN ====================
        panel_derecho = ttk.Frame(main_container)
        panel_derecho.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Frame para imágenes
        self.frame_imagenes = ttk.LabelFrame(panel_derecho, text="Visualización", padding=10)
        self.frame_imagenes.pack(fill=tk.BOTH, expand=True)
        
        # Canvas para mostrar imágenes
        self.crear_canvas_imagenes()
    
    def crear_tab_preprocesamiento(self):
        """Crea la pestaña de preprocesamiento"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔧 Preprocesamiento")
    self.notebook.add(tab, text="🎯 Umbralización")
        
        # Scroll
        canvas = tk.Canvas(tab, width=350)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Métodos automáticos
        ttk.Label(scrollable_frame, text="Métodos Automáticos:", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        ttk.Button(scrollable_frame, text="Método de Otsu",
                  command=lambda: self.aplicar_umbralizacion('otsu')).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="Entropía de Kapur",
                  command=lambda: self.aplicar_umbralizacion('kapur')).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="Mínimo de Histogramas",
                  command=lambda: self.aplicar_umbralizacion('minimo_hist')).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="Método de la Media",
                  command=lambda: self.aplicar_umbralizacion('media')).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Multiumbralización
        ttk.Label(scrollable_frame, text="Multiumbralización:", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        ttk.Label(scrollable_frame, text="Número de umbrales:").pack(pady=2)
        self.var_num_umbrales = tk.IntVar(value=2)
        ttk.Spinbox(scrollable_frame, from_=2, to=5, textvariable=self.var_num_umbrales, 
                   width=10).pack(pady=2)
        
        ttk.Button(scrollable_frame, text="Aplicar Multiumbralización",
                  command=self.aplicar_multiumbral).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Umbralización por banda
        ttk.Label(scrollable_frame, text="Umbralización por Banda:", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        ttk.Label(scrollable_frame, text="Umbral mínimo:").pack(pady=2)
        self.var_umbral_min = tk.IntVar(value=100)
        ttk.Scale(scrollable_frame, from_=0, to=255, variable=self.var_umbral_min, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)
        
        ttk.Label(scrollable_frame, text="Umbral máximo:").pack(pady=2)
        self.var_umbral_max = tk.IntVar(value=200)
        ttk.Scale(scrollable_frame, from_=0, to=255, variable=self.var_umbral_max, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)
        
        ttk.Button(scrollable_frame, text="Aplicar Banda",
                  command=self.aplicar_umbral_banda).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Umbralización adaptativa
        ttk.Label(scrollable_frame, text="Umbralización Adaptativa:", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        ttk.Button(scrollable_frame, text="Adaptativa por Media",
                  command=lambda: sel1️⃣aplicar_umbralizacion('adaptativa_media')).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="Adaptativa Gaussiana",
                  command=lambda: self.aplicar_umbralizacion('adaptativa_gauss')).pack(fill=tk.X, padx=10, pady=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def crear_tab_ecualizacion(self):
        """Crea la pestaña de técnicas de ecualización"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📊 Ecualización")
        
        canvas = tk.Canvas(tab, width=350)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        ttk.Label(scrollable_frame, text="Métodos de Ecualización:", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        ttk.Button(scrollable_frame, text="Ecualización Uniforme",
                  command=lambda: self.aplicar_ecualizacion('uniforme')).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="Ecualización Exponencial",
                  command=lambda: self.aplicar_ecualizacion('exponencial')).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="Ecualización Rayleigh",
                  command=lambda: self.aplicar_ecualizacion('rayleigh')).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="Ecualización Hipercúbica",
                  command=lambda: self.aplicar_ecualizacion('hipercubica')).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="Ecualización Logarítmica Hiperbólica",
                  command=lambda: self.aplicar_ecualizacion('log_hiperbolica')).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="CLAHE (Adaptativa)",
                  command=lambda: self.aplicar_ecualizacion('clahe')).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Label(scrollable_frame, text="Parámetros:", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        ttk.Label(scrollable_frame, text="Alpha/Parámetro:").pack(pady=2)
        self.var_alpha_ecual = tk.DoubleVar(value=0.5)
        ttk.Scale(scrollable_frame, from_=0.1, to=3.0, variable=self.var_alpha_ecual, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def crear_tab_ajustes(self):
        """Crea la pestaña de ajustes de histograma"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="⚙️ Ajustes")
        
        canvas = tk.Canvas(tab, width=350)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Corrección Gamma
        ttk.Label(scrollable_frame, text="Corrección Gamma:", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        ttk.Label(scrollable_frame, text="Valor Gamma:").pack(pady=2)
        self.var_gamma = tk.DoubleVar(value=1.0)
        ttk.Scale(scrollable_frame, f2️⃣m_=0.1, to=3.0, variable=self.var_gamma, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)
        
        ttk.Button(scrollable_frame, text="Aplicar Gamma",
                  command=self.aplicar_gamma).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Desplazamiento
        ttk.Label(scrollable_frame, text="Desplazamiento de Brillo:", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        ttk.Label(scrollable_frame, text="Desplazamiento:").pack(pady=2)
        self.var_desplazamiento = tk.IntVar(value=0)
        ttk.Scale(scrollable_frame, from_=-100, to=100, variable=self.var_desplazamiento, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)
        
        ttk.Button(scrollable_frame, text="Aplicar Desplazamiento",
                  command=self.aplicar_desplazamiento).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Contracción/Expansión
        ttk.Label(scrollable_frame, text="Contracción del Histograma:", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        ttk.Label(scrollable_frame, text="Factor (0-1):").pack(pady=2)
        self.var_factor_contraccion = tk.DoubleVar(value=0.5)
        ttk.Scale(scrollable_frame, from_=0.0, to=1.0, variable=self.var_factor_contraccion, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)
        
        ttk.Button(scrollable_frame, text="Aplicar Contracción",
                  command=self.aplicar_contraccion).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Label(scrollable_frame, text="Expansión del Histograma:", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        ttk.Button(scrollable_frame, text="Expansión Completa",
                  command=lambda: self.aplicar_ajuste('expansion')).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="Expansión por Percentiles",
                  command=lambda: self.aplicar_ajuste('expansion_percentil')).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Otras transformaciones
        ttk.Label(scrollable_frame, text="Otras Transformaciones:", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        ttk.Button(scrollable_frame, 3️⃣xt="Transformación Logarítmica",
                  command=lambda: self.aplicar_ajuste('logaritmica')).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="Función Potencia",
                  command=self.aplicar_potencia).pack(fill=tk.X, padx=10, pady=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def crear_tab_comparacion(self):
        """Crea la pestaña de comparación"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔍 Comparación")
        
        ttk.Label(tab, text="Comparar Segmentación:", 
                 font=('Arial', 11, 'bold')).pack(pady=20)
        
        ttk.Button(tab, text="Comparar: Original vs Preprocesada",
                  command=lambda: self.comparar_segmentaciones('original_vs_prepro')).pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Button(tab, text="Comparar: Sin Preprocesar vs Con Preprocesar",
                  command=lambda: self.comparar_segmentaciones('efecto_prepro')).pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Button(tab, text="Comparar Todos los Métodos de Umbralización",
                  command=self.comparar_todos_umbrales).pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Button(tab, text="Comparar Todas las Ecualizaciones",
                  command=self.comparar_todas_ecualizaciones).pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Separator(tab, orient='horizontal').pack(fill=tk.X, pady=20)
        
        ttk.Label(tab, text="Información:", 
                 font=('Arial', 10, 'bold')).pack(pady=10)
        
        self.info_text = tk.Text(tab, height=10, width=40, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    def crear_canvas_imagenes(self):
        """Crea el canvas para mostrar las imágenes"""
        # Frame para layout de imágenes
        self.frame_imgs = ttk.Frame(self.frame_imagenes)
        self.frame_imgs.pack(fill=tk.BOTH, expand=True)
        
        # Labels para las imágenes
        self.label_original = ttk.Label(self.frame_imgs)
        self.label_procesada = ttk.Label(self.frame_imgs)
        self.label_histograma = ttk.Label(self.frame_imgs)
        
        # Layout inicial
        self.actualizar_layout_imagenes()
    
    def actualizar_layout_imagenes(self, modo='simple'):
        """Actualiza el layout de visualización de imágenes"""
        # Limpiar layout actual
        for widget in self.frame_imgs.winfo_children():
            widget.grid_forget()
        
        if modo == 'simple':
            # Mostrar solo imagen procesada grande
            self.label_procesada.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
            self.frame_imgs.grid_rowconfigure(0, weight=1)
            self.frame_imgs.grid_columnconfigure(0, weight=1)
        
        elif modo == 'comparacion':
            # Mostrar original y procesada lado a lado
            ttk.Label(self.frame_imgs, text="Original", font=('Arial', 10, 'bold')).grid(row=0, column=0)
            self.label_original.grid(row=1, column=0, padx=5, pady=5)
            
            ttk.Label(self.frame_imgs, text="Procesada", font=('Arial', 10, 'bold')).grid(row=0, column=1)
            self.label_procesada.grid(row=1, column=1, padx=5, pady=5)
            
            self.frame_imgs.grid_rowconfigure(1, weight=1)
            self.frame_imgs.grid_columnconfigure(0, weight=1)
            self.frame_imgs.grid_columnconfigure(1, weight=1)
        
        elif modo == 'histograma':
            # Mostrar imagen y histograma
            self.label_procesada.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
            self.label_histograma.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
            
            self.frame_imgs.grid_rowconfigure(0, weight=1)
            self.frame_imgs.grid_columTécnicas:", 
                 font=('Arial', 11, 'bold')).pack(pady=20)
        
        ttk.Button(tab, text="Comparar Todos los Métodos de Umbralización",
                  command=self.comparar_todos_umbrales).pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(tab, text="Comparar Todas las Ecualizaciones",
                  command=self.comparar_todas_ecualizaciones).pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(tab, text="Comparar Todos los Ajustes",
                  command=self.comparar_todos_ajustes).pack(fill=tk.X, padx=20, pady=10
                messagebox.showerror("Error", "No se pudo cargar la imagen")
                return
            
            self.imagen_preprocesada = None
            self.mostrar_imagen(self.imagen_original, self.label_procesada)
            self.actualizar_layout_imagenes('simple')
            messagebox.showinfo("Éxito", "Imagen cargada correctamente")
    
    def mostrar_imagen(self, imagen, label, tamano_max=(700, 700)):
        """Muestra una imagen en un label de Tkinter"""
        if imagen is None:
            return
        
        # Convertir de BGR a RGB
        if len(imagen.shape) == 3:
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        else:
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
        
        # Redimensionar manteniendo proporción
        h, w = imagen_rgb.shape[:2]
        factor = min(tamano_max[0]/w, tamano_max[1]/h)
        if factor < 1:
            nuevo_w, nuevo_h = int(w*factor), int(h*factor)
            imagen_rgb = cv2.resize(imagen_rgb, (nuevo_w, nuevo_h))
        
        # Convertir a ImageTk
        imagen_pil = Image.fromarray(imagen_rgb)
        imagen_tk = ImageTk.PhotoImage(imagen_pil)
        
        label.config(image=imagen_tk)
        label.image = imagen_tk  # Mantener referencia
    
    def mostrar_histograma(self, imagen, label):
        """Muestra el histograma de una imagen"""
        if imagen is None:
            return
        
        if len(imagen.shape) == 3:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        fig = Figure(figsize=(4, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        hist = cv2.calcHist([imagen], [0], None, [256], [0, 256])
        ax.plot(hist, color='black')
        ax.set_xlim([0, 256])
        ax.set_xlabel('Nivel de Gris')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Histograma')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        # Convertir a imagen para mostrar en Tkinter
        canvas = FigureCanvasTkAgg(fig, master=label)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def aplicar_preprocesamiento(self, metodo):
        """Aplica un método de preprocesamiento"""
        if self.imagen_original is None:
    
        if imagen_base is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            if metodo == 'otsu':
                umbral, resultado = metodo_otsu(imagen_base)
                info = f"Umbral Otsu: {umbral}"
            elif metodo == 'kapur':
                umbral, resultado = metodo_entropia_kapur(imagen_base)
                info = f"Umbral Kapur: {umbral}"
            elif metodo == 'minimo_hist':
                umbral, resultado = metodo_minimo_histograma(imagen_base)
                info = f"Umbral Mínimo: {umbral}"
            elif metodo == 'media':
                umbral, resultado = metodo_media(imagen_base)
                info = f"Umbral Media: {umbral}"
            elif metodo == 'adaptativa_media':
                resultado = umbral_adaptativo_media(imagen_base)
                info = "Umbralización Adaptativa por Media"
            elif metodo == 'adaptativa_gauss':
                resultado = umbral_adaptativo_gaussiano(imagen_base)
                info = "Umbralización Adaptativa Gaussiana"
            
            self.imagen_procesada = resultado
            self.actualizar_layout_imagenes('comparacion')
            self.mostrar_imagen(imagen_base, self.label_original)
            self.mostrar_imagen(resultado, self.label_procesada)
            
            self.info_text.delete('1.0', tk.END)
            self.info_text.insert('1.0', info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en umbralización: {str(e)}")
    
    def aplicar_multiumbral(self):
        """Aplica multiumbralización"""
        imagen_base = self.imagen_preprocesada if self.imagen_preprocesada is not None else self.imagen_original
        
        if imagen_base is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            num_umbrales = self.var_num_umbrales.get()
            umbrales, resultado = metodo_multiumbral(imagen_base, num_umbrales)
            
            info = f"Multiumbralización con {num_umbrales} umbrales:\n"
            info += f"Umbrales: {umbrales}"
            
            self.imagen_procesada = resultado
            self.actualizar_layout_imagenes('comparacion')
            self.mostrar_imagen(imagen_base, self.label_original)
            self.mostrar_imagen(resultado, self.label_procesada)
            
            self.info_text.delete('1.0', tk.END)
            self.info_text.insert('1.0', info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en multiumbralización: {str(e)}")
    
    def aplicar_umbral_banda(self)
        """Aplica umbralización por banda"""
        imagen_base = self.imagen_preprocesada if self.imagen_preprocesada is not None else self.imagen_original
        
        if imagen_base is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            umbral_min = self.var_umbral_min.get()
            umbral_max = self.var_umbral_max.get()
            
            resultado = umbral_por_banda(imagen_base, umbral_min, umbral_max)
            
            info = f"Umbralización por banda:\nMín: {umbral_min}, Máx: {umbral_max}"
            
            self.imagen_procesada = resultado
            self.actualizar_layout_imagenes('comparacion')
            self.mostrar_imagen(imagen_base, self.label_original)
            self.mostrar_imagen(resultado, self.label_procesada)
            
            self.info_text.delete('1.0', tk.END)
            self.info_text.insert('1.0', info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en banda: {str(e)}")
    
    def aplicar_ecualizacion(self, metodo):
        """Aplica un método de ecualización"""
        imagen_base = self.imagen_preprocesada if self.imagen_preprocesada is not None else self.imagen_original
        
        if imagen_base is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            alpha = self.var_alpha_ecual.get()
            
            if metodo == 'uniforme':
                resultado, hist_or
            elif metodo == 'exponencial':
                resultado, hist_orig, hist_ecual = ecualizacion_exponencial(imagen_base, alpha)
            elif metodo == 'rayleigh':
                resultado, hist_orig, hist_ecual = ecualizacion_rayleigh(imagen_base, alpha)
            elif metodo == 'hipercubica':
                resultado, hist_orig, hist_ecual = ecualizacion_hipercubica(imagen_base, alpha)
            elif metodo == 'log_hiperbolica':
                resultado, hist_orig, hist_ecual = ecualizacion_logaritmica_hiperbolica(imagen_base, alpha)
            elif metodo == 'clahe':
                resultado, hist_orig, hist_ecual = ecualizacion_clahe(imagen_base)
            
            self.imagen_procesada = resultado
            self.actualizar_layout_imagenes('comparacion')
            self.mostrar_imagen(imagen_base, self.label_original)
            self.mostrar_imagen(resultado, self.label_procesada)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en ecualización: {str(e)}")
    
    def aplicar_gamma(self):
        """Aplica corrección gamma"""
        imagen_base = self.imagen_preprocesada if self.imagen_preprocesada is not None else self.imagen_original
        
        if imagen_base is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            gamma = self.var_gamma.get()
            resultado, _, _ = correccion_gamma(imagen_base, gamma)
            
            self.imagen_procesada = resultado
            self.actualizar_layout_imagenes('comparacion')
            self.mostrar_imagen(imagen_base, self.label_original)
            self.mostrar_imagen(resultado, self.label_procesada)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en gamma: {str(e)}")
    
    def aplicar_desplazamiento(self):
        """Aplica desplazamiento de histograma"""
        imagen_base = self.imagen_preprocesada if self.imagen_preprocesada is not None else self.imagen_original
        
        if imagen_base is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            desp = self.var_desplazamiento.get()
            resultado, _, _ = desplazamiento_histograma(imagen_base, desp)
            
            self.imagen_procesada = resultado
            self.actualizar_layout
            self.mostrar_imagen(imagen_base, self.label_original)
            self.mostrar_imagen(resultado, self.label_procesada)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en desplazamiento: {str(e)}")
    
    def aplicar_contraccion(self):
        """Aplica contracción de histograma"""
        imagen_base = self.imagen_preprocesada if self.imagen_preprocesada is not None else self.imagen_original
        
        if imagen_base is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            factor = self.var_factor_contraccion.get()
            resultado, _, _ = contraccion_histograma(imagen_base, factor)
            
            self.imagen_procesada = resultado
            self.actualizar_layout_imagenes('comparacion')
            self.mostrar_imagen(imagen_base, self.label_original)
            self.mostrar_imagen(resultado, self.label_procesada)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en contracción: {str(e)}")
    
    def aplicar_ajuste(self, tipo):
        """Aplica otros ajustes"""
        imagen_base = self.imagen_preprocesada if self.imagen_preprocesada is not None else self.imagen_original
        
        if imagen_base is None:
            messagebox.showwarning
            return
        
        try:
            if tipo == 'expansion':
                resultado, _, _ = expansion_histograma(imagen_base)
            elif tipo == 'expansion_percentil':
                resultado, _, _ = expansion_histograma_percentil(imagen_base)
            elif tipo == 'logaritmica':
                resultado, _, _ = transformacion_logaritmica(imagen_base)
            
            self.imagen_procesada = resultado
            self.actualizar_layout_imagenes('comparacion')
            self.mostrar_imagen(imagen_base, self.label_original)
            self.mostrar_imagen(resultado, self.label_procesada)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en ajuste: {str(e)}")
    
    def aplicar_potencia(self):
        """Aplica función potencia
        imagen_base = self.imagen_preprocesada if self.imagen_preprocesada is not None else self.imagen_original
        
        if imagen_base is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            gamma = self.var_gamma.get()
            resultado, _, _ = funcion_potencia(imagen_base, gamma=gamma)
            
            self.imagen_procesada = resultado
            self.actualizar_layout_imagenes('comparacion')
            self.mostrar_imagen(imagen_base, self.label_original)
            self.mostrar_imagen(resultado, self.label_procesada)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en potencia: {str(e)}")
    
    def comparar_segmentaciones(self, tipo):
        """Compara diferentes segm
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        messagebox.showinfo("Comparación", f"Función de comparación: {tipo}\nEn desarrollo...")
    
    def comparar_todos_umbrales(self):
        """Compara todos los métodos de umbralización en una ventana nueva"""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            # Crear nueva ventana
            ventana = tk.Toplevel(self.root)
            ventana.title("Comparación de Métodos de Umbralización")
            ventana.geometry("1400
            
            imagen_base = self.imagen_preprocesada if self.imagen_preprocesada is not None else self.imagen_original
            
            # Aplicar todos los métodos
            _, img_otsu = metodo_otsu(imagen_base)
            _, img_kapur = metodo_entropia_kapur(imagen_base)
            _, img_minimo = metodo_minimo_histograma(imagen_base)
            _, img_media = metodo_media(imagen_base)
            
            # Crear figura con matplotlib
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            fig.suptitle('Comparación de Métodos de Umbralización', fontsize=16)
            
            if len(imagen_base.shape) == 3:
                original_gray = cv2.cvtColor(imagen_base, cv2.COLOR_BGR2GRAY)
            else:
                original_gray = imagen_base
            
            axes[0, 0].imshow(original_gray, cmap='gray')
            axes[0, 0].set_title('Original')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(img_otsu, cmap='gray')
            axes[0, 1].set_title('
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(img_kapur, cmap='gray')
            axes[0, 2].set_title('Kapur')
            axes[0, 2].axis('off')
            
            axes[1, 0].imshow(img_minimo, cmap='gray')
            axes[1, 0].set_title('Mínimo Histograma')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(img_media, cmap='gray')
            axes[1, 1].set_title('Media')
            axes[1, 1].axis('off')
            
            axes[1, 2].axis('off')
            
    except Exception as e:
            messagebox.showerror("Error", f"Error en comparación: {str(e)}")
    
    def comparar_todas_ecualizaciones(self):
        """Compara todos los métodos de ecualización"""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            ventana = tk.Toplevel(self.root)
            ventana.title("Comparación de Métodos de Ecualización")
            ventana.geometry("1400x800")
            
            imagen_base = self.imagen_preprocesada if self.imagen_preprocesada is not None else self.imagen_original
            
            # Aplicar métodos
            img_unif, _, _ = ecualizacion_uniforme(imagen_base)
            img_exp, _, _ = ecualizacion_exponencial(imagen_base)
            img_ray, _, _ = ecualizacion_rayleigh(imagen_base)
            img_hip, _, _ = ecualizacion_hipercubica(imagen_base)
            img_log, _, _ = ecualizaci
            img_clahe, _, _ = ecualizacion_clahe(imagen_base)
            
            fig, axes = plt.subplots(2, 4, figsize=(14, 8))
            fig.suptitle('Comparación de Métodos de Ecualización', fontsize=16)
            
            if len(imagen_base.shape) == 3:
                original_gray = cv2.cvtColor(imagen_base, cv2.COLOR_BGR2GRAY)
            else:
                original_gray = imagen_base
            
            axes[0, 0].imshow(original_gray, cmap='gray')
            axes[0, 0].set_title('Original')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(img_unif, cmap='gray')
            axes[0, 1].set_title('Uniforme')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(img_exp, cmap='gray')
            axes[0, 2].set_title('Exponencial')
            axes[0, 2].axis('off')
            
            axes[0, 3].imshow(img_ray, cmap='gray')
            axes[0, 3].set_title('Rayleigh')
            axes[0, 3].axis('off')
            
            axes[1, 0].imshow(img_hip, cmap='gray')
            axes[1, 0].set_title('Hipercúbica')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(img_log, cmap='gray')
            axes[1, 1].set_title('Log Hiperbólica')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(img_clahe, cmap='gray')
            axes[1, 2].set_title('CLAHE')
            axes[1, 2].axis('off')
            
            axes[1, 3].axis('off')
            
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=ventana)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en comparación: {str(e)}")
    
    def comparar_todos_ajustes(self):
        """Compara todos los métodos de ajuste"""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "No hay imagen procesada para guardar")
            return
        
        ruta = filedialog.asksaveasfilename(
        comparar_todos_ajustes(self):
        """Compara todos los métodos de ajuste"""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            ventana = tk.Toplevel(self.root)
            ventana.title("Comparación de Métodos de Ajuste")
            ventana.geometry("1400x800")
            
            imagen_base = self.imagen_original
            
            # Aplicar métodos
            img_gamma, _, _ = correccion_gamma(imagen_base, 0.5)
            img_desp, _, _ = desplazamiento_histograma(imagen_base, 50)
            img_contr, _, _ = contraccion_histograma(imagen_base, 0.5)
            img_exp, _, _ = expansion_histograma(imagen_base)
            img_pot, _, _ = funcion_ion_logaritmica(imagen_base)
            
            fig, axes = plt.subplots(2, 4, figsize=(14, 8))
            fig.suptitle('Comparación de Métodos de Ajuste', fontsize=16)
            
            if len(imagen_base.shape) == 3:
                original_gray = cv2.cvtColor(imagen_base, cv2.COLOR_BGR2GRAY)
            else:
                original_gray = imagen_base
            
            axes[0, 0].imshow(original_gray, cmap='gray')
            axes[0, 0].set_title('Original')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(img_gamma, cmap='gray')
            axes[0, 1].set_title('Gamma (0.5)')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(img_desp, cmap='gray')
            axes[0, 2].set_title('Desplazamiento')
            axes[0, 2].axis('off')
            
            axes[0, 3].imshow(img_contr, cmap='gray')
            axes[0, 3].set_title('Contracción')
            axes[0, 3].axis('off')
            
            axes[1, 0].imshow(img_exp, cmap='gray')
            axes[1, 0].set_title('Expansión')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(img_pot, cmap='gray')
            axes[1, 1].set_title('Función Potencia')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(img_log, cmap='gray')
            axes[1, 2].set_title('Logarítmica')
            axes[1, 2].axis('off')
            
            axes[1, 3].axis('off')
            
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=ventana)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en comparación: {str(e)}")
    
    def     defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos", "*.*")]
        )
        
        if ruta:
            cv2.imwrite(ruta, self.imagen_procesada)
            messagebox.showinfo("Éxito", "Imagen guardada correctamente")
    
    def reiniciar(self):
        """Reinicia el estado de la aplicación"""
        self.imagen_original = None
        self.imagen_procesada = None
        self.imagen_preprocesada = None
        
        self.label_original.config(image='')
        self.label_procesada.config(image='')
        
        self.info_text.delete('1.0', tk.END)


def main():
    root = tk.Tk()
    app = InterfazSegmentacion(root)
    root.mainloop()


if __name__ == "__main__":
    main()
