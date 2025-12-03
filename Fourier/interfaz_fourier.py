# ===================================================================
# INTERFAZ GRÁFICA PARA TRANSFORMADA DE FOURIER
# Permite cargar imágenes y visualizar el espectro de magnitud y fase
# ===================================================================

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from logica_fft import TransformadaFourier
from visualizador import Visualizador
from filtros_fourier import aplicar_filtro
from analisis_filtros import AnalizadorFiltros
from logica_dct import TransformadaDCT, CompresorDCT


# ===================================================================
# CLASE PRINCIPAL DE LA INTERFAZ
# ===================================================================

class InterfazFourier:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis de Imágenes - Transformada de Fourier")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Configurar el protocolo de cierre de ventana
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar_aplicacion)
        
        # Variables para la imagen
        self.imagen = None
        self.imagen_resultado = None
        self.nombre_operacion = None
        
        # Módulo de FFT
        self.fft = TransformadaFourier()
        
        # Módulo de DCT
        self.dct = TransformadaDCT()
        self.compresor = CompresorDCT()
        
        # Visualizador (se inicializa después de crear la interfaz)
        self.visualizador = None
        
        # Variables para filtros
        self.radio_filtro = tk.IntVar(value=30)
        self.sigma_filtro = tk.IntVar(value=30)
        self.orden_filtro = tk.IntVar(value=2)
        
        # Variables para compresión DCT
        self.umbral_compresion = tk.IntVar(value=50)
        self.num_coeficientes = tk.IntVar(value=15)
        
        # Variables para almacenar último filtro aplicado
        self.ultima_imagen_filtrada = None
        self.ultima_mascara = None
        self.ultimo_nombre_filtro = None
        
        # Diccionario de operaciones disponibles
        self.operaciones = {
            'Espectro de Magnitud': self.mostrar_magnitud,
            'Espectro de Fase': self.mostrar_fase,
            'Magnitud + Fase': self.mostrar_magnitud_fase,
            'FFT Completa': self.mostrar_fft_completa,
        }
        
        # Diccionario de operaciones DCT
        self.operaciones_dct = {
            'DCT Completa': self.mostrar_dct_completa,
            'DCT por Bloques 8x8': self.mostrar_dct_bloques,
        }
        
        # Diccionario de filtros disponibles
        self.filtros = {
            'Ideal Pasa-Bajas': ('ideal_pb', 'radio'),
            'Gaussiano Pasa-Bajas': ('gaussiano_pb', 'sigma'),
            'Butterworth Pasa-Bajas': ('butterworth_pb', 'radio_orden'),
            'Ideal Pasa-Altas': ('ideal_pa', 'radio'),
            'Gaussiano Pasa-Altas': ('gaussiano_pa', 'sigma'),
            'Butterworth Pasa-Altas': ('butterworth_pa', 'radio_orden'),
        }
        
        self.crear_interfaz()
    
    def crear_interfaz(self):
        """Crea la interfaz gráfica completa"""
        
        # ===== FRAME SUPERIOR: TÍTULO =====
        frame_titulo = tk.Frame(self.root, bg='#8e44ad', height=60)
        frame_titulo.pack(fill=tk.X, padx=10, pady=10)
        
        titulo = tk.Label(
            frame_titulo, 
            text="ANALISIS DE IMAGENES - TRANSFORMADA DE FOURIER",
            font=('Arial', 18, 'bold'),
            bg='#8e44ad',
            fg='white'
        )
        titulo.pack(pady=15)
        
        # ===== FRAME PRINCIPAL CON DOS COLUMNAS =====
        frame_principal = tk.Frame(self.root, bg='#f0f0f0')
        frame_principal.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Columna izquierda con scrollbar: Carga de imágenes y operaciones
        frame_izquierda_container = tk.Frame(frame_principal, bg='#f0f0f0', width=400)
        frame_izquierda_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # Canvas para scroll en columna izquierda
        canvas_izquierda = tk.Canvas(frame_izquierda_container, bg='#f0f0f0', highlightthickness=0)
        scrollbar_izquierda = tk.Scrollbar(frame_izquierda_container, orient="vertical", command=canvas_izquierda.yview)
        self.frame_izquierda_scroll = tk.Frame(canvas_izquierda, bg='#f0f0f0')
        
        self.frame_izquierda_scroll.bind(
            "<Configure>",
            lambda e: canvas_izquierda.configure(scrollregion=canvas_izquierda.bbox("all"))
        )
        
        canvas_izquierda.create_window((0, 0), window=self.frame_izquierda_scroll, anchor="nw")
        canvas_izquierda.configure(yscrollcommand=scrollbar_izquierda.set)
        
        canvas_izquierda.pack(side="left", fill="both", expand=True)
        scrollbar_izquierda.pack(side="right", fill="y")
        
        # Habilitar scroll con rueda del mouse
        def _on_mousewheel(event):
            canvas_izquierda.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas_izquierda.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Usar el frame con scroll como frame_izquierda
        frame_izquierda = self.frame_izquierda_scroll
        
        # Columna derecha: Visualización de resultados
        frame_derecha = tk.Frame(frame_principal, bg='white')
        frame_derecha.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ===== SECCIÓN: CARGA DE IMAGEN =====
        self.crear_seccion_carga_imagen(frame_izquierda)
        
        # ===== SECCIÓN: INFORMACIÓN FFT =====
        self.crear_seccion_info_fft(frame_izquierda)
        
        # ===== SECCIÓN: PARÁMETROS DE FILTROS =====
        self.crear_seccion_parametros_filtros(frame_izquierda)
        
        # ===== SECCIÓN: OPERACIONES =====
        self.crear_seccion_operaciones(frame_izquierda)
        
        # ===== SECCIÓN: DCT =====
        self.crear_seccion_dct(frame_izquierda)
        
        # ===== SECCIÓN: COMPRESIÓN DCT =====
        self.crear_seccion_compresion_dct(frame_izquierda)
        
        # ===== SECCIÓN: FILTROS =====
        self.crear_seccion_filtros(frame_izquierda)
        
        # ===== SECCIÓN: VISUALIZACIÓN =====
        self.crear_seccion_visualizacion(frame_derecha)
    
    def crear_seccion_carga_imagen(self, parent):
        """Crea la sección para cargar imagen"""
        frame = tk.LabelFrame(
            parent, 
            text="Cargar Imagen",
            font=('Arial', 11, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        frame.pack(fill=tk.X, pady=(0, 10))
        
        btn_cargar = tk.Button(
            frame,
            text="Cargar Imagen",
            command=self.cargar_imagen,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief=tk.RAISED,
            cursor='hand2'
        )
        btn_cargar.pack(pady=5)
        
        btn_guardar = tk.Button(
            frame,
            text="Guardar Resultado",
            command=self.guardar_resultado,
            bg='#27ae60',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief=tk.RAISED,
            cursor='hand2'
        )
        btn_guardar.pack(pady=5)
        
        self.label_imagen = tk.Label(
            frame,
            text="No se ha cargado ninguna imagen",
            font=('Arial', 9),
            bg='#ecf0f1',
            fg='#7f8c8d'
        )
        self.label_imagen.pack(pady=5)
    
    def crear_seccion_info_fft(self, parent):
        """Crea la sección de información sobre FFT"""
        frame = tk.LabelFrame(
            parent,
            text="Información de la Transformada",
            font=('Arial', 11, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        frame.pack(fill=tk.X, pady=(0, 10))
        
        self.label_info_fft = tk.Label(
            frame,
            text="Cargue una imagen para calcular la FFT",
            font=('Arial', 9),
            bg='#ecf0f1',
            fg='#7f8c8d',
            justify=tk.LEFT
        )
        self.label_info_fft.pack(anchor=tk.W, pady=5)
    
    def crear_seccion_parametros_filtros(self, parent):
        """Crea la sección de parámetros para filtros"""
        frame = tk.LabelFrame(
            parent,
            text="Parámetros de Filtros",
            font=('Arial', 11, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        frame.pack(fill=tk.X, pady=(0, 10))
        
        # Radio de corte
        label_radio = tk.Label(
            frame,
            text="Radio de Corte:",
            font=('Arial', 10),
            bg='#ecf0f1'
        )
        label_radio.pack(anchor=tk.W, pady=(5, 0))
        
        frame_radio = tk.Frame(frame, bg='#ecf0f1')
        frame_radio.pack(fill=tk.X, pady=5)
        
        scale_radio = tk.Scale(
            frame_radio,
            from_=5,
            to=200,
            orient=tk.HORIZONTAL,
            variable=self.radio_filtro,
            bg='#ecf0f1',
            font=('Arial', 9)
        )
        scale_radio.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Sigma (para Gaussiano)
        label_sigma = tk.Label(
            frame,
            text="Sigma (Gaussiano):",
            font=('Arial', 10),
            bg='#ecf0f1'
        )
        label_sigma.pack(anchor=tk.W, pady=(5, 0))
        
        frame_sigma = tk.Frame(frame, bg='#ecf0f1')
        frame_sigma.pack(fill=tk.X, pady=5)
        
        scale_sigma = tk.Scale(
            frame_sigma,
            from_=5,
            to=200,
            orient=tk.HORIZONTAL,
            variable=self.sigma_filtro,
            bg='#ecf0f1',
            font=('Arial', 9)
        )
        scale_sigma.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Orden (para Butterworth)
        label_orden = tk.Label(
            frame,
            text="Orden (Butterworth):",
            font=('Arial', 10),
            bg='#ecf0f1'
        )
        label_orden.pack(anchor=tk.W, pady=(5, 0))
        
        frame_orden = tk.Frame(frame, bg='#ecf0f1')
        frame_orden.pack(fill=tk.X, pady=5)
        
        scale_orden = tk.Scale(
            frame_orden,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
            variable=self.orden_filtro,
            bg='#ecf0f1',
            font=('Arial', 9)
        )
        scale_orden.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def crear_seccion_operaciones(self, parent):
        """Crea la sección de operaciones de Fourier"""
        frame = tk.LabelFrame(
            parent,
            text="Visualizaciones de Fourier",
            font=('Arial', 11, 'bold'),
            bg='#f5eef8',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        frame.pack(fill=tk.X, pady=(0, 10))
        
        row = 0
        for op in self.operaciones.keys():
            btn = tk.Button(
                frame,
                text=op,
                command=lambda o=op: self.ejecutar_operacion(o),
                bg='#9b59b6',
                fg='white',
                font=('Arial', 10),
                width=25,
                cursor='hand2'
            )
            btn.grid(row=row, column=0, padx=5, pady=5, sticky='ew')
            row += 1
        
        frame.columnconfigure(0, weight=1)
    
    def crear_seccion_dct(self, parent):
        """Crea la sección de operaciones DCT"""
        frame = tk.LabelFrame(
            parent,
            text="Transformada del Coseno (DCT)",
            font=('Arial', 11, 'bold'),
            bg='#fef5e7',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        frame.pack(fill=tk.X, pady=(0, 10))
        
        row = 0
        for op in self.operaciones_dct.keys():
            btn = tk.Button(
                frame,
                text=op,
                command=lambda o=op: self.ejecutar_operacion_dct(o),
                bg='#f39c12',
                fg='white',
                font=('Arial', 10),
                width=25,
                cursor='hand2'
            )
            btn.grid(row=row, column=0, padx=5, pady=5, sticky='ew')
            row += 1
        
        frame.columnconfigure(0, weight=1)
    
    def crear_seccion_compresion_dct(self, parent):
        """Crea la sección de compresión por DCT"""
        frame = tk.LabelFrame(
            parent,
            text="Compresión DCT 8x8",
            font=('Arial', 11, 'bold'),
            bg='#fef5e7',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        frame.pack(fill=tk.X, pady=(0, 10))
        
        # Umbral de compresión
        label_umbral = tk.Label(
            frame,
            text="% Coeficientes a mantener:",
            font=('Arial', 10),
            bg='#fef5e7'
        )
        label_umbral.pack(anchor=tk.W, pady=(5, 0))
        
        frame_umbral = tk.Frame(frame, bg='#fef5e7')
        frame_umbral.pack(fill=tk.X, pady=5)
        
        scale_umbral = tk.Scale(
            frame_umbral,
            from_=1,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.umbral_compresion,
            bg='#fef5e7',
            font=('Arial', 9)
        )
        scale_umbral.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Número de coeficientes
        label_coefs = tk.Label(
            frame,
            text="Coeficientes por bloque:",
            font=('Arial', 10),
            bg='#fef5e7'
        )
        label_coefs.pack(anchor=tk.W, pady=(5, 0))
        
        frame_coefs = tk.Frame(frame, bg='#fef5e7')
        frame_coefs.pack(fill=tk.X, pady=5)
        
        scale_coefs = tk.Scale(
            frame_coefs,
            from_=1,
            to=64,
            orient=tk.HORIZONTAL,
            variable=self.num_coeficientes,
            bg='#fef5e7',
            font=('Arial', 9)
        )
        scale_coefs.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Botones de compresión
        btn_umbral = tk.Button(
            frame,
            text="Comprimir por Umbral",
            command=self.comprimir_por_umbral,
            bg='#d68910',
            fg='white',
            font=('Arial', 9, 'bold'),
            cursor='hand2'
        )
        btn_umbral.pack(fill=tk.X, pady=3)
        
        btn_frecuencias = tk.Button(
            frame,
            text="Comprimir por Frecuencias",
            command=self.comprimir_por_frecuencias,
            bg='#d68910',
            fg='white',
            font=('Arial', 9, 'bold'),
            cursor='hand2'
        )
        btn_frecuencias.pack(fill=tk.X, pady=3)
    
    def crear_seccion_filtros(self, parent):
        """Crea la sección de filtros de Fourier"""
        frame = tk.LabelFrame(
            parent,
            text="Filtros en Dominio de Fourier",
            font=('Arial', 11, 'bold'),
            bg='#e8f8f5',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        frame.pack(fill=tk.X, pady=(0, 10))
        
        row = 0
        col = 0
        for nombre_filtro in self.filtros.keys():
            btn = tk.Button(
                frame,
                text=nombre_filtro,
                command=lambda f=nombre_filtro: self.aplicar_filtro(f),
                bg='#16a085',
                fg='white',
                font=('Arial', 9),
                width=20,
                cursor='hand2'
            )
            btn.grid(row=row, column=col, padx=3, pady=3, sticky='ew')
            
            col += 1
            if col > 1:  # 2 columnas
                col = 0
                row += 1
        
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
    
    def crear_seccion_visualizacion(self, parent):
        """Crea la sección de visualización de resultados"""
        # Frame superior con título y botón de análisis
        frame_superior = tk.Frame(parent, bg='white')
        frame_superior.pack(fill=tk.X, pady=10)
        
        label = tk.Label(
            frame_superior,
            text="Resultados de la Transformada de Fourier",
            font=('Arial', 12, 'bold'),
            bg='white'
        )
        label.pack(side=tk.LEFT, padx=10)
        
        # Botón para análisis detallado
        btn_analisis = tk.Button(
            frame_superior,
            text="📊 Análisis Detallado",
            command=self.mostrar_analisis_detallado,
            bg='#e67e22',
            fg='white',
            font=('Arial', 10, 'bold'),
            cursor='hand2'
        )
        btn_analisis.pack(side=tk.RIGHT, padx=10)
        
        # Frame para el canvas de matplotlib
        self.frame_canvas = tk.Frame(parent, bg='white')
        self.frame_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Inicializar visualizador
        self.visualizador = Visualizador(self.frame_canvas)
    
    # ===== MÉTODOS DE CARGA DE IMÁGENES =====
    
    def cargar_imagen(self):
        """Carga una imagen y calcula automáticamente la FFT"""
        ruta = filedialog.askopenfilename(
            title="Seleccionar Imagen",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if ruta:
            # Cargar imagen en escala de grises
            self.imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
            
            if self.imagen is None:
                messagebox.showerror("Error", "No se pudo cargar la imagen")
                return
            
            # Calcular la FFT usando el módulo
            self.fft.cargar_imagen(self.imagen)
            
            nombre_archivo = os.path.basename(ruta)
            self.label_imagen.config(
                text=f"{nombre_archivo}\n{self.imagen.shape[1]}x{self.imagen.shape[0]} px",
                fg='#8e44ad'
            )
            
            # Actualizar información de FFT
            self.actualizar_info_fft()
            
            messagebox.showinfo(
                "Éxito", 
                f"Imagen cargada correctamente:\n{nombre_archivo}\n\nLa FFT ha sido calculada."
            )
    
    def actualizar_info_fft(self):
        """Actualiza la información mostrada sobre la FFT"""
        info = self.fft.obtener_info()
        if info is None:
            return
        
        texto = f"Tamaño FFT: {info['tamaño'][1]} x {info['tamaño'][0]}\n"
        texto += f"Magnitud - Min: {info['magnitud_min']:.2f}, Max: {info['magnitud_max']:.2f}\n"
        texto += f"Fase - Min: {info['fase_min']:.2f}, Max: {info['fase_max']:.2f} rad"
        
        self.label_info_fft.config(text=texto, fg='#27ae60')
    
    def guardar_resultado(self):
        """Guarda la imagen resultado"""
        if self.imagen_resultado is None:
            messagebox.showwarning("Advertencia", "No hay ningún resultado para guardar.\nPrimero aplique una operación.")
            return
        
        nombre_sugerido = f"resultado_{self.nombre_operacion}.png"
        
        ruta = filedialog.asksaveasfilename(
            title="Guardar Resultado",
            defaultextension=".png",
            initialfile=nombre_sugerido,
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tiff"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if ruta:
            try:
                cv2.imwrite(ruta, self.imagen_resultado)
                messagebox.showinfo("Éxito", f"Imagen guardada correctamente:\n{os.path.basename(ruta)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar la imagen:\n{str(e)}")
    
    # ===== MÉTODOS DE VISUALIZACIÓN =====
    
    def mostrar_magnitud(self):
        """Muestra solo el espectro de magnitud"""
        self.imagen_resultado = self.fft.obtener_magnitud_normalizada()
        self.nombre_operacion = "magnitud"
        
        self.visualizador.mostrar_dos_imagenes(
            self.imagen,
            self.fft.obtener_magnitud(),
            "Imagen Original",
            "Espectro de Magnitud (log)",
            "Espectro de Magnitud"
        )
    
    def mostrar_fase(self):
        """Muestra solo el espectro de fase"""
        self.imagen_resultado = self.fft.obtener_fase_normalizada()
        self.nombre_operacion = "fase"
        
        self.visualizador.mostrar_dos_imagenes(
            self.imagen,
            self.fft.obtener_fase(),
            "Imagen Original",
            "Espectro de Fase",
            "Espectro de Fase"
        )
    
    def mostrar_magnitud_fase(self):
        """Muestra magnitud y fase lado a lado"""
        self.nombre_operacion = "magnitud_fase"
        
        self.visualizador.mostrar_tres_imagenes(
            self.imagen,
            self.fft.obtener_magnitud(),
            self.fft.obtener_fase(),
            "Imagen Original",
            "Espectro de Magnitud (log)",
            "Espectro de Fase",
            "Magnitud y Fase"
        )
    
    def mostrar_fft_completa(self):
        """Muestra la visualización completa de la FFT"""
        self.nombre_operacion = "fft_completa"
        
        imagenes = [
            self.imagen,
            self.fft.obtener_magnitud(),
            self.fft.obtener_fase(),
            self.fft.reconstruir_imagen()
        ]
        
        titulos = [
            "Imagen Original",
            "Espectro de Magnitud (log)",
            "Espectro de Fase",
            "Imagen Reconstruida (IFFT)"
        ]
        
        cmaps = ['gray', 'gray', 'hsv', 'gray']
        
        self.visualizador.mostrar_cuadricula(
            imagenes, titulos,
            "Análisis Completo de Fourier",
            filas=2, columnas=2,
            cmaps=cmaps,
            con_colorbar=[1, 2]
        )
    
    # ===== MÉTODOS DE EJECUCIÓN =====
    
    def ejecutar_operacion(self, operacion):
        """Ejecuta una operación de visualización de Fourier"""
        if self.imagen is None:
            messagebox.showwarning("Advertencia", "Debe cargar una imagen primero")
            return
        
        try:
            self.operaciones[operacion]()
        except Exception as e:
            messagebox.showerror("Error", f"Error al aplicar {operacion}:\n{str(e)}")
    
    def aplicar_filtro(self, nombre_filtro):
        """Aplica un filtro de Fourier a la imagen"""
        if self.imagen is None:
            messagebox.showwarning("Advertencia", "Debe cargar una imagen primero")
            return
        
        try:
            tipo_filtro, tipo_parametro = self.filtros[nombre_filtro]
            
            # Obtener parámetros según el tipo de filtro
            if tipo_parametro == 'radio':
                param1 = self.radio_filtro.get()
                param2 = None
            elif tipo_parametro == 'sigma':
                param1 = self.sigma_filtro.get()
                param2 = None
            elif tipo_parametro == 'radio_orden':
                param1 = self.radio_filtro.get()
                param2 = self.orden_filtro.get()
            
            # Aplicar el filtro
            imagen_filtrada, mascara = aplicar_filtro(self.imagen, tipo_filtro, param1, param2)
            
            # Guardar para análisis posterior
            self.ultima_imagen_filtrada = imagen_filtrada
            self.ultima_mascara = mascara
            self.ultimo_nombre_filtro = nombre_filtro
            
            # Normalizar para guardar
            self.imagen_resultado = cv2.normalize(imagen_filtrada, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            self.nombre_operacion = nombre_filtro.replace(' ', '_').replace('-', '_').lower()
            
            # Visualizar resultado
            self.visualizar_filtro(imagen_filtrada, mascara, nombre_filtro)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al aplicar {nombre_filtro}:\n{str(e)}")
    
    def visualizar_filtro(self, imagen_filtrada, mascara, nombre_filtro):
        """Visualiza el resultado del filtro con la imagen original, máscara y resultado"""
        # Normalizar máscara para visualización
        mascara_norm = cv2.normalize(mascara, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        
        imagenes = [
            self.imagen,
            mascara_norm,
            imagen_filtrada
        ]
        
        titulos = [
            "Imagen Original",
            "Máscara del Filtro",
            "Imagen Filtrada"
        ]
        
        cmaps = ['gray', 'viridis', 'gray']
        
        self.visualizador.mostrar_tres_imagenes(
            imagenes[0], imagenes[1], imagenes[2],
            titulos[0], titulos[1], titulos[2],
            f"Filtro: {nombre_filtro}",
            cmap1='gray', cmap2='viridis', cmap3='gray'
        )
    
    def mostrar_analisis_detallado(self):
        """Muestra un análisis detallado del último filtro aplicado"""
        if self.ultima_imagen_filtrada is None:
            messagebox.showwarning(
                "Advertencia", 
                "Primero debe aplicar un filtro para poder analizarlo."
            )
            return
        
        try:
            # Crear analizador
            analizador = AnalizadorFiltros(self.imagen, self.ultima_imagen_filtrada)
            
            # Generar reporte de texto
            texto_analisis = analizador.generar_texto_reporte()
            
            # Visualizar análisis completo
            self.visualizador.mostrar_analisis_completo(
                self.imagen,
                self.ultima_imagen_filtrada,
                self.ultima_mascara,
                self.ultimo_nombre_filtro,
                texto_analisis
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar análisis:\n{str(e)}")
    
    # ===== MÉTODOS DE DCT =====
    
    def ejecutar_operacion_dct(self, operacion):
        """Ejecuta una operación de DCT"""
        if self.imagen is None:
            messagebox.showwarning("Advertencia", "Debe cargar una imagen primero")
            return
        
        try:
            self.operaciones_dct[operacion]()
        except Exception as e:
            messagebox.showerror("Error", f"Error al aplicar {operacion}:\n{str(e)}")
    
    def mostrar_dct_completa(self):
        """Muestra la DCT completa de la imagen"""
        # Cargar imagen en DCT
        self.dct.cargar_imagen(self.imagen)
        
        # Calcular DCT
        dct_data = self.dct.aplicar_dct_completa()
        magnitud_log = self.dct.obtener_magnitud_log()
        
        # Reconstruir imagen
        imagen_reconstruida = self.dct.aplicar_idct_completa()
        
        # Guardar para posible guardado
        self.imagen_resultado = cv2.normalize(magnitud_log, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        self.nombre_operacion = "dct_completa"
        
        # Visualizar
        imagenes = [
            self.imagen,
            magnitud_log,
            imagen_reconstruida
        ]
        
        titulos = [
            "Imagen Original",
            "DCT (Magnitud log)",
            "Reconstruida (IDCT)"
        ]
        
        cmaps = ['gray', 'viridis', 'gray']
        
        self.visualizador.mostrar_tres_imagenes(
            imagenes[0], imagenes[1], imagenes[2],
            titulos[0], titulos[1], titulos[2],
            "Transformada del Coseno Discreta (DCT)",
            cmap1='gray', cmap2='viridis', cmap3='gray'
        )
    
    def mostrar_dct_bloques(self):
        """Muestra la DCT por bloques 8x8"""
        # Cargar imagen en compresor
        self.compresor.cargar_imagen(self.imagen)
        
        # Aplicar DCT por bloques
        dct_bloques = self.compresor.aplicar_dct_por_bloques()
        
        # Visualizar con magnitud logarítmica
        magnitud_log = np.log1p(np.abs(dct_bloques))
        
        # Reconstruir imagen (sin compresión)
        imagen_reconstruida = self.compresor._reconstruir_desde_dct(dct_bloques)
        
        # Guardar para posible guardado
        self.imagen_resultado = cv2.normalize(magnitud_log, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        self.nombre_operacion = "dct_bloques_8x8"
        
        # Visualizar
        self.visualizador.mostrar_tres_imagenes(
            self.imagen,
            magnitud_log,
            imagen_reconstruida,
            "Imagen Original",
            "DCT por Bloques 8x8",
            "Reconstruida (IDCT)",
            "DCT por Bloques 8x8",
            cmap1='gray', cmap2='hot', cmap3='gray'
        )
    
    def comprimir_por_umbral(self):
        """Comprime la imagen usando umbral en coeficientes DCT"""
        if self.imagen is None:
            messagebox.showwarning("Advertencia", "Debe cargar una imagen primero")
            return
        
        try:
            # Cargar imagen
            self.compresor.cargar_imagen(self.imagen)
            
            # Aplicar compresión
            umbral = self.umbral_compresion.get()
            imagen_comprimida, tasa_compresion, dct_comprimida = self.compresor.comprimir_por_umbral(umbral)
            
            # Calcular estadísticas
            stats = self.compresor.obtener_estadisticas_compresion(self.imagen, imagen_comprimida)
            
            # Guardar resultado
            self.imagen_resultado = imagen_comprimida.astype('uint8')
            self.nombre_operacion = f"dct_umbral_{umbral}"
            
            # Crear texto de información
            info_text = f"Compresión por Umbral\n"
            info_text += f"{'='*40}\n"
            info_text += f"Umbral: {umbral}% coeficientes\n"
            info_text += f"Tasa compresión: {tasa_compresion:.2f}%\n"
            info_text += f"MSE: {stats['mse']:.2f}\n"
            info_text += f"PSNR: {stats['psnr']:.2f} dB\n"
            
            # Visualizar comparación
            self.visualizar_compresion_dct(
                imagen_comprimida,
                dct_comprimida,
                f"Umbral {umbral}%",
                info_text
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al comprimir:\n{str(e)}")
    
    def comprimir_por_frecuencias(self):
        """Comprime la imagen manteniendo solo bajas frecuencias"""
        if self.imagen is None:
            messagebox.showwarning("Advertencia", "Debe cargar una imagen primero")
            return
        
        try:
            # Cargar imagen
            self.compresor.cargar_imagen(self.imagen)
            
            # Aplicar compresión
            num_coefs = self.num_coeficientes.get()
            imagen_comprimida, tasa_compresion, dct_comprimida = self.compresor.comprimir_por_frecuencias(num_coefs)
            
            # Calcular estadísticas
            stats = self.compresor.obtener_estadisticas_compresion(self.imagen, imagen_comprimida)
            
            # Guardar resultado
            self.imagen_resultado = imagen_comprimida.astype('uint8')
            self.nombre_operacion = f"dct_frecuencias_{num_coefs}"
            
            # Crear texto de información
            info_text = f"Compresión por Frecuencias\n"
            info_text += f"{'='*40}\n"
            info_text += f"Coeficientes: {num_coefs}/64 por bloque\n"
            info_text += f"Tasa compresión: {tasa_compresion:.2f}%\n"
            info_text += f"MSE: {stats['mse']:.2f}\n"
            info_text += f"PSNR: {stats['psnr']:.2f} dB\n"
            
            # Visualizar comparación
            self.visualizar_compresion_dct(
                imagen_comprimida,
                dct_comprimida,
                f"Frecuencias {num_coefs}",
                info_text
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al comprimir:\n{str(e)}")
    
    def visualizar_compresion_dct(self, imagen_comprimida, dct_comprimida, titulo, info_text):
        """Visualiza el resultado de la compresión DCT con análisis de pérdida de información"""
        # Calcular diferencia
        diferencia = np.abs(self.imagen.astype(float) - imagen_comprimida.astype(float))
        
        # Magnitud log de DCT
        magnitud_log = np.log1p(np.abs(dct_comprimida))
        
        # Calcular estadísticas detalladas
        stats = self.compresor.obtener_estadisticas_compresion(self.imagen, imagen_comprimida)
        mse = stats['mse']
        psnr = stats['psnr']
        
        # Interpretación de PSNR
        if psnr > 40:
            calidad = "Excelente"
            color_calidad = "green"
            interpretacion = "Pérdida imperceptible"
        elif psnr > 35:
            calidad = "Muy buena"
            color_calidad = "lightgreen"
            interpretacion = "Pérdida apenas perceptible"
        elif psnr > 30:
            calidad = "Buena"
            color_calidad = "yellow"
            interpretacion = "Pérdida perceptible pero aceptable"
        elif psnr > 25:
            calidad = "Aceptable"
            color_calidad = "orange"
            interpretacion = "Pérdida visible"
        else:
            calidad = "Pobre"
            color_calidad = "red"
            interpretacion = "Pérdida significativa"
        
        # Crear visualización
        imagenes = [
            self.imagen,
            imagen_comprimida,
            magnitud_log,
            diferencia
        ]
        
        titulos = [
            "Imagen Original",
            f"Comprimida - {titulo}",
            "DCT Comprimida",
            "Diferencia"
        ]
        
        cmaps = ['gray', 'gray', 'hot', 'hot']
        
        # Usar mostrar_cuadricula pero con texto
        fig = self.visualizador._crear_canvas(figsize=(14, 10))
        fig.suptitle(f'Compresión DCT - {titulo}', fontsize=14, fontweight='bold')
        
        # Primeras 4 imágenes
        for idx in range(4):
            ax = fig.add_subplot(2, 3, idx + 1)
            im = ax.imshow(imagenes[idx], cmap=cmaps[idx])
            ax.set_title(titulos[idx], fontsize=11, fontweight='bold')
            ax.axis('off')
            if idx >= 2:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Histograma comparativo
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.hist(self.imagen.ravel(), bins=64, alpha=0.6, label='Original', color='blue')
        ax5.hist(imagen_comprimida.ravel(), bins=64, alpha=0.6, label='Comprimida', color='red')
        ax5.set_title('Histogramas', fontsize=11, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Información ampliada con análisis de PSNR
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        info_completa = info_text + "\n"
        info_completa += f"\nAnálisis de Calidad Visual\n"
        info_completa += f"{'='*40}\n"
        info_completa += f"Calidad: {calidad}\n"
        info_completa += f"Interpretación: {interpretacion}\n\n"
        info_completa += f"Referencia PSNR:\n"
        info_completa += f"  > 40 dB: Excelente\n"
        info_completa += f"  35-40 dB: Muy buena\n"
        info_completa += f"  30-35 dB: Buena\n"
        info_completa += f"  25-30 dB: Aceptable\n"
        info_completa += f"  < 25 dB: Pobre\n\n"
        info_completa += f"Máx. diferencia: {diferencia.max():.1f}\n"
        info_completa += f"Media diferencia: {diferencia.mean():.2f}"
        
        ax6.text(0.05, 0.95, info_completa,
                transform=ax6.transAxes,
                fontsize=10,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=color_calidad, alpha=0.3))
        
        self.visualizador._mostrar_canvas()
    
    def cerrar_aplicacion(self):
        """Cierra correctamente la aplicación liberando todos los recursos"""
        if self.visualizador:
            self.visualizador.cerrar()
        self.root.quit()
        self.root.destroy()


# ===================================================================
# FUNCIÓN PRINCIPAL
# ===================================================================

def main():
    root = tk.Tk()
    app = InterfazFourier(root)
    root.mainloop()


if __name__ == "__main__":
    main()
