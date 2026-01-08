"""
Interfaz gráfica para la Práctica 2: Generación de Ruido y Aplicación de Filtros.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from generacion_ruido import (
    aplicar_ruido_sal_pimienta,
    aplicar_ruido_sal,
    aplicar_ruido_pimienta,
    aplicar_ruido_gaussiano,
    calcular_histograma
)

from filtros_lineales import (
    # Paso altas
    filtro_sobel, filtro_prewitt, filtro_roberts, filtro_kirsch, filtro_canny,
    filtro_laplaciano_clasico, filtro_laplaciano_8_vecinos,
    filtro_laplaciano_horizontal, filtro_laplaciano_vertical,
    filtro_laplaciano_diagonal_principal, filtro_laplaciano_diagonal_secundaria,
    # Paso bajas
    filtro_promediador, filtro_promediador_pesado, filtro_gaussiano, filtro_bilateral
)

from filtros_no_lineales import (
    filtro_mediana, filtro_moda, filtro_maximo, filtro_minimo,
    filtro_mediana_adaptativa, filtro_contraharmonic_mean, filtro_mediana_ponderada
)

from visualizador import crear_canvas_matplotlib, mostrar_imagen_en_canvas, mostrar_histograma_en_canvas


class InterfazPractica2:
    """
    Interfaz gráfica principal para la Práctica 2.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Práctica 2 - Ruido y Filtros")
        self.root.geometry("1400x900")
        
        # Variables de estado
        self.imagen_original = None
        self.imagen_procesada = None
        self.imagen_base_filtros = None  # Imagen con ruido para aplicar filtros
        self.ruta_imagen = None
        
        # Configurar interfaz
        self.crear_interfaz()
    
    def crear_interfaz(self):
        """Crea todos los elementos de la interfaz."""
        
        # Frame principal con scroll
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # ==================== PANEL SUPERIOR: CARGA DE IMAGEN ====================
        panel_carga = ttk.LabelFrame(main_container, text="Carga de Imagen", padding=10)
        panel_carga.pack(fill='x', pady=(0, 10))
        
        ttk.Button(panel_carga, text="Cargar Imagen", command=self.cargar_imagen).pack(side='left', padx=5)
        self.label_ruta = ttk.Label(panel_carga, text="No se ha cargado ninguna imagen")
        self.label_ruta.pack(side='left', padx=10)
        
        # ==================== PANEL CENTRAL: NOTEBOOK CON PESTAÑAS ====================
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill='both', expand=True)
        
        # Pestaña 1: Generación de Ruido
        self.crear_pestana_ruido()
        
        # Pestaña 2: Filtros Paso Altas
        self.crear_pestana_filtros_paso_altas()
        
        # Pestaña 3: Filtros Paso Bajas
        self.crear_pestana_filtros_paso_bajas()
        
        # Pestaña 4: Filtros No Lineales
        self.crear_pestana_filtros_no_lineales()
    
    def crear_pestana_ruido(self):
        """Crea la pestaña de generación de ruido."""
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Generación de Ruido")
        
        # Panel de controles
        panel_controles = ttk.LabelFrame(frame, text="Controles de Ruido", padding=10)
        panel_controles.pack(fill='x', pady=(0, 10))
        
        # Ruido Sal y Pimienta
        frame_sp = ttk.Frame(panel_controles)
        frame_sp.pack(fill='x', pady=5)
        
        ttk.Label(frame_sp, text="Ruido Sal y Pimienta - Probabilidad:").pack(side='left', padx=5)
        self.prob_sp = tk.DoubleVar(value=0.05)
        ttk.Scale(frame_sp, from_=0.01, to=0.2, variable=self.prob_sp, 
                 orient='horizontal', length=200).pack(side='left', padx=5)
        ttk.Label(frame_sp, textvariable=self.prob_sp).pack(side='left', padx=5)
        ttk.Button(frame_sp, text="Aplicar Ambos", 
                  command=self.aplicar_ruido_sal_pimienta).pack(side='left', padx=5)
        ttk.Button(frame_sp, text="Solo Sal", 
                  command=self.aplicar_ruido_sal).pack(side='left', padx=5)
        ttk.Button(frame_sp, text="Solo Pimienta", 
                  command=self.aplicar_ruido_pimienta).pack(side='left', padx=5)
        
        # Ruido Gaussiano
        frame_gauss = ttk.Frame(panel_controles)
        frame_gauss.pack(fill='x', pady=5)
        
        ttk.Label(frame_gauss, text="Ruido Gaussiano - Sigma:").pack(side='left', padx=5)
        self.sigma_gauss = tk.DoubleVar(value=25)
        ttk.Scale(frame_gauss, from_=5, to=100, variable=self.sigma_gauss, 
                 orient='horizontal', length=200).pack(side='left', padx=5)
        ttk.Label(frame_gauss, textvariable=self.sigma_gauss).pack(side='left', padx=5)
        ttk.Button(frame_gauss, text="Aplicar", 
                  command=self.aplicar_ruido_gaussiano).pack(side='left', padx=5)
        
        # Botón para guardar imagen
        frame_guardar = ttk.Frame(panel_controles)
        frame_guardar.pack(fill='x', pady=10)
        ttk.Button(frame_guardar, text="💾 Guardar Imagen con Ruido", 
                  command=self.guardar_imagen).pack()
        
        # Área de visualización
        frame_visualizacion = ttk.Frame(frame)
        frame_visualizacion.pack(fill='both', expand=True)
        
        # Canvas para imágenes
        self.canvas_ruido_original = crear_canvas_matplotlib(frame_visualizacion, figsize=(5, 4))
        self.canvas_ruido_procesada = crear_canvas_matplotlib(frame_visualizacion, figsize=(5, 4))
        self.canvas_histograma = crear_canvas_matplotlib(frame_visualizacion, figsize=(5, 4))
    
    def crear_pestana_filtros_paso_altas(self):
        """Crea la pestaña de filtros paso altas."""
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Filtros Paso Altas")
        
        # Panel de controles
        panel_controles = ttk.LabelFrame(frame, text="Operadores de Detección de Bordes", padding=10)
        panel_controles.pack(fill='x', pady=(0, 10))
        
        # Frame para operadores de primer orden
        frame_primer_orden = ttk.LabelFrame(panel_controles, text="Operadores de Primer Orden", padding=5)
        frame_primer_orden.pack(fill='x', pady=5)
        
        ttk.Button(frame_primer_orden, text="Sobel", 
                  command=lambda: self.aplicar_filtro(filtro_sobel)).pack(side='left', padx=3)
        ttk.Button(frame_primer_orden, text="Prewitt", 
                  command=lambda: self.aplicar_filtro(filtro_prewitt)).pack(side='left', padx=3)
        ttk.Button(frame_primer_orden, text="Roberts", 
                  command=lambda: self.aplicar_filtro(filtro_roberts)).pack(side='left', padx=3)
        ttk.Button(frame_primer_orden, text="Kirsch", 
                  command=lambda: self.aplicar_filtro(filtro_kirsch)).pack(side='left', padx=3)
        ttk.Button(frame_primer_orden, text="Canny", 
                  command=lambda: self.aplicar_filtro(filtro_canny)).pack(side='left', padx=3)
        
        # Frame para operadores de segundo orden
        frame_segundo_orden = ttk.LabelFrame(panel_controles, text="Operadores de Segundo Orden (Laplacianos)", padding=5)
        frame_segundo_orden.pack(fill='x', pady=5)
        
        fila1 = ttk.Frame(frame_segundo_orden)
        fila1.pack(fill='x', pady=2)
        ttk.Button(fila1, text="Laplaciano Clásico", 
                  command=lambda: self.aplicar_filtro(filtro_laplaciano_clasico)).pack(side='left', padx=3)
        ttk.Button(fila1, text="Laplaciano 8 Vecinos", 
                  command=lambda: self.aplicar_filtro(filtro_laplaciano_8_vecinos)).pack(side='left', padx=3)
        
        fila2 = ttk.Frame(frame_segundo_orden)
        fila2.pack(fill='x', pady=2)
        ttk.Button(fila2, text="Laplaciano Horizontal", 
                  command=lambda: self.aplicar_filtro(filtro_laplaciano_horizontal)).pack(side='left', padx=3)
        ttk.Button(fila2, text="Laplaciano Vertical", 
                  command=lambda: self.aplicar_filtro(filtro_laplaciano_vertical)).pack(side='left', padx=3)
        
        fila3 = ttk.Frame(frame_segundo_orden)
        fila3.pack(fill='x', pady=2)
        ttk.Button(fila3, text="Laplaciano Diagonal Principal", 
                  command=lambda: self.aplicar_filtro(filtro_laplaciano_diagonal_principal)).pack(side='left', padx=3)
        ttk.Button(fila3, text="Laplaciano Diagonal Secundaria", 
                  command=lambda: self.aplicar_filtro(filtro_laplaciano_diagonal_secundaria)).pack(side='left', padx=3)
        
        # Botón para guardar imagen
        ttk.Button(panel_controles, text="💾 Guardar Imagen Filtrada", 
                  command=self.guardar_imagen).pack(pady=10)
        
        # Área de visualización - 3 imágenes horizontales
        frame_visualizacion = ttk.Frame(frame)
        frame_visualizacion.pack(fill='both', expand=True)
        
        self.canvas_pa_original = crear_canvas_matplotlib(frame_visualizacion, figsize=(4, 4))
        self.canvas_pa_con_ruido = crear_canvas_matplotlib(frame_visualizacion, figsize=(4, 4))
        self.canvas_pa_procesada = crear_canvas_matplotlib(frame_visualizacion, figsize=(4, 4))
    
    def crear_pestana_filtros_paso_bajas(self):
        """Crea la pestaña de filtros paso bajas."""
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Filtros Paso Bajas")
        
        # Panel de controles
        panel_controles = ttk.LabelFrame(frame, text="Filtros de Suavizado", padding=10)
        panel_controles.pack(fill='x', pady=(0, 10))
        
        # Tamaño de kernel
        frame_kernel = ttk.Frame(panel_controles)
        frame_kernel.pack(fill='x', pady=5)
        ttk.Label(frame_kernel, text="Tamaño de Kernel:").pack(side='left', padx=5)
        self.tamano_kernel = tk.IntVar(value=5)
        ttk.Spinbox(frame_kernel, from_=3, to=15, increment=2, 
                   textvariable=self.tamano_kernel, width=5).pack(side='left', padx=5)
        
        # Botones de filtros
        frame_botones = ttk.Frame(panel_controles)
        frame_botones.pack(fill='x', pady=5)
        
        ttk.Button(frame_botones, text="Filtro Promediador", 
                  command=self.aplicar_filtro_promediador).pack(side='left', padx=3)
        ttk.Button(frame_botones, text="Filtro Promediador Pesado", 
                  command=lambda: self.aplicar_filtro(filtro_promediador_pesado)).pack(side='left', padx=3)
        ttk.Button(frame_botones, text="Filtro Gaussiano", 
                  command=self.aplicar_filtro_gaussiano).pack(side='left', padx=3)
        ttk.Button(frame_botones, text="Filtro Bilateral", 
                  command=lambda: self.aplicar_filtro(filtro_bilateral)).pack(side='left', padx=3)
        
        # Botón para guardar imagen
        frame_guardar = ttk.Frame(panel_controles)
        frame_guardar.pack(fill='x', pady=10)
        ttk.Button(frame_guardar, text="💾 Guardar Imagen Filtrada", 
                  command=self.guardar_imagen).pack()
        
        # Área de visualización - 3 imágenes horizontales
        frame_visualizacion = ttk.Frame(frame)
        frame_visualizacion.pack(fill='both', expand=True)
        
        self.canvas_pb_original = crear_canvas_matplotlib(frame_visualizacion, figsize=(4, 4))
        self.canvas_pb_con_ruido = crear_canvas_matplotlib(frame_visualizacion, figsize=(4, 4))
        self.canvas_pb_procesada = crear_canvas_matplotlib(frame_visualizacion, figsize=(4, 4))
    
    def crear_pestana_filtros_no_lineales(self):
        """Crea la pestaña de filtros no lineales."""
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Filtros No Lineales")
        
        # Panel de controles
        panel_controles = ttk.LabelFrame(frame, text="Filtros de Orden", padding=10)
        panel_controles.pack(fill='x', pady=(0, 10))
        
        # Tamaño de kernel
        frame_kernel = ttk.Frame(panel_controles)
        frame_kernel.pack(fill='x', pady=5)
        ttk.Label(frame_kernel, text="Tamaño de Kernel:").pack(side='left', padx=5)
        self.tamano_kernel_nl = tk.IntVar(value=5)
        ttk.Spinbox(frame_kernel, from_=3, to=15, increment=2, 
                   textvariable=self.tamano_kernel_nl, width=5).pack(side='left', padx=5)
        
        # Filtros básicos
        frame_basicos = ttk.LabelFrame(panel_controles, text="Filtros Básicos", padding=5)
        frame_basicos.pack(fill='x', pady=5)
        
        ttk.Button(frame_basicos, text="Mediana", 
                  command=self.aplicar_filtro_mediana).pack(side='left', padx=3)
        ttk.Button(frame_basicos, text="Moda", 
                  command=self.aplicar_filtro_moda).pack(side='left', padx=3)
        ttk.Button(frame_basicos, text="Máximo", 
                  command=self.aplicar_filtro_maximo).pack(side='left', padx=3)
        ttk.Button(frame_basicos, text="Mínimo", 
                  command=self.aplicar_filtro_minimo).pack(side='left', padx=3)
        
        # Filtros opcionales
        frame_opcionales = ttk.LabelFrame(panel_controles, text="Filtros Avanzados (Opcionales)", padding=5)
        frame_opcionales.pack(fill='x', pady=5)
        
        ttk.Button(frame_opcionales, text="Mediana Adaptativa", 
                  command=lambda: self.aplicar_filtro(filtro_mediana_adaptativa)).pack(side='left', padx=3)
        ttk.Button(frame_opcionales, text="Contraharmonic Mean", 
                  command=lambda: self.aplicar_filtro(filtro_contraharmonic_mean)).pack(side='left', padx=3)
        ttk.Button(frame_opcionales, text="Mediana Ponderada", 
                  command=self.aplicar_filtro_mediana_ponderada).pack(side='left', padx=3)
        
        # Botón para guardar imagen
        frame_guardar = ttk.Frame(panel_controles)
        frame_guardar.pack(fill='x', pady=10)
        ttk.Button(frame_guardar, text="💾 Guardar Imagen Filtrada", 
                  command=self.guardar_imagen).pack()
        
        # Área de visualización - 3 imágenes horizontales
        frame_visualizacion = ttk.Frame(frame)
        frame_visualizacion.pack(fill='both', expand=True)
        
        self.canvas_nl_original = crear_canvas_matplotlib(frame_visualizacion, figsize=(4, 4))
        self.canvas_nl_con_ruido = crear_canvas_matplotlib(frame_visualizacion, figsize=(4, 4))
        self.canvas_nl_procesada = crear_canvas_matplotlib(frame_visualizacion, figsize=(4, 4))
    

    
    # ==================== MÉTODOS DE FUNCIONALIDAD ====================
    
    def cargar_imagen(self):
        """Carga una imagen desde el sistema de archivos."""
        ruta = filedialog.askopenfilename(
            title="Seleccionar Imagen",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("Todos", "*.*")]
        )
        
        if ruta:
            self.ruta_imagen = ruta
            self.imagen_original = cv2.imread(ruta)
            
            if self.imagen_original is None:
                messagebox.showerror("Error", "No se pudo cargar la imagen")
                return
            
            self.imagen_procesada = self.imagen_original.copy()
            self.label_ruta.config(text=f"Imagen: {ruta.split('/')[-1]}")
            
            # Actualizar visualización en todas las pestañas
            self.actualizar_todas_vistas()
            
            messagebox.showinfo("Éxito", "Imagen cargada correctamente")
    
    def actualizar_todas_vistas(self):
        """Actualiza la visualización en todas las pestañas."""
        if self.imagen_original is None:
            return
        
        # Pestaña de ruido
        mostrar_imagen_en_canvas(self.canvas_ruido_original, self.imagen_original, "Imagen Original")
        
        # Pestaña de filtros paso altas - mostrar original en primer canvas
        mostrar_imagen_en_canvas(self.canvas_pa_original, self.imagen_original, "Original")
        
        # Pestaña de filtros paso bajas - mostrar original en primer canvas
        mostrar_imagen_en_canvas(self.canvas_pb_original, self.imagen_original, "Original")
        
        # Pestaña de filtros no lineales - mostrar original en primer canvas
        mostrar_imagen_en_canvas(self.canvas_nl_original, self.imagen_original, "Original")
    
    def actualizar_imagen_base_todas_pestanas(self):
        """Actualiza la imagen base (con ruido) en todas las pestañas de filtros."""
        if self.imagen_base_filtros is None:
            return
        
        # Actualizar imagen con ruido en el canvas central de cada pestaña de filtros
        mostrar_imagen_en_canvas(self.canvas_pa_con_ruido, self.imagen_base_filtros, "Con Ruido")
        mostrar_imagen_en_canvas(self.canvas_pb_con_ruido, self.imagen_base_filtros, "Con Ruido")
        mostrar_imagen_en_canvas(self.canvas_nl_con_ruido, self.imagen_base_filtros, "Con Ruido")
    
    def aplicar_ruido_sal_pimienta(self):
        """Aplica ruido sal y pimienta a la imagen."""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        prob = self.prob_sp.get()
        # Siempre aplicar sobre la imagen original
        self.imagen_procesada = aplicar_ruido_sal_pimienta(self.imagen_original, prob)
        self.imagen_base_filtros = self.imagen_procesada.copy()  # Esta será la base para filtros
        
        mostrar_imagen_en_canvas(self.canvas_ruido_procesada, self.imagen_procesada, 
                                f"Sal y Pimienta (p={prob:.3f})")
        mostrar_histograma_en_canvas(self.canvas_histograma, self.imagen_procesada, 
                                    "Histograma con Ruido")
        
        # Actualizar imagen base en todas las pestañas
        self.actualizar_imagen_base_todas_pestanas()
    
    def aplicar_ruido_sal(self):
        """Aplica solo ruido sal (píxeles blancos) a la imagen."""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        prob = self.prob_sp.get()
        # Siempre aplicar sobre la imagen original
        self.imagen_procesada = aplicar_ruido_sal(self.imagen_original, prob)
        self.imagen_base_filtros = self.imagen_procesada.copy()  # Esta será la base para filtros
        
        mostrar_imagen_en_canvas(self.canvas_ruido_procesada, self.imagen_procesada, 
                                f"Solo Sal (p={prob:.3f})")
        mostrar_histograma_en_canvas(self.canvas_histograma, self.imagen_procesada, 
                                    "Histograma con Ruido Sal")
        
        # Actualizar imagen base en todas las pestañas
        self.actualizar_imagen_base_todas_pestanas()
    
    def aplicar_ruido_pimienta(self):
        """Aplica solo ruido pimienta (píxeles negros) a la imagen."""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        prob = self.prob_sp.get()
        # Siempre aplicar sobre la imagen original
        self.imagen_procesada = aplicar_ruido_pimienta(self.imagen_original, prob)
        self.imagen_base_filtros = self.imagen_procesada.copy()  # Esta será la base para filtros
        
        mostrar_imagen_en_canvas(self.canvas_ruido_procesada, self.imagen_procesada, 
                                f"Solo Pimienta (p={prob:.3f})")
        mostrar_histograma_en_canvas(self.canvas_histograma, self.imagen_procesada, 
                                    "Histograma con Ruido Pimienta")
        
        # Actualizar imagen base en todas las pestañas
        self.actualizar_imagen_base_todas_pestanas()
    
    def aplicar_ruido_gaussiano(self):
        """Aplica ruido gaussiano a la imagen."""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        sigma = self.sigma_gauss.get()
        # Siempre aplicar sobre la imagen original
        self.imagen_procesada = aplicar_ruido_gaussiano(self.imagen_original, media=0, sigma=sigma)
        self.imagen_base_filtros = self.imagen_procesada.copy()  # Esta será la base para filtros
        
        mostrar_imagen_en_canvas(self.canvas_ruido_procesada, self.imagen_procesada, 
                                f"Ruido Gaussiano (σ={sigma:.1f})")
        mostrar_histograma_en_canvas(self.canvas_histograma, self.imagen_procesada, 
                                    "Histograma con Ruido Gaussiano")
        
        # Actualizar imagen base en todas las pestañas
        self.actualizar_imagen_base_todas_pestanas()
    
    def aplicar_filtro(self, funcion_filtro):
        """Aplica un filtro genérico a la imagen."""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            # Usar imagen_base_filtros si existe (imagen con ruido), sino la original
            imagen_base = self.imagen_base_filtros if self.imagen_base_filtros is not None else self.imagen_original
            resultado = funcion_filtro(imagen_base)
            self.imagen_procesada = resultado
            
            # Actualizar visualización según la pestaña actual
            pestaña_actual = self.notebook.index(self.notebook.select())
            
            if pestaña_actual == 1:  # Filtros paso altas
                # Mostrar las 3 imágenes
                mostrar_imagen_en_canvas(self.canvas_pa_original, self.imagen_original, "Original")
                mostrar_imagen_en_canvas(self.canvas_pa_con_ruido, imagen_base, "Con Ruido" if self.imagen_base_filtros is not None else "Original")
                mostrar_imagen_en_canvas(self.canvas_pa_procesada, resultado, 
                                        f"Procesada")
            elif pestaña_actual == 2:  # Filtros paso bajas
                # Mostrar las 3 imágenes
                mostrar_imagen_en_canvas(self.canvas_pb_original, self.imagen_original, "Original")
                mostrar_imagen_en_canvas(self.canvas_pb_con_ruido, imagen_base, "Con Ruido" if self.imagen_base_filtros is not None else "Original")
                mostrar_imagen_en_canvas(self.canvas_pb_procesada, resultado, 
                                        f"Procesada")
            elif pestaña_actual == 3:  # Filtros no lineales
                # Mostrar las 3 imágenes
                mostrar_imagen_en_canvas(self.canvas_nl_original, self.imagen_original, "Original")
                mostrar_imagen_en_canvas(self.canvas_nl_con_ruido, imagen_base, "Con Ruido" if self.imagen_base_filtros is not None else "Original")
                mostrar_imagen_en_canvas(self.canvas_nl_procesada, resultado, 
                                        f"Procesada")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al aplicar filtro: {str(e)}")
    
    def aplicar_filtro_promediador(self):
        """Aplica filtro promediador con tamaño de kernel configurable."""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        tamano = self.tamano_kernel.get()
        imagen_base = self.imagen_base_filtros if self.imagen_base_filtros is not None else self.imagen_original
        resultado = filtro_promediador(imagen_base, tamano)
        self.imagen_procesada = resultado
        
        # Mostrar las 3 imágenes
        mostrar_imagen_en_canvas(self.canvas_pb_original, self.imagen_original, "Original")
        mostrar_imagen_en_canvas(self.canvas_pb_con_ruido, imagen_base, "Con Ruido" if self.imagen_base_filtros is not None else "Original")
        mostrar_imagen_en_canvas(self.canvas_pb_procesada, resultado, 
                                f"Procesada")
    
    def aplicar_filtro_gaussiano(self):
        """Aplica filtro gaussiano con tamaño de kernel configurable."""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        tamano = self.tamano_kernel.get()
        imagen_base = self.imagen_base_filtros if self.imagen_base_filtros is not None else self.imagen_original
        resultado = filtro_gaussiano(imagen_base, tamano)
        self.imagen_procesada = resultado
        
        # Mostrar las 3 imágenes
        mostrar_imagen_en_canvas(self.canvas_pb_original, self.imagen_original, "Original")
        mostrar_imagen_en_canvas(self.canvas_pb_con_ruido, imagen_base, "Con Ruido" if self.imagen_base_filtros is not None else "Original")
        mostrar_imagen_en_canvas(self.canvas_pb_procesada, resultado, 
                                f"Procesada")
    
    def aplicar_filtro_mediana(self):
        """Aplica filtro de mediana con tamaño de kernel configurable."""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        tamano = self.tamano_kernel_nl.get()
        imagen_base = self.imagen_base_filtros if self.imagen_base_filtros is not None else self.imagen_original
        resultado = filtro_mediana(imagen_base, tamano)
        self.imagen_procesada = resultado
        
        # Mostrar las 3 imágenes
        mostrar_imagen_en_canvas(self.canvas_nl_original, self.imagen_original, "Original")
        mostrar_imagen_en_canvas(self.canvas_nl_con_ruido, imagen_base, "Con Ruido" if self.imagen_base_filtros is not None else "Original")
        mostrar_imagen_en_canvas(self.canvas_nl_procesada, resultado, 
                                f"Procesada")
    
    def aplicar_filtro_moda(self):
        """Aplica filtro de moda con tamaño de kernel configurable."""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        tamano = self.tamano_kernel_nl.get()
        imagen_base = self.imagen_base_filtros if self.imagen_base_filtros is not None else self.imagen_original
        resultado = filtro_moda(imagen_base, tamano)
        self.imagen_procesada = resultado
        
        # Mostrar las 3 imágenes
        mostrar_imagen_en_canvas(self.canvas_nl_original, self.imagen_original, "Original")
        mostrar_imagen_en_canvas(self.canvas_nl_con_ruido, imagen_base, "Con Ruido" if self.imagen_base_filtros is not None else "Original")
        mostrar_imagen_en_canvas(self.canvas_nl_procesada, resultado, 
                                f"Procesada")
    
    def aplicar_filtro_maximo(self):
        """Aplica filtro de máximo con tamaño de kernel configurable."""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        tamano = self.tamano_kernel_nl.get()
        imagen_base = self.imagen_base_filtros if self.imagen_base_filtros is not None else self.imagen_original
        resultado = filtro_maximo(imagen_base, tamano)
        self.imagen_procesada = resultado
        
        # Mostrar las 3 imágenes
        mostrar_imagen_en_canvas(self.canvas_nl_original, self.imagen_original, "Original")
        mostrar_imagen_en_canvas(self.canvas_nl_con_ruido, imagen_base, "Con Ruido" if self.imagen_base_filtros is not None else "Original")
        mostrar_imagen_en_canvas(self.canvas_nl_procesada, resultado, 
                                f"Procesada")
    
    def aplicar_filtro_minimo(self):
        """Aplica filtro de mínimo con tamaño de kernel configurable."""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        tamano = self.tamano_kernel_nl.get()
        imagen_base = self.imagen_base_filtros if self.imagen_base_filtros is not None else self.imagen_original
        resultado = filtro_minimo(imagen_base, tamano)
        self.imagen_procesada = resultado
        
        # Mostrar las 3 imágenes
        mostrar_imagen_en_canvas(self.canvas_nl_original, self.imagen_original, "Original")
        mostrar_imagen_en_canvas(self.canvas_nl_con_ruido, imagen_base, "Con Ruido" if self.imagen_base_filtros is not None else "Original")
        mostrar_imagen_en_canvas(self.canvas_nl_procesada, resultado, 
                                f"Procesada")
    
    def aplicar_filtro_mediana_ponderada(self):
        """Aplica filtro de mediana ponderada con tamaño de kernel configurable."""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        tamano = self.tamano_kernel_nl.get()
        imagen_base = self.imagen_base_filtros if self.imagen_base_filtros is not None else self.imagen_original
        resultado = filtro_mediana_ponderada(imagen_base, tamano)
        self.imagen_procesada = resultado
        
        # Mostrar las 3 imágenes
        mostrar_imagen_en_canvas(self.canvas_nl_original, self.imagen_original, "Original")
        mostrar_imagen_en_canvas(self.canvas_nl_con_ruido, imagen_base, "Con Ruido" if self.imagen_base_filtros is not None else "Original")
        mostrar_imagen_en_canvas(self.canvas_nl_procesada, resultado, 
                                f"Procesada")
    
    def guardar_imagen(self):
        """Guarda la imagen procesada."""
        if self.imagen_procesada is None:
            messagebox.showwarning("Advertencia", "No hay imagen procesada para guardar")
            return
        
        ruta = filedialog.asksaveasfilename(
            title="Guardar Imagen",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos", "*.*")]
        )
        
        if ruta:
            cv2.imwrite(ruta, self.imagen_procesada)
            messagebox.showinfo("Éxito", f"Imagen guardada en: {ruta}")
    
def iniciar_interfaz():
    """Inicia la interfaz gráfica."""
    root = tk.Tk()
    app = InterfazPractica2(root)
    root.mainloop()


if __name__ == "__main__":
    iniciar_interfaz()
