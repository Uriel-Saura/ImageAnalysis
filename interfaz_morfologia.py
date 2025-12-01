# ===================================================================
# INTERFAZ GRÁFICA PARA OPERACIONES MORFOLÓGICAS
# Permite cargar imágenes binarias y en escala de grises
# y aplicar diversas operaciones morfológicas seleccionadas por el usuario
# ===================================================================

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os


# ===================================================================
# CLASE PRINCIPAL DE LA INTERFAZ
# ===================================================================

class InterfazMorfologia:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis de Imágenes - Operaciones Morfológicas")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Configurar el protocolo de cierre de ventana
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar_aplicacion)
        
        # Variables para la imagen
        self.imagen = None
        self.imagen_resultado = None  # Guardar el resultado de la última operación
        self.nombre_operacion = None  # Nombre de la última operación aplicada
        self.tipo_imagen = None  # 'binaria' o 'grises'
        
        # Variables para parámetros
        self.kernel_size = tk.IntVar(value=5)
        
        # Diccionario unificado de operaciones disponibles (para cualquier tipo de imagen)
        self.operaciones = {
            'Erosión': self.aplicar_erosion,
            'Dilatación': self.aplicar_dilatacion,
            'Apertura': self.aplicar_apertura,
            'Cierre': self.aplicar_cierre,
            'Frontera': self.aplicar_frontera,
            'Adelgazamiento': self.aplicar_adelgazamiento,
            'Hit-or-Miss': self.aplicar_hit_or_miss,
            'Esqueleto': self.aplicar_esqueleto,
            'Gradiente Simétrico': self.aplicar_grad_simetrico,
            'Gradiente Externo': self.aplicar_grad_externo,
            'Gradiente Interno': self.aplicar_grad_interno,
            'Top Hat': self.aplicar_top_hat,
            'Black Hat': self.aplicar_black_hat,
            'Filtro Combinado': self.aplicar_filtro_combinado,
            'Negación': self.aplicar_negacion,
        }
        
        self.crear_interfaz()
    
    def crear_interfaz(self):
        """Crea la interfaz gráfica completa"""
        
        # ===== FRAME SUPERIOR: TÍTULO =====
        frame_titulo = tk.Frame(self.root, bg='#2c3e50', height=60)
        frame_titulo.pack(fill=tk.X, padx=10, pady=10)
        
        titulo = tk.Label(
            frame_titulo, 
            text="ANALISIS DE IMAGENES - OPERACIONES MORFOLOGICAS",
            font=('Arial', 18, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        titulo.pack(pady=15)
        
        # ===== FRAME PRINCIPAL CON DOS COLUMNAS =====
        frame_principal = tk.Frame(self.root, bg='#f0f0f0')
        frame_principal.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Columna izquierda: Carga de imágenes y parámetros
        frame_izquierda = tk.Frame(frame_principal, bg='#f0f0f0', width=400)
        frame_izquierda.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # Columna derecha: Visualización de resultados
        frame_derecha = tk.Frame(frame_principal, bg='white')
        frame_derecha.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ===== SECCIÓN: CARGA DE IMAGEN =====
        self.crear_seccion_carga_imagen(frame_izquierda)
        
        # ===== SECCIÓN: PARÁMETROS =====
        self.crear_seccion_parametros(frame_izquierda)
        
        # ===== SECCIÓN: OPERACIONES =====
        self.crear_seccion_operaciones(frame_izquierda)
        
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
            bg='#3498db',
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
    
    def crear_seccion_parametros(self, parent):
        """Crea la sección de parámetros del kernel"""
        frame = tk.LabelFrame(
            parent,
            text="Parametros del Elemento Estructurante",
            font=('Arial', 11, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        frame.pack(fill=tk.X, pady=(0, 10))
        
        # Tamaño del kernel
        label_size = tk.Label(
            frame,
            text="Tamaño del Kernel (Rectangular):",
            font=('Arial', 10),
            bg='#ecf0f1'
        )
        label_size.pack(anchor=tk.W, pady=(5, 0))
        
        frame_size = tk.Frame(frame, bg='#ecf0f1')
        frame_size.pack(fill=tk.X, pady=5)
        
        scale_size = tk.Scale(
            frame_size,
            from_=3,
            to=256,
            orient=tk.HORIZONTAL,
            variable=self.kernel_size,
            bg='#ecf0f1',
            font=('Arial', 9)
        )
        scale_size.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def crear_seccion_operaciones(self, parent):
        """Crea la sección de operaciones morfológicas"""
        frame = tk.LabelFrame(
            parent,
            text="Operaciones Morfologicas",
            font=('Arial', 11, 'bold'),
            bg='#e8f4f8',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        frame.pack(fill=tk.X, pady=(0, 10))
        
        # Usar las claves del diccionario de operaciones
        col = 0
        row = 0
        
        for op in self.operaciones.keys():
            btn = tk.Button(
                frame,
                text=op,
                command=lambda o=op: self.ejecutar_operacion_ambas(o),
                bg='#3498db',
                fg='white',
                font=('Arial', 9),
                width=20,
                cursor='hand2'
            )
            btn.grid(row=row, column=col, padx=5, pady=3, sticky='ew')
            
            col += 1
            if col > 1:  # 2 columnas
                col = 0
                row += 1
    
    def crear_seccion_visualizacion(self, parent):
        """Crea la sección de visualización de resultados"""
        label = tk.Label(
            parent,
            text="Resultados de las Operaciones",
            font=('Arial', 12, 'bold'),
            bg='white'
        )
        label.pack(pady=10)
        
        # Frame para el canvas de matplotlib
        self.frame_canvas = tk.Frame(parent, bg='white')
        self.frame_canvas.pack(fill=tk.BOTH, expand=True)
    
    # ===== MÉTODOS DE CARGA DE IMÁGENES =====
    
    def cargar_imagen(self):
        """Carga una imagen y detecta automáticamente si es binaria o en escala de grises"""
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
            
            # Detectar automáticamente el tipo de imagen
            valores_unicos = np.unique(self.imagen)
            num_valores = len(valores_unicos)
            
            if num_valores <= 2:
                self.tipo_imagen = 'binaria'
                tipo_texto = "BINARIA"
                color_tipo = '#e74c3c'  # Rojo
                info_extra = f"Valores: {valores_unicos.tolist()}"
            else:
                self.tipo_imagen = 'grises'
                tipo_texto = "ESCALA DE GRISES"
                color_tipo = '#27ae60'  # Verde
                info_extra = f"Rango: [{self.imagen.min()}-{self.imagen.max()}]"
            
            nombre_archivo = os.path.basename(ruta)
            self.label_imagen.config(
                text=f"{nombre_archivo}\n{self.imagen.shape[1]}x{self.imagen.shape[0]} px\n[{tipo_texto}] - {info_extra}",
                fg=color_tipo
            )
            
            messagebox.showinfo(
                "Éxito", 
                f"Imagen cargada correctamente:\n{nombre_archivo}\n\nTipo: {tipo_texto}\n{info_extra}"
            )
    
    def guardar_resultado(self):
        """Guarda la imagen resultado con el filtro aplicado"""
        if self.imagen_resultado is None:
            messagebox.showwarning("Advertencia", "No hay ningún resultado para guardar.\nPrimero aplique una operación.")
            return
        
        # Sugerir nombre de archivo
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
    
    # ===== MÉTODOS PARA CREAR KERNEL =====
    
    def crear_kernel(self):
        """Crea el elemento estructurante rectangular según el tamaño seleccionado"""
        size = self.kernel_size.get()
        # Asegurar que el tamaño sea impar
        if size % 2 == 0:
            size += 1
        
        # Siempre rectangular
        kernel = np.ones((size, size), np.uint8)
        
        return kernel
    
    # ===== MÉTODOS DE OPERACIONES MORFOLÓGICAS =====
    
    def aplicar_erosion(self, imagen):
        """Erosión: Reduce/oscurece objetos"""
        kernel = self.crear_kernel()
        return cv2.erode(imagen, kernel)
    
    def aplicar_dilatacion(self, imagen):
        """Dilatación: Expande/aclara objetos"""
        kernel = self.crear_kernel()
        return cv2.dilate(imagen, kernel)
    
    def aplicar_apertura(self, imagen):
        """Apertura: Erosión seguida de Dilatación"""
        kernel = self.crear_kernel()
        return cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
    
    def aplicar_cierre(self, imagen):
        """Cierre: Dilatación seguida de Erosión"""
        kernel = self.crear_kernel()
        return cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)
    
    def aplicar_frontera(self, imagen):
        """Frontera: Diferencia entre imagen y su erosión"""
        kernel = self.crear_kernel()
        erosion = cv2.erode(imagen, kernel)
        return cv2.subtract(imagen, erosion)
    
    def aplicar_adelgazamiento(self, imagen):
        """Adelgazamiento (Thinning): Reduce el grosor de objetos"""
        kernel = np.ones((3, 3), np.uint8)
        resultado = imagen.copy()
        for _ in range(1):
            erosion = cv2.erode(resultado, kernel)
            temp = cv2.dilate(erosion, kernel)
            temp = cv2.subtract(resultado, temp)
            resultado = erosion
        return resultado
    
    def aplicar_hit_or_miss(self, imagen):
        """Transformada Hit-or-Miss: Detecta patrones específicos"""
        kernel = np.ones((3, 3), np.uint8)
        hit = cv2.erode(imagen, kernel)
        miss = cv2.erode(cv2.bitwise_not(imagen), kernel)
        return cv2.bitwise_and(hit, miss)
    
    def aplicar_esqueleto(self, imagen):
        """Esqueleto Morfológico: Reduce la forma a su representación esquelética"""
        kernel = np.ones((3, 3), np.uint8)
        skeleton = np.zeros(imagen.shape, np.uint8)
        temp = imagen.copy()
        max_iteraciones = 100  # Límite para evitar bucles infinitos
        iteracion = 0
        
        while iteracion < max_iteraciones:
            eroded = cv2.erode(temp, kernel)
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
            subset = cv2.subtract(eroded, opened)
            skeleton = cv2.bitwise_or(skeleton, subset)
            temp = eroded.copy()
            
            if cv2.countNonZero(temp) == 0:
                break
            iteracion += 1
        
        return skeleton
    
    def aplicar_grad_simetrico(self, imagen):
        """Gradiente morfológico simétrico: Dilatación - Erosión"""
        kernel = self.crear_kernel()
        dilatacion = cv2.dilate(imagen, kernel)
        erosion = cv2.erode(imagen, kernel)
        return cv2.subtract(dilatacion, erosion)
    
    def aplicar_grad_externo(self, imagen):
        """Gradiente externo: Dilatación - Original"""
        kernel = self.crear_kernel()
        dilatacion = cv2.dilate(imagen, kernel)
        return cv2.subtract(dilatacion, imagen)
    
    def aplicar_grad_interno(self, imagen):
        """Gradiente interno: Original - Erosión"""
        kernel = self.crear_kernel()
        erosion = cv2.erode(imagen, kernel)
        return cv2.subtract(imagen, erosion)
    
    def aplicar_top_hat(self, imagen):
        """Top Hat: Original - Apertura (resalta objetos claros)"""
        kernel = self.crear_kernel()
        return cv2.morphologyEx(imagen, cv2.MORPH_TOPHAT, kernel)
    
    def aplicar_black_hat(self, imagen):
        """Black Hat: Cierre - Original (resalta objetos oscuros)"""
        kernel = self.crear_kernel()
        return cv2.morphologyEx(imagen, cv2.MORPH_BLACKHAT, kernel)
    
    def aplicar_filtro_combinado(self, imagen):
        """Filtro de suavizado: Apertura seguida de Cierre"""
        kernel = self.crear_kernel()
        apertura = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(apertura, cv2.MORPH_CLOSE, kernel)
    
    def aplicar_negacion(self, imagen):
        """Negación: Invierte los valores de la imagen (complemento)"""
        return cv2.bitwise_not(imagen)
    
    # ===== MÉTODOS DE EJECUCIÓN =====
    
    def ejecutar_operacion_ambas(self, operacion):
        """Ejecuta una operación morfológica sobre la imagen cargada"""
        if self.imagen is None:
            messagebox.showwarning("Advertencia", "Debe cargar una imagen primero")
            return
        
        try:
            # Aplicar la operación
            resultado = self.operaciones[operacion](self.imagen)
            
            # Guardar el resultado y el nombre de la operación
            self.imagen_resultado = resultado
            self.nombre_operacion = operacion.replace(' ', '_').replace('ó', 'o').replace('é', 'e').replace('í', 'i')
            
            # Visualizar
            tipo_texto = "Binaria" if self.tipo_imagen == 'binaria' else "Grises"
            self.visualizar_resultado_simple(self.imagen, resultado, f"{tipo_texto} - {operacion}", operacion)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al aplicar {operacion}:\n{str(e)}")
    
    def visualizar_resultado_simple(self, imagen_original, imagen_resultado, titulo, operacion):
        """Visualiza una sola imagen con su resultado"""
        # Limpiar el frame anterior
        for widget in self.frame_canvas.winfo_children():
            widget.destroy()
        
        # Crear figura de matplotlib
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(titulo, fontsize=14, fontweight='bold')
        
        # Imagen original
        axes[0].imshow(imagen_original, cmap='gray')
        axes[0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Imagen resultado
        axes[1].imshow(imagen_resultado, cmap='gray')
        axes[1].set_title(f'{operacion}', fontsize=12, fontweight='bold', color='blue')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Integrar matplotlib en tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.frame_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def cerrar_aplicacion(self):
        """Cierra correctamente la aplicación liberando todos los recursos"""
        plt.close('all')
        self.root.quit()
        self.root.destroy()


# ===================================================================
# FUNCIÓN PRINCIPAL
# ===================================================================

def main():
    root = tk.Tk()
    app = InterfazMorfologia(root)
    root.mainloop()


if __name__ == "__main__":
    main()
