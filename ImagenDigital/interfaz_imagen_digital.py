"""
Interfaz Gráfica para Procesamiento Básico de Imágenes Digitales
- Conversión RGB a escala de grises
- Binarización con umbral fijo y automático
- Visualización de histogramas
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
    calcular_histograma
)


class InterfazImagenDigital:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesamiento Básico de Imágenes Digitales")
        self.root.geometry("1400x800")
        
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
        
        # Panel izquierdo - Operaciones
        panel_izquierdo = ttk.LabelFrame(frame_contenido, text="Operaciones", padding=10)
        panel_izquierdo.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Conversión a escala de grises
        ttk.Label(panel_izquierdo, text="1. Conversión a Escala de Grises", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        ttk.Button(panel_izquierdo, text="Convertir a Grises",
                  command=self.convertir_a_grises, width=30).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Separator(panel_izquierdo, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # Binarización
        ttk.Label(panel_izquierdo, text="2. Binarización", 
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
                  command=self.aplicar_umbral_fijo, width=30).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Separator(panel_izquierdo, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Umbral automático (Otsu)
        ttk.Label(panel_izquierdo, text="Umbral Automático (Otsu):").pack(pady=(5, 2))
        
        ttk.Button(panel_izquierdo, text="Aplicar Umbral Automático",
                  command=self.aplicar_umbral_otsu, width=30).pack(fill=tk.X, padx=10, pady=5)
        
        self.label_umbral_calculado = ttk.Label(panel_izquierdo, text="", 
                                               font=('Arial', 9, 'italic'))
        self.label_umbral_calculado.pack(pady=5)
        
        ttk.Separator(panel_izquierdo, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # Histogramas
        ttk.Label(panel_izquierdo, text="3. Visualización", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        ttk.Button(panel_izquierdo, text="Mostrar Histogramas",
                  command=self.mostrar_histogramas, width=30).pack(fill=tk.X, padx=10, pady=5)
        
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
            
            # Mostrar imagen original
            self.mostrar_imagen(self.imagen_original, self.label_procesada)
            self.actualizar_layout('simple')
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


def main():
    root = tk.Tk()
    app = InterfazImagenDigital(root)
    root.mainloop()


if __name__ == "__main__":
    main()
