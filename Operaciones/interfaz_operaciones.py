"""
Interfaz Gráfica para Operaciones sobre Imágenes
- Operaciones con escalares (suma, resta, multiplicación)
- Operaciones lógicas entre imágenes (AND, OR, XOR, NOT)
- Operaciones aritméticas entre imágenes (suma, resta, multiplicación, división)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from procesamiento_operaciones import (
    suma_escalar, resta_escalar, multiplicacion_escalar, division_escalar,
    operacion_and, operacion_or, operacion_xor, operacion_not,
    suma_imagenes, resta_imagenes, multiplicacion_imagenes, 
    division_imagenes, diferencia_absoluta
)


class InterfazOperaciones:
    def __init__(self, root):
        self.root = root
        self.root.title("Operaciones sobre Imágenes")
        self.root.geometry("1500x850")
        
        self.imagen1 = None
        self.imagen2 = None
        self.imagen_resultado = None
        
        self.crear_interfaz()
    
    def crear_interfaz(self):
        """Crea la interfaz gráfica principal"""
        
        # Frame principal
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel superior - Controles
        panel_superior = ttk.LabelFrame(main_container, text="Controles", padding=10)
        panel_superior.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(panel_superior, text="Cargar Imagen 1", 
                  command=self.cargar_imagen1).pack(side=tk.LEFT, padx=5)
        ttk.Button(panel_superior, text="Cargar Imagen 2", 
                  command=self.cargar_imagen2).pack(side=tk.LEFT, padx=5)
        ttk.Button(panel_superior, text="Guardar Resultado", 
                  command=self.guardar_resultado).pack(side=tk.LEFT, padx=5)
        ttk.Button(panel_superior, text="Reiniciar", 
                  command=self.reiniciar).pack(side=tk.LEFT, padx=5)
        
        # Frame contenedor
        frame_contenido = ttk.Frame(main_container)
        frame_contenido.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo - Operaciones
        panel_izquierdo = ttk.LabelFrame(frame_contenido, text="Operaciones", padding=10)
        panel_izquierdo.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Canvas con scrollbar para operaciones
        canvas = tk.Canvas(panel_izquierdo, width=380)
        scrollbar = ttk.Scrollbar(panel_izquierdo, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # ===== OPERACIONES CON ESCALARES =====
        ttk.Label(scrollable_frame, text="1. Operaciones con Escalar", 
                 font=('Arial', 11, 'bold')).pack(pady=(10, 5))
        
        ttk.Label(scrollable_frame, text="Valor escalar:").pack(pady=(5, 2))
        
        frame_escalar = ttk.Frame(scrollable_frame)
        frame_escalar.pack(fill=tk.X, padx=10, pady=5)
        
        self.var_escalar = tk.DoubleVar(value=50)
        self.label_escalar = ttk.Label(frame_escalar, text="50")
        self.label_escalar.pack(side=tk.RIGHT)
        
        scale_escalar = ttk.Scale(frame_escalar, from_=-255, to=255, 
                                 variable=self.var_escalar, orient=tk.HORIZONTAL,
                                 command=self.actualizar_label_escalar)
        scale_escalar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(scrollable_frame, text="Suma Escalar",
                  command=lambda: self.aplicar_operacion_escalar('suma'), 
                  width=35).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="Resta Escalar",
                  command=lambda: self.aplicar_operacion_escalar('resta'), 
                  width=35).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="Multiplicación Escalar",
                  command=lambda: self.aplicar_operacion_escalar('multiplicacion'), 
                  width=35).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="División Escalar",
                  command=lambda: self.aplicar_operacion_escalar('division'), 
                  width=35).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # ===== OPERACIONES LÓGICAS =====
        ttk.Label(scrollable_frame, text="2. Operaciones Lógicas", 
                 font=('Arial', 11, 'bold')).pack(pady=(10, 5))
        
        ttk.Label(scrollable_frame, text="(Requiere 2 imágenes)", 
                 font=('Arial', 9, 'italic')).pack(pady=2)
        
        ttk.Button(scrollable_frame, text="AND (Imagen1 AND Imagen2)",
                  command=lambda: self.aplicar_operacion_logica('and'), 
                  width=35).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="OR (Imagen1 OR Imagen2)",
                  command=lambda: self.aplicar_operacion_logica('or'), 
                  width=35).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="XOR (Imagen1 XOR Imagen2)",
                  command=lambda: self.aplicar_operacion_logica('xor'), 
                  width=35).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="NOT (Inversión de Imagen1)",
                  command=lambda: self.aplicar_operacion_logica('not'), 
                  width=35).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # ===== OPERACIONES ARITMÉTICAS =====
        ttk.Label(scrollable_frame, text="3. Operaciones Aritméticas", 
                 font=('Arial', 11, 'bold')).pack(pady=(10, 5))
        
        ttk.Label(scrollable_frame, text="(Requiere 2 imágenes)", 
                 font=('Arial', 9, 'italic')).pack(pady=2)
        
        # Pesos para suma ponderada
        ttk.Label(scrollable_frame, text="Peso Imagen 1:").pack(pady=(10, 2))
        
        frame_peso1 = ttk.Frame(scrollable_frame)
        frame_peso1.pack(fill=tk.X, padx=10, pady=2)
        
        self.var_peso1 = tk.DoubleVar(value=0.5)
        self.label_peso1 = ttk.Label(frame_peso1, text="0.5")
        self.label_peso1.pack(side=tk.RIGHT)
        
        scale_peso1 = ttk.Scale(frame_peso1, from_=0.0, to=1.0, 
                               variable=self.var_peso1, orient=tk.HORIZONTAL,
                               command=self.actualizar_label_peso1)
        scale_peso1.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(scrollable_frame, text="Peso Imagen 2:").pack(pady=(5, 2))
        
        frame_peso2 = ttk.Frame(scrollable_frame)
        frame_peso2.pack(fill=tk.X, padx=10, pady=2)
        
        self.var_peso2 = tk.DoubleVar(value=0.5)
        self.label_peso2 = ttk.Label(frame_peso2, text="0.5")
        self.label_peso2.pack(side=tk.RIGHT)
        
        scale_peso2 = ttk.Scale(frame_peso2, from_=0.0, to=1.0, 
                               variable=self.var_peso2, orient=tk.HORIZONTAL,
                               command=self.actualizar_label_peso2)
        scale_peso2.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(scrollable_frame, text="Suma (Ponderada)",
                  command=lambda: self.aplicar_operacion_aritmetica('suma'), 
                  width=35).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(scrollable_frame, text="Resta (Img1 - Img2)",
                  command=lambda: self.aplicar_operacion_aritmetica('resta'), 
                  width=35).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="Multiplicación",
                  command=lambda: self.aplicar_operacion_aritmetica('multiplicacion'), 
                  width=35).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="División (Img1 / Img2)",
                  command=lambda: self.aplicar_operacion_aritmetica('division'), 
                  width=35).pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Button(scrollable_frame, text="Diferencia Absoluta |Img1-Img2|",
                  command=lambda: self.aplicar_operacion_aritmetica('diferencia'), 
                  width=35).pack(fill=tk.X, padx=10, pady=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Panel derecho - Visualización
        panel_derecho = ttk.LabelFrame(frame_contenido, text="Visualización", padding=10)
        panel_derecho.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.frame_visualizacion = ttk.Frame(panel_derecho)
        self.frame_visualizacion.pack(fill=tk.BOTH, expand=True)
        
        # Labels para imágenes
        self.label_img1 = ttk.Label(self.frame_visualizacion)
        self.label_img2 = ttk.Label(self.frame_visualizacion)
        self.label_resultado = ttk.Label(self.frame_visualizacion)
        
        self.actualizar_layout()
    
    def actualizar_layout(self):
        """Actualiza el layout de visualización"""
        for widget in self.frame_visualizacion.winfo_children():
            widget.grid_forget()
        
        # Layout con 3 columnas
        ttk.Label(self.frame_visualizacion, text="Imagen 1", 
                 font=('Arial', 11, 'bold')).grid(row=0, column=0, pady=5)
        self.label_img1.grid(row=1, column=0, padx=10, pady=5)
        
        ttk.Label(self.frame_visualizacion, text="Imagen 2", 
                 font=('Arial', 11, 'bold')).grid(row=0, column=1, pady=5)
        self.label_img2.grid(row=1, column=1, padx=10, pady=5)
        
        ttk.Label(self.frame_visualizacion, text="Resultado", 
                 font=('Arial', 11, 'bold')).grid(row=0, column=2, pady=5)
        self.label_resultado.grid(row=1, column=2, padx=10, pady=5)
        
        self.frame_visualizacion.grid_rowconfigure(1, weight=1)
        self.frame_visualizacion.grid_columnconfigure(0, weight=1)
        self.frame_visualizacion.grid_columnconfigure(1, weight=1)
        self.frame_visualizacion.grid_columnconfigure(2, weight=1)
    
    def actualizar_label_escalar(self, valor):
        """Actualiza el label del valor escalar"""
        self.label_escalar.config(text=f"{float(valor):.1f}")
    
    def actualizar_label_peso1(self, valor):
        """Actualiza el label del peso 1"""
        self.label_peso1.config(text=f"{float(valor):.2f}")
    
    def actualizar_label_peso2(self, valor):
        """Actualiza el label del peso 2"""
        self.label_peso2.config(text=f"{float(valor):.2f}")
    
    def cargar_imagen1(self):
        """Carga la primera imagen"""
        ruta = filedialog.askopenfilename(
            title="Seleccionar Imagen 1",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), 
                      ("Todos los archivos", "*.*")]
        )
        
        if ruta:
            self.imagen1 = cv2.imread(ruta)
            if self.imagen1 is None:
                messagebox.showerror("Error", "No se pudo cargar la imagen 1")
                return
            
            self.mostrar_imagen(self.imagen1, self.label_img1)
            messagebox.showinfo("Éxito", "Imagen 1 cargada correctamente")
    
    def cargar_imagen2(self):
        """Carga la segunda imagen"""
        ruta = filedialog.askopenfilename(
            title="Seleccionar Imagen 2",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), 
                      ("Todos los archivos", "*.*")]
        )
        
        if ruta:
            self.imagen2 = cv2.imread(ruta)
            if self.imagen2 is None:
                messagebox.showerror("Error", "No se pudo cargar la imagen 2")
                return
            
            self.mostrar_imagen(self.imagen2, self.label_img2)
            messagebox.showinfo("Éxito", "Imagen 2 cargada correctamente")
    
    def mostrar_imagen(self, imagen, label, tamano_max=(350, 550)):
        """Muestra una imagen en un label"""
        if imagen is None:
            return
        
        # Convertir a RGB
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
    
    def aplicar_operacion_escalar(self, operacion):
        """Aplica operación con escalar sobre imagen 1"""
        if self.imagen1 is None:
            messagebox.showwarning("Advertencia", "Primero carga la Imagen 1")
            return
        
        try:
            valor = self.var_escalar.get()
            
            if operacion == 'suma':
                self.imagen_resultado = suma_escalar(self.imagen1, valor)
                op_texto = f"Suma con {valor:.1f}"
            elif operacion == 'resta':
                self.imagen_resultado = resta_escalar(self.imagen1, valor)
                op_texto = f"Resta de {valor:.1f}"
            elif operacion == 'multiplicacion':
                self.imagen_resultado = multiplicacion_escalar(self.imagen1, valor)
                op_texto = f"Multiplicación por {valor:.1f}"
            elif operacion == 'division':
                self.imagen_resultado = division_escalar(self.imagen1, valor)
                op_texto = f"División por {valor:.1f}"
            
            self.mostrar_imagen(self.imagen_resultado, self.label_resultado)
            messagebox.showinfo("Éxito", f"Operación aplicada: {op_texto}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en operación: {str(e)}")
    
    def aplicar_operacion_logica(self, operacion):
        """Aplica operación lógica"""
        if operacion == 'not':
            if self.imagen1 is None:
                messagebox.showwarning("Advertencia", "Primero carga la Imagen 1")
                return
        else:
            if self.imagen1 is None or self.imagen2 is None:
                messagebox.showwarning("Advertencia", "Primero carga ambas imágenes")
                return
        
        try:
            if operacion == 'and':
                self.imagen_resultado = operacion_and(self.imagen1, self.imagen2)
                op_texto = "AND lógico"
            elif operacion == 'or':
                self.imagen_resultado = operacion_or(self.imagen1, self.imagen2)
                op_texto = "OR lógico"
            elif operacion == 'xor':
                self.imagen_resultado = operacion_xor(self.imagen1, self.imagen2)
                op_texto = "XOR lógico"
            elif operacion == 'not':
                self.imagen_resultado = operacion_not(self.imagen1)
                op_texto = "NOT lógico (inversión)"
            
            self.mostrar_imagen(self.imagen_resultado, self.label_resultado)
            messagebox.showinfo("Éxito", f"Operación aplicada: {op_texto}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en operación lógica: {str(e)}")
    
    def aplicar_operacion_aritmetica(self, operacion):
        """Aplica operación aritmética entre dos imágenes"""
        if self.imagen1 is None or self.imagen2 is None:
            messagebox.showwarning("Advertencia", "Primero carga ambas imágenes")
            return
        
        try:
            if operacion == 'suma':
                peso1 = self.var_peso1.get()
                peso2 = self.var_peso2.get()
                self.imagen_resultado = suma_imagenes(self.imagen1, self.imagen2, peso1, peso2)
                op_texto = f"Suma ponderada (w1={peso1:.2f}, w2={peso2:.2f})"
            elif operacion == 'resta':
                self.imagen_resultado = resta_imagenes(self.imagen1, self.imagen2)
                op_texto = "Resta (Img1 - Img2)"
            elif operacion == 'multiplicacion':
                self.imagen_resultado = multiplicacion_imagenes(self.imagen1, self.imagen2)
                op_texto = "Multiplicación"
            elif operacion == 'division':
                self.imagen_resultado = division_imagenes(self.imagen1, self.imagen2)
                op_texto = "División (Img1 / Img2)"
            elif operacion == 'diferencia':
                self.imagen_resultado = diferencia_absoluta(self.imagen1, self.imagen2)
                op_texto = "Diferencia absoluta |Img1-Img2|"
            
            self.mostrar_imagen(self.imagen_resultado, self.label_resultado)
            messagebox.showinfo("Éxito", f"Operación aplicada: {op_texto}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en operación aritmética: {str(e)}")
    
    def guardar_resultado(self):
        """Guarda la imagen resultado"""
        if self.imagen_resultado is None:
            messagebox.showwarning("Advertencia", "No hay resultado para guardar")
            return
        
        ruta = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos", "*.*")]
        )
        
        if ruta:
            cv2.imwrite(ruta, self.imagen_resultado)
            messagebox.showinfo("Éxito", "Resultado guardado correctamente")
    
    def reiniciar(self):
        """Reinicia el estado de la aplicación"""
        self.imagen1 = None
        self.imagen2 = None
        self.imagen_resultado = None
        
        self.label_img1.config(image='')
        self.label_img2.config(image='')
        self.label_resultado.config(image='')


def main():
    root = tk.Tk()
    app = InterfazOperaciones(root)
    root.mainloop()


if __name__ == "__main__":
    main()
