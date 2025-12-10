"""
Interfaz gráfica para aplicar mapas de calor a imágenes
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from logica_heatmap import HeatMapProcessor


class HeatMapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplicación de Mapas de Calor")
        self.root.geometry("1400x800")
        
        self.processor = HeatMapProcessor()
        self.imagen_original = None
        self.imagen_path = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar peso de filas y columnas
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Frame de controles
        control_frame = ttk.LabelFrame(main_frame, text="Controles", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.N, tk.S, tk.W), padx=(0, 10))
        
        # Botón cargar imagen
        ttk.Button(control_frame, text="Cargar Imagen", 
                   command=self.cargar_imagen).grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Separador
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Label para mapas de calor
        ttk.Label(control_frame, text="Mapas de Calor OpenCV:", 
                  font=('Arial', 10, 'bold')).grid(row=2, column=0, pady=(5, 10), sticky=tk.W)
        
        # Botones para cada mapa de calor de OpenCV
        self.crear_botones_opencv(control_frame, start_row=3)
        
        # Separador
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=20, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Label para mapa personalizado
        ttk.Label(control_frame, text="Mapa Personalizado:", 
                  font=('Arial', 10, 'bold')).grid(row=21, column=0, pady=(5, 10), sticky=tk.W)
        
        # Botón para mapa personalizado pastel
        ttk.Button(control_frame, text="Mapa Pastel", 
                   command=lambda: self.aplicar_mapa("PASTEL")).grid(row=22, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Separador
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=23, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Botón para limpiar
        ttk.Button(control_frame, text="Limpiar Todo", 
                   command=self.limpiar_todo).grid(row=24, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Frame para imagen original
        original_frame = ttk.LabelFrame(main_frame, text="Imagen Original", padding="10")
        original_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.W, tk.E), padx=5, pady=5)
        original_frame.columnconfigure(0, weight=1)
        original_frame.rowconfigure(0, weight=1)
        
        self.canvas_original = tk.Canvas(original_frame, bg='gray80')
        self.canvas_original.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        
        # Frame para imagen procesada
        procesada_frame = ttk.LabelFrame(main_frame, text="Imagen con Mapa de Calor", padding="10")
        procesada_frame.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.W, tk.E), padx=5, pady=5)
        procesada_frame.columnconfigure(0, weight=1)
        procesada_frame.rowconfigure(0, weight=1)
        
        self.canvas_procesada = tk.Canvas(procesada_frame, bg='gray80')
        self.canvas_procesada.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        
        # Label de información
        self.info_label = ttk.Label(main_frame, text="Carga una imagen para comenzar", 
                                     font=('Arial', 9))
        self.info_label.grid(row=2, column=0, columnspan=2, pady=5)
    
    def crear_botones_opencv(self, parent, start_row):
        """Crear botones para todos los mapas de calor de OpenCV"""
        mapas_opencv = [
            ("AUTUMN", "Autumn"),
            ("BONE", "Bone"),
            ("JET", "Jet"),
            ("WINTER", "Winter"),
            ("RAINBOW", "Rainbow"),
            ("OCEAN", "Ocean"),
            ("SUMMER", "Summer"),
            ("SPRING", "Spring"),
            ("COOL", "Cool"),
            ("HSV", "HSV"),
            ("PINK", "Pink"),
            ("HOT", "Hot"),
            ("PARULA", "Parula"),
            ("MAGMA", "Magma"),
            ("INFERNO", "Inferno"),
            ("PLASMA", "Plasma"),
            ("VIRIDIS", "Viridis"),
            ("CIVIDIS", "Cividis"),
            ("TWILIGHT", "Twilight"),
            ("TURBO", "Turbo")
        ]
        
        row = start_row
        for mapa_id, mapa_nombre in mapas_opencv:
            ttk.Button(parent, text=mapa_nombre, 
                       command=lambda m=mapa_id: self.aplicar_mapa(m)).grid(
                row=row, column=0, pady=2, sticky=(tk.W, tk.E))
            row += 1
    
    def cargar_imagen(self):
        """Cargar una imagen desde el sistema de archivos"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar Imagen",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Cargar imagen con OpenCV
                self.imagen_original = cv2.imread(file_path)
                if self.imagen_original is None:
                    raise ValueError("No se pudo cargar la imagen")
                
                self.imagen_path = file_path
                
                # Mostrar imagen original
                self.mostrar_imagen(self.imagen_original, self.canvas_original)
                
                # Limpiar canvas procesada
                self.canvas_procesada.delete("all")
                
                self.info_label.config(text=f"Imagen cargada: {file_path.split('/')[-1]}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar la imagen:\n{str(e)}")
    
    def aplicar_mapa(self, mapa_tipo):
        """Aplicar un mapa de calor a la imagen"""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Por favor, carga una imagen primero")
            return
        
        try:
            imagen_procesada = self.processor.aplicar_mapa_calor(
                self.imagen_original.copy(), mapa_tipo
            )
            
            self.mostrar_imagen(imagen_procesada, self.canvas_procesada)
            self.info_label.config(text=f"Mapa aplicado: {mapa_tipo}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al aplicar mapa de calor:\n{str(e)}")
    
    def mostrar_imagen(self, imagen_cv, canvas):
        """Mostrar imagen en un canvas"""
        # Convertir de BGR a RGB
        imagen_rgb = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2RGB)
        
        # Obtener dimensiones del canvas
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Redimensionar imagen manteniendo aspecto
        imagen_pil = Image.fromarray(imagen_rgb)
        img_width, img_height = imagen_pil.size
        
        # Calcular escala
        scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Redimensionar
        imagen_pil = imagen_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convertir a PhotoImage
        imagen_tk = ImageTk.PhotoImage(imagen_pil)
        
        # Limpiar canvas y mostrar imagen
        canvas.delete("all")
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        canvas.create_image(x, y, anchor=tk.NW, image=imagen_tk)
        
        # Guardar referencia para evitar garbage collection
        canvas.image = imagen_tk
    
    def limpiar_todo(self):
        """Limpiar todas las imágenes"""
        self.imagen_original = None
        self.imagen_path = None
        self.canvas_original.delete("all")
        self.canvas_procesada.delete("all")
        self.info_label.config(text="Carga una imagen para comenzar")


def main():
    root = tk.Tk()
    app = HeatMapApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
