"""
Interfaz Gráfica para Análisis de Componentes Conexos
Permite analizar imágenes binarias usando conectividad 4 y 8
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from logica_componentes import (
    componentes_conexos_4,
    componentes_conexos_8,
    comparar_conectividades,
    dibujar_componentes_con_info,
    binarizar_imagen,
    filtrar_por_area,
    obtener_resumen_estadistico
)


class InterfazComponentesConexos:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis de Componentes Conexos")
        self.root.geometry("1600x900")
        
        self.imagen_original = None
        self.imagen_binaria = None
        self.resultados_4 = None
        self.resultados_8 = None
        
        self.crear_interfaz()
    
    def crear_interfaz(self):
        """Crea la interfaz gráfica principal"""
        
        # Frame principal
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel superior - Controles
        panel_superior = ttk.LabelFrame(main_container, text="Control Principal", padding=10)
        panel_superior.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(panel_superior, text="Cargar Imagen", 
                  command=self.cargar_imagen).pack(side=tk.LEFT, padx=5)
        ttk.Button(panel_superior, text="Guardar Resultado", 
                  command=self.guardar_resultado).pack(side=tk.LEFT, padx=5)
        ttk.Button(panel_superior, text="Reiniciar", 
                  command=self.reiniciar).pack(side=tk.LEFT, padx=5)
        
        # Frame contenedor
        frame_contenido = ttk.Frame(main_container)
        frame_contenido.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo - Operaciones
        panel_izquierdo = ttk.LabelFrame(frame_contenido, text="Análisis", padding=10)
        panel_izquierdo.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Sección de binarización
        ttk.Label(panel_izquierdo, text="1. Binarización", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        ttk.Label(panel_izquierdo, text="Umbral:").pack(pady=(5, 2))
        self.var_umbral = tk.IntVar(value=127)
        self.label_umbral = ttk.Label(panel_izquierdo, text="127")
        self.label_umbral.pack()
        
        scale_umbral = ttk.Scale(panel_izquierdo, from_=0, to=255,
                                variable=self.var_umbral, orient=tk.HORIZONTAL,
                                command=self.actualizar_label_umbral)
        scale_umbral.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(panel_izquierdo, text="Binarizar Imagen",
                  command=self.binarizar, width=30).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Separator(panel_izquierdo, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # Sección de análisis
        ttk.Label(panel_izquierdo, text="2. Análisis de Conectividad", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        ttk.Button(panel_izquierdo, text="Conectividad 4",
                  command=self.analizar_conectividad_4, width=30).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(panel_izquierdo, text="Conectividad 8",
                  command=self.analizar_conectividad_8, width=30).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(panel_izquierdo, text="Comparar 4 vs 8",
                  command=self.comparar_conectividades, width=30).pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Separator(panel_izquierdo, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # Sección de resultados
        ttk.Label(panel_izquierdo, text="Resultados:", 
                 font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        self.text_resultados = tk.Text(panel_izquierdo, height=15, width=35, 
                                      font=('Courier', 9))
        self.text_resultados.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = ttk.Scrollbar(panel_izquierdo, command=self.text_resultados.yview)
        self.text_resultados.config(yscrollcommand=scrollbar.set)
        
        # Panel derecho - Visualización
        panel_derecho = ttk.LabelFrame(frame_contenido, text="Visualización", padding=10)
        panel_derecho.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.frame_visualizacion = ttk.Frame(panel_derecho)
        self.frame_visualizacion.pack(fill=tk.BOTH, expand=True)
        
        self.label_imagen = ttk.Label(self.frame_visualizacion)
        self.label_imagen.pack(fill=tk.BOTH, expand=True)
    
    def actualizar_label_umbral(self, valor):
        """Actualiza el label del umbral"""
        self.label_umbral.config(text=str(int(float(valor))))
    
    def cargar_imagen(self):
        """Carga una imagen desde el disco"""
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), 
                      ("Todos los archivos", "*.*")]
        )
        
        if ruta:
            self.imagen_original = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
            if self.imagen_original is None:
                messagebox.showerror("Error", "No se pudo cargar la imagen")
                return
            
            # Resetear estado
            self.imagen_binaria = None
            self.resultados_4 = None
            self.resultados_8 = None
            self.text_resultados.delete('1.0', tk.END)
            
            # Mostrar imagen
            self.mostrar_imagen(self.imagen_original)
            messagebox.showinfo("Éxito", "Imagen cargada correctamente")
    
    def mostrar_imagen(self, imagen, es_color=False):
        """Muestra una imagen en el panel de visualización"""
        if imagen is None:
            return
        
        # Limpiar frame
        for widget in self.frame_visualizacion.winfo_children():
            widget.destroy()
        
        # Convertir a RGB
        if es_color:
            imagen_rgb = imagen
        else:
            if len(imagen.shape) == 2:
                imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
            else:
                imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        
        # Redimensionar si es necesario
        h, w = imagen_rgb.shape[:2]
        max_size = 800
        factor = min(max_size/w, max_size/h)
        if factor < 1:
            nuevo_w, nuevo_h = int(w*factor), int(h*factor)
            imagen_rgb = cv2.resize(imagen_rgb, (nuevo_w, nuevo_h))
        
        # Convertir a formato Tkinter
        imagen_pil = Image.fromarray(imagen_rgb)
        imagen_tk = ImageTk.PhotoImage(imagen_pil)
        
        label = ttk.Label(self.frame_visualizacion, image=imagen_tk)
        label.image = imagen_tk
        label.pack(fill=tk.BOTH, expand=True)
    
    def binarizar(self):
        """Binariza la imagen cargada"""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            umbral = self.var_umbral.get()
            self.imagen_binaria = binarizar_imagen(self.imagen_original, umbral)
            
            self.mostrar_imagen(self.imagen_binaria)
            
            self.text_resultados.delete('1.0', tk.END)
            self.text_resultados.insert('1.0', 
                f"Imagen binarizada con umbral: {umbral}\n"
                f"Dimensiones: {self.imagen_binaria.shape[1]}x{self.imagen_binaria.shape[0]}\n"
                f"Píxeles blancos: {np.sum(self.imagen_binaria == 255)}\n"
                f"Píxeles negros: {np.sum(self.imagen_binaria == 0)}\n"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al binarizar: {str(e)}")
    
    def analizar_conectividad_4(self):
        """Analiza componentes conexos con conectividad 4"""
        if self.imagen_binaria is None:
            messagebox.showwarning("Advertencia", "Primero binariza la imagen")
            return
        
        try:
            num_obj, labels, img_coloreada, stats = componentes_conexos_4(self.imagen_binaria)
            self.resultados_4 = (num_obj, labels, img_coloreada, stats)
            
            # Dibujar componentes con información
            img_con_info = dibujar_componentes_con_info(self.imagen_binaria, labels, stats)
            
            # Mostrar
            self.mostrar_imagen(img_con_info, es_color=True)
            
            # Actualizar resultados
            resumen = obtener_resumen_estadistico(stats)
            texto = f"=== CONECTIVIDAD 4 ===\n\n"
            texto += f"Objetos detectados: {num_obj}\n\n"
            texto += f"Área total: {resumen['area_total']:.0f} px²\n"
            texto += f"Área promedio: {resumen['area_promedio']:.2f} px²\n"
            texto += f"Área mínima: {resumen['area_min']:.0f} px²\n"
            texto += f"Área máxima: {resumen['area_max']:.0f} px²\n\n"
            
            texto += "Objetos individuales:\n"
            texto += "-" * 40 + "\n"
            for obj in stats:
                texto += f"Objeto #{obj['id']}: Área={obj['area']} px²\n"
            
            self.text_resultados.delete('1.0', tk.END)
            self.text_resultados.insert('1.0', texto)
            
            messagebox.showinfo("Éxito", f"Detectados {num_obj} objetos con conectividad 4")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en análisis: {str(e)}")
    
    def analizar_conectividad_8(self):
        """Analiza componentes conexos con conectividad 8"""
        if self.imagen_binaria is None:
            messagebox.showwarning("Advertencia", "Primero binariza la imagen")
            return
        
        try:
            num_obj, labels, img_coloreada, stats = componentes_conexos_8(self.imagen_binaria)
            self.resultados_8 = (num_obj, labels, img_coloreada, stats)
            
            # Dibujar componentes con información
            img_con_info = dibujar_componentes_con_info(self.imagen_binaria, labels, stats)
            
            # Mostrar
            self.mostrar_imagen(img_con_info, es_color=True)
            
            # Actualizar resultados
            resumen = obtener_resumen_estadistico(stats)
            texto = f"=== CONECTIVIDAD 8 ===\n\n"
            texto += f"Objetos detectados: {num_obj}\n\n"
            texto += f"Área total: {resumen['area_total']:.0f} px²\n"
            texto += f"Área promedio: {resumen['area_promedio']:.2f} px²\n"
            texto += f"Área mínima: {resumen['area_min']:.0f} px²\n"
            texto += f"Área máxima: {resumen['area_max']:.0f} px²\n\n"
            
            texto += "Objetos individuales:\n"
            texto += "-" * 40 + "\n"
            for obj in stats:
                texto += f"Objeto #{obj['id']}: Área={obj['area']} px²\n"
            
            self.text_resultados.delete('1.0', tk.END)
            self.text_resultados.insert('1.0', texto)
            
            messagebox.showinfo("Éxito", f"Detectados {num_obj} objetos con conectividad 8")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en análisis: {str(e)}")
    
    def comparar_conectividades(self):
        """Compara los resultados de conectividad 4 y 8"""
        if self.imagen_binaria is None:
            messagebox.showwarning("Advertencia", "Primero binariza la imagen")
            return
        
        try:
            resultados = comparar_conectividades(self.imagen_binaria)
            
            # Crear ventana de comparación
            ventana = tk.Toplevel(self.root)
            ventana.title("Comparación de Conectividades")
            ventana.geometry("1400x700")
            
            fig = Figure(figsize=(14, 7))
            
            # Imagen original
            ax1 = fig.add_subplot(2, 3, 1)
            ax1.imshow(self.imagen_binaria, cmap='gray')
            ax1.set_title('Imagen Binaria Original', fontweight='bold')
            ax1.axis('off')
            
            # Conectividad 4 - Coloreada
            ax2 = fig.add_subplot(2, 3, 2)
            ax2.imshow(resultados['conectividad_4']['imagen_coloreada'])
            ax2.set_title(f'Conectividad 4\n{resultados["conectividad_4"]["num_objetos"]} objetos', 
                         fontweight='bold', color='blue')
            ax2.axis('off')
            
            # Conectividad 8 - Coloreada
            ax3 = fig.add_subplot(2, 3, 3)
            ax3.imshow(resultados['conectividad_8']['imagen_coloreada'])
            ax3.set_title(f'Conectividad 8\n{resultados["conectividad_8"]["num_objetos"]} objetos', 
                         fontweight='bold', color='green')
            ax3.axis('off')
            
            # Conectividad 4 - Con información
            img_4_info = dibujar_componentes_con_info(
                self.imagen_binaria, 
                resultados['conectividad_4']['labels'],
                resultados['conectividad_4']['estadisticas']
            )
            ax4 = fig.add_subplot(2, 3, 5)
            ax4.imshow(cv2.cvtColor(img_4_info, cv2.COLOR_BGR2RGB))
            ax4.set_title('Conectividad 4 - Anotada', fontweight='bold')
            ax4.axis('off')
            
            # Conectividad 8 - Con información
            img_8_info = dibujar_componentes_con_info(
                self.imagen_binaria,
                resultados['conectividad_8']['labels'],
                resultados['conectividad_8']['estadisticas']
            )
            ax5 = fig.add_subplot(2, 3, 6)
            ax5.imshow(cv2.cvtColor(img_8_info, cv2.COLOR_BGR2RGB))
            ax5.set_title('Conectividad 8 - Anotada', fontweight='bold')
            ax5.axis('off')
            
            # Texto comparativo
            ax6 = fig.add_subplot(2, 3, 4)
            ax6.axis('off')
            
            texto_comp = f"COMPARACIÓN DE CONECTIVIDADES\n\n"
            texto_comp += f"Conectividad 4: {resultados['conectividad_4']['num_objetos']} objetos\n"
            texto_comp += f"Conectividad 8: {resultados['conectividad_8']['num_objetos']} objetos\n\n"
            texto_comp += f"Diferencia: {resultados['diferencia']} objetos\n\n"
            texto_comp += "Conectividad 4:\n"
            texto_comp += "- Considera solo vecinos ortogonales\n"
            texto_comp += "  (arriba, abajo, izq, der)\n\n"
            texto_comp += "Conectividad 8:\n"
            texto_comp += "- Considera vecinos ortogonales\n"
            texto_comp += "  y diagonales (8 vecinos totales)"
            
            ax6.text(0.1, 0.5, texto_comp, fontsize=11, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=ventana)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Actualizar panel de resultados
            texto = f"=== COMPARACIÓN ===\n\n"
            texto += f"Conectividad 4: {resultados['conectividad_4']['num_objetos']} objetos\n"
            texto += f"Conectividad 8: {resultados['conectividad_8']['num_objetos']} objetos\n"
            texto += f"Diferencia: {resultados['diferencia']} objetos\n"
            
            self.text_resultados.delete('1.0', tk.END)
            self.text_resultados.insert('1.0', texto)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en comparación: {str(e)}")
    
    def guardar_resultado(self):
        """Guarda la imagen resultado"""
        if self.imagen_binaria is None:
            messagebox.showwarning("Advertencia", "No hay imagen procesada para guardar")
            return
        
        ruta = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos", "*.*")]
        )
        
        if ruta:
            cv2.imwrite(ruta, self.imagen_binaria)
            messagebox.showinfo("Éxito", "Imagen guardada correctamente")
    
    def reiniciar(self):
        """Reinicia el estado de la aplicación"""
        self.imagen_original = None
        self.imagen_binaria = None
        self.resultados_4 = None
        self.resultados_8 = None
        
        for widget in self.frame_visualizacion.winfo_children():
            widget.destroy()
        
        self.text_resultados.delete('1.0', tk.END)


def main():
    root = tk.Tk()
    app = InterfazComponentesConexos(root)
    root.mainloop()


if __name__ == "__main__":
    main()
