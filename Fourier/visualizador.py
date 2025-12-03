# ===================================================================
# VISUALIZADOR DE IMÁGENES
# Clase optimizada para mostrar imágenes con matplotlib en tkinter
# ===================================================================

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Visualizador:
    """Clase para manejar la visualización de imágenes en un frame tkinter"""
    
    def __init__(self, frame_contenedor):
        self.frame = frame_contenedor
        self.figura = None
        self.canvas = None
    
    def limpiar(self):
        """Limpia el canvas actual"""
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        
        if self.figura:
            plt.close(self.figura)
            self.figura = None
        
        # Limpiar widgets restantes
        for widget in self.frame.winfo_children():
            widget.destroy()
    
    def _crear_canvas(self, figsize=(12, 5)):
        """Crea una nueva figura y canvas"""
        self.limpiar()
        self.figura = plt.figure(figsize=figsize)
        return self.figura
    
    def _mostrar_canvas(self):
        """Muestra el canvas en el frame"""
        self.figura.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figura, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def mostrar_imagen_simple(self, imagen, titulo, cmap='gray'):
        """Muestra una sola imagen"""
        fig = self._crear_canvas(figsize=(8, 6))
        
        ax = fig.add_subplot(111)
        ax.imshow(imagen, cmap=cmap)
        ax.set_title(titulo, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        self._mostrar_canvas()
    
    def mostrar_dos_imagenes(self, img1, img2, titulo1, titulo2, titulo_principal, 
                              cmap1='gray', cmap2='gray', color_titulo2='purple'):
        """Muestra dos imágenes lado a lado"""
        fig = self._crear_canvas(figsize=(12, 5))
        fig.suptitle(titulo_principal, fontsize=14, fontweight='bold')
        
        ax1 = fig.add_subplot(121)
        ax1.imshow(img1, cmap=cmap1)
        ax1.set_title(titulo1, fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(122)
        ax2.imshow(img2, cmap=cmap2)
        ax2.set_title(titulo2, fontsize=12, fontweight='bold', color=color_titulo2)
        ax2.axis('off')
        
        self._mostrar_canvas()
    
    def mostrar_tres_imagenes(self, img1, img2, img3, titulo1, titulo2, titulo3, 
                               titulo_principal, cmap1='gray', cmap2='gray', cmap3='hsv'):
        """Muestra tres imágenes lado a lado"""
        fig = self._crear_canvas(figsize=(14, 5))
        fig.suptitle(titulo_principal, fontsize=14, fontweight='bold')
        
        ax1 = fig.add_subplot(131)
        ax1.imshow(img1, cmap=cmap1)
        ax1.set_title(titulo1, fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(132)
        ax2.imshow(img2, cmap=cmap2)
        ax2.set_title(titulo2, fontsize=11, fontweight='bold', color='purple')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(133)
        ax3.imshow(img3, cmap=cmap3)
        ax3.set_title(titulo3, fontsize=11, fontweight='bold', color='blue')
        ax3.axis('off')
        
        self._mostrar_canvas()
    
    def mostrar_cuadricula(self, imagenes, titulos, titulo_principal, 
                           filas=2, columnas=2, cmaps=None, con_colorbar=None):
        """
        Muestra múltiples imágenes en una cuadrícula
        
        Args:
            imagenes: Lista de imágenes
            titulos: Lista de títulos para cada imagen
            titulo_principal: Título general
            filas: Número de filas
            columnas: Número de columnas
            cmaps: Lista de colormaps (opcional)
            con_colorbar: Lista de índices que tendrán colorbar (opcional)
        """
        fig = self._crear_canvas(figsize=(5*columnas, 4.5*filas))
        fig.suptitle(titulo_principal, fontsize=14, fontweight='bold')
        
        if cmaps is None:
            cmaps = ['gray'] * len(imagenes)
        
        if con_colorbar is None:
            con_colorbar = []
        
        for idx, (img, titulo, cmap) in enumerate(zip(imagenes, titulos, cmaps)):
            ax = fig.add_subplot(filas, columnas, idx + 1)
            im = ax.imshow(img, cmap=cmap)
            ax.set_title(titulo, fontsize=11, fontweight='bold')
            ax.axis('off')
            
            if idx in con_colorbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        self._mostrar_canvas()
    
    def cerrar(self):
        """Cierra todas las figuras de matplotlib"""
        self.limpiar()
        plt.close('all')
    
    def mostrar_analisis_completo(self, imagen_original, imagen_filtrada, mascara, 
                                   nombre_filtro, texto_analisis):
        """
        Muestra un análisis completo del filtrado con 6 subplots:
        - Imagen original
        - Imagen filtrada
        - Máscara del filtro
        - Diferencia absoluta
        - Histogramas comparativos
        - Texto del análisis
        """
        fig = self._crear_canvas(figsize=(16, 10))
        fig.suptitle(f'Análisis Completo - {nombre_filtro}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Imagen Original
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(imagen_original, cmap='gray')
        ax1.set_title('Imagen Original', fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        # 2. Imagen Filtrada
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.imshow(imagen_filtrada, cmap='gray')
        ax2.set_title('Imagen Filtrada', fontsize=11, fontweight='bold', color='blue')
        ax2.axis('off')
        
        # 3. Máscara del Filtro
        ax3 = fig.add_subplot(2, 3, 3)
        mascara_norm = (mascara - mascara.min()) / (mascara.max() - mascara.min() + 1e-10)
        im3 = ax3.imshow(mascara_norm, cmap='viridis')
        ax3.set_title('Máscara del Filtro', fontsize=11, fontweight='bold', color='green')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # 4. Diferencia Absoluta
        ax4 = fig.add_subplot(2, 3, 4)
        diferencia = abs(imagen_original.astype(float) - imagen_filtrada.astype(float))
        im4 = ax4.imshow(diferencia, cmap='hot')
        ax4.set_title('Diferencia Absoluta', fontsize=11, fontweight='bold', color='red')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        
        # 5. Histogramas Comparativos
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.hist(imagen_original.ravel(), bins=64, alpha=0.6, label='Original', 
                color='blue', edgecolor='black')
        ax5.hist(imagen_filtrada.ravel(), bins=64, alpha=0.6, label='Filtrada', 
                color='red', edgecolor='black')
        ax5.set_title('Histogramas', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Intensidad', fontsize=9)
        ax5.set_ylabel('Frecuencia', fontsize=9)
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        # 6. Texto de Análisis
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        ax6.text(0.05, 0.95, texto_analisis, 
                transform=ax6.transAxes,
                fontsize=7,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        self._mostrar_canvas()

