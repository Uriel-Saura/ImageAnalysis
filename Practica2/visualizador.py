"""
Módulo para visualización de imágenes y histogramas.
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


def mostrar_imagen_en_canvas(canvas, imagen, titulo=""):
    """
    Muestra una imagen en un canvas de matplotlib.
    
    Args:
        canvas: Canvas de matplotlib
        imagen: Imagen a mostrar
        titulo: Título de la imagen
    """
    fig = canvas.figure
    fig.clear()
    ax = fig.add_subplot(111)
    
    if len(imagen.shape) == 2:
        # Imagen en escala de grises
        ax.imshow(imagen, cmap='gray')
    else:
        # Imagen en color (BGR a RGB)
        import cv2
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        ax.imshow(imagen_rgb)
    
    ax.set_title(titulo)
    ax.axis('off')
    canvas.draw()


def crear_canvas_matplotlib(parent, figsize=(5, 4)):
    """
    Crea un canvas de matplotlib para tkinter.
    
    Args:
        parent: Widget padre de tkinter
        figsize: Tamaño de la figura
        
    Returns:
        Canvas de matplotlib
    """
    fig = plt.Figure(figsize=figsize, dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.get_tk_widget().pack(side='left', fill='both', expand=True)
    return canvas


def mostrar_histograma_en_canvas(canvas, imagen, titulo="Histograma"):
    """
    Muestra el histograma de una imagen en un canvas.
    
    Args:
        canvas: Canvas de matplotlib
        imagen: Imagen para calcular histograma
        titulo: Título del histograma
    """
    import cv2
    
    fig = canvas.figure
    fig.clear()
    ax = fig.add_subplot(111)
    
    if len(imagen.shape) == 2:
        # Imagen en escala de grises
        hist = cv2.calcHist([imagen], [0], None, [256], [0, 256])
        ax.plot(hist, color='black')
        ax.set_xlim([0, 256])
    else:
        # Imagen en color - histograma por canal
        colores = ('b', 'g', 'r')
        nombres = ('Azul', 'Verde', 'Rojo')
        for i, color in enumerate(colores):
            hist = cv2.calcHist([imagen], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, label=nombres[i])
        ax.legend()
        ax.set_xlim([0, 256])
    
    ax.set_title(titulo)
    ax.set_xlabel('Intensidad de píxel')
    ax.set_ylabel('Frecuencia')
    ax.grid(True, alpha=0.3)
    canvas.draw()


def mostrar_comparacion(imagenes, titulos, filas=2, columnas=2):
    """
    Muestra múltiples imágenes en una cuadrícula.
    
    Args:
        imagenes: Lista de imágenes
        titulos: Lista de títulos
        filas: Número de filas
        columnas: Número de columnas
    """
    import cv2
    
    fig, axes = plt.subplots(filas, columnas, figsize=(12, 10))
    axes = axes.ravel() if isinstance(axes, np.ndarray) else [axes]
    
    for idx, (img, titulo) in enumerate(zip(imagenes, titulos)):
        if idx < len(axes):
            if len(img.shape) == 2:
                axes[idx].imshow(img, cmap='gray')
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[idx].imshow(img_rgb)
            axes[idx].set_title(titulo)
            axes[idx].axis('off')
    
    # Ocultar axes no utilizados
    for idx in range(len(imagenes), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
