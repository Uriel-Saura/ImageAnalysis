"""
Programa para visualizar todas las operaciones morfológicas
aplicadas a todas las imágenes de la carpeta img/
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def crear_kernel(size=5):
    """Crea un kernel rectangular"""
    if size % 2 == 0:
        size += 1
    return np.ones((size, size), np.uint8)


def aplicar_todas_operaciones(imagen, kernel_size=5):
    """Aplica todas las operaciones morfológicas a una imagen"""
    kernel = crear_kernel(kernel_size)
    kernel_pequeno = np.ones((3, 3), np.uint8)
    
    resultados = {}
    
    # Operaciones básicas
    resultados['Original'] = imagen
    resultados['Erosion'] = cv2.erode(imagen, kernel)
    resultados['Dilatacion'] = cv2.dilate(imagen, kernel)
    resultados['Apertura'] = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
    resultados['Cierre'] = cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)
    
    # Frontera
    erosion = cv2.erode(imagen, kernel)
    resultados['Frontera'] = cv2.subtract(imagen, erosion)
    
    # Adelgazamiento
    resultado = imagen.copy()
    erosion = cv2.erode(resultado, kernel_pequeno)
    temp = cv2.dilate(erosion, kernel_pequeno)
    temp = cv2.subtract(resultado, temp)
    resultados['Adelgazamiento'] = erosion
    
    # Hit-or-Miss
    hit = cv2.erode(imagen, kernel_pequeno)
    miss = cv2.erode(cv2.bitwise_not(imagen), kernel_pequeno)
    resultados['Hit-or-Miss'] = cv2.bitwise_and(hit, miss)
    
    # Esqueleto
    skeleton = np.zeros(imagen.shape, np.uint8)
    temp = imagen.copy()
    max_iteraciones = 100  # Límite de iteraciones para evitar bucles infinitos
    iteracion = 0
    
    while iteracion < max_iteraciones:
        eroded = cv2.erode(temp, kernel_pequeno)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel_pequeno)
        subset = cv2.subtract(eroded, opened)
        skeleton = cv2.bitwise_or(skeleton, subset)
        temp = eroded.copy()
        
        if cv2.countNonZero(temp) == 0:
            break
        iteracion += 1
    
    resultados['Esqueleto'] = skeleton
    
    # Gradientes
    dilatacion = cv2.dilate(imagen, kernel)
    erosion = cv2.erode(imagen, kernel)
    resultados['Grad. Simetrico'] = cv2.subtract(dilatacion, erosion)
    resultados['Grad. Externo'] = cv2.subtract(dilatacion, imagen)
    resultados['Grad. Interno'] = cv2.subtract(imagen, erosion)
    
    # Top Hat y Black Hat
    resultados['Top Hat'] = cv2.morphologyEx(imagen, cv2.MORPH_TOPHAT, kernel)
    resultados['Black Hat'] = cv2.morphologyEx(imagen, cv2.MORPH_BLACKHAT, kernel)
    
    # Filtro combinado
    apertura = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
    resultados['Filtro Combinado'] = cv2.morphologyEx(apertura, cv2.MORPH_CLOSE, kernel)
    
    return resultados


def visualizar_imagen_completa(ruta_imagen, kernel_size=5):
    """Visualiza una imagen con todas sus operaciones morfológicas"""
    try:
        # Cargar imagen
        imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        
        if imagen is None:
            print(f"Error al cargar: {ruta_imagen}")
            return None
        
        # Detectar tipo de imagen
        valores_unicos = np.unique(imagen)
        tipo = "BINARIA" if len(valores_unicos) <= 2 else "GRISES"
        
        # Aplicar operaciones
        resultados = aplicar_todas_operaciones(imagen, kernel_size)
        
        # Crear figura con subplots
        num_ops = len(resultados)
        cols = 4
        rows = (num_ops + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5))
        fig.suptitle(f'{os.path.basename(ruta_imagen)} - [{tipo}] - Kernel: {kernel_size}x{kernel_size}', 
                     fontsize=16, fontweight='bold')
        
        # Aplanar el array de axes para iterar fácilmente
        axes_flat = axes.flatten() if rows > 1 else axes
        
        # Mostrar cada operación
        for idx, (nombre, resultado) in enumerate(resultados.items()):
            ax = axes_flat[idx]
            ax.imshow(resultado, cmap='gray', vmin=0, vmax=255)
            
            # Color según el tipo de operación
            if nombre == 'Original':
                color = 'black'
                weight = 'bold'
            elif 'Grad' in nombre:
                color = 'red'
                weight = 'normal'
            elif nombre in ['Top Hat', 'Black Hat']:
                color = 'purple'
                weight = 'normal'
            else:
                color = 'blue'
                weight = 'normal'
            
            ax.set_title(nombre, fontsize=10, fontweight=weight, color=color)
            ax.axis('off')
        
        # Ocultar axes sobrantes
        for idx in range(num_ops, len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"ERROR al procesar {ruta_imagen}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def procesar_todas_imagenes(carpeta='img', kernel_size=5, guardar=False):
    """Procesa todas las imágenes de una carpeta"""
    # Buscar todas las imágenes
    extensiones = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    imagenes = []
    
    if not os.path.exists(carpeta):
        print(f"Error: La carpeta '{carpeta}' no existe")
        return
    
    for ext in extensiones:
        imagenes.extend(Path(carpeta).glob(ext))
    
    if not imagenes:
        print(f"No se encontraron imágenes en '{carpeta}'")
        return
    
    print(f"\n{'='*70}")
    print(f"PROCESANDO {len(imagenes)} IMAGENES DE LA CARPETA '{carpeta}'")
    print(f"Tamaño del kernel: {kernel_size}x{kernel_size}")
    print(f"{'='*70}\n")
    
    # Procesar cada imagen
    for idx, ruta_imagen in enumerate(sorted(imagenes), 1):
        print(f"\n[{idx}/{len(imagenes)}] Procesando: {ruta_imagen.name}")
        
        try:
            fig = visualizar_imagen_completa(str(ruta_imagen), kernel_size)
            
            if fig is None:
                print(f"    -> ERROR: No se pudo procesar la imagen, continuando...")
                continue
            
            if guardar:
                # Guardar la figura
                output_dir = 'resultados'
                os.makedirs(output_dir, exist_ok=True)
                nombre_salida = f"{output_dir}/{ruta_imagen.stem}_todas_operaciones.png"
                fig.savefig(nombre_salida, dpi=150, bbox_inches='tight')
                print(f"    -> Guardado en: {nombre_salida}")
            
            # Mostrar la figura y esperar a que se cierre
            print(f"    -> Mostrando figura... (cierre la ventana para continuar)")
            plt.show()  # Esto bloquea hasta que se cierre la ventana
            plt.close(fig)  # Asegurar que se cierra
            
        except KeyboardInterrupt:
            print("\n\n*** Programa interrumpido por el usuario ***")
            plt.close('all')
            break
        except Exception as e:
            print(f"    -> ERROR inesperado: {str(e)}")
            import traceback
            traceback.print_exc()
            plt.close('all')
            continue
    
    print(f"\n{'='*70}")
    print("PROCESAMIENTO COMPLETADO")
    print(f"{'='*70}\n")


def main():
    """Función principal"""
    import sys
    
    # Configuración
    carpeta = 'img'
    kernel_size = 5
    guardar = False
    
    # Procesar argumentos de línea de comandos (opcional)
    if len(sys.argv) > 1:
        carpeta = sys.argv[1]
    if len(sys.argv) > 2:
        kernel_size = int(sys.argv[2])
    if len(sys.argv) > 3:
        guardar = sys.argv[3].lower() in ['true', '1', 'si', 'yes']
    
    print("\n" + "="*70)
    print("VISUALIZADOR DE OPERACIONES MORFOLOGICAS")
    print("="*70)
    print(f"\nCarpeta: {carpeta}")
    print(f"Tamaño del kernel: {kernel_size}x{kernel_size}")
    print(f"Guardar resultados: {'Si' if guardar else 'No'}")
    
    # Procesar todas las imágenes
    procesar_todas_imagenes(carpeta, kernel_size, guardar)


if __name__ == "__main__":
    main()
