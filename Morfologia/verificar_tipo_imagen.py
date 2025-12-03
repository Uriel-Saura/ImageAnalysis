import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def verificar_tipo_imagen(imagen_path):
    """
    Verifica si una imagen es binaria o en escala de grises
    
    Retorna:
    - 'binaria': Si la imagen tiene solo 2 valores únicos
    - 'grises': Si la imagen tiene más de 2 valores
    - None: Si hay error al cargar la imagen
    """
    # Leer la imagen en escala de grises
    img = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"❌ Error: No se pudo cargar la imagen {imagen_path}")
        return None
    
    # Obtener valores únicos en la imagen
    valores_unicos = np.unique(img)
    num_valores = len(valores_unicos)
    
    # Determinar el tipo
    if num_valores <= 2:
        tipo = 'binaria'
    else:
        tipo = 'grises'
    
    return tipo, valores_unicos, img


def analizar_imagen(imagen_path, mostrar_grafico=True):
    """
    Analiza y muestra información detallada sobre el tipo de imagen
    con visualización gráfica
    """
    print(f"\n{'='*70}")
    print(f"Analizando: {os.path.basename(imagen_path)}")
    print(f"{'='*70}")
    
    resultado = verificar_tipo_imagen(imagen_path)
    
    if resultado is None:
        return
    
    tipo, valores_unicos, img = resultado
    
    # Información básica
    print(f"📁 Ruta: {imagen_path}")
    print(f"📐 Tamaño: {img.shape[0]} x {img.shape[1]} píxeles")
    print(f"🎨 Tipo: {'BINARIA' if tipo == 'binaria' else 'ESCALA DE GRISES'}")
    
    # Información sobre valores
    print(f"\n📊 Estadísticas:")
    print(f"   • Valores únicos: {len(valores_unicos)}")
    print(f"   • Valor mínimo: {img.min()}")
    print(f"   • Valor máximo: {img.max()}")
    print(f"   • Valor medio: {img.mean():.2f}")
    
    # Detalles específicos según el tipo
    if tipo == 'binaria':
        print(f"\n✅ IMAGEN BINARIA DETECTADA")
        print(f"   • Valores presentes: {valores_unicos}")
        
        # Contar píxeles de cada valor
        for valor in valores_unicos:
            cantidad = np.sum(img == valor)
            porcentaje = (cantidad / img.size) * 100
            print(f"   • Píxeles con valor {valor}: {cantidad} ({porcentaje:.2f}%)")
    else:
        print(f"\n✅ IMAGEN EN ESCALA DE GRISES DETECTADA")
        print(f"   • Rango de valores: [{valores_unicos[0]} - {valores_unicos[-1]}]")
        print(f"   • Primeros 10 valores: {valores_unicos[:10]}")
        if len(valores_unicos) > 10:
            print(f"   • ... ({len(valores_unicos) - 10} valores más)")
        
        # Distribución de intensidades
        print(f"\n   Distribución de intensidades:")
        print(f"   • Píxeles oscuros (0-85): {np.sum(img < 85)} ({(np.sum(img < 85)/img.size)*100:.1f}%)")
        print(f"   • Píxeles medios (85-170): {np.sum((img >= 85) & (img < 170))} ({(np.sum((img >= 85) & (img < 170))/img.size)*100:.1f}%)")
        print(f"   • Píxeles claros (170-255): {np.sum(img >= 170)} ({(np.sum(img >= 170)/img.size)*100:.1f}%)")
    
    # Visualización gráfica
    if mostrar_grafico:
        visualizar_imagen_con_histograma(img, tipo, imagen_path)
    
    return tipo


def visualizar_imagen_con_histograma(img, tipo, imagen_path):
    """
    Crea una visualización con la imagen, su histograma y estadísticas
    """
    fig = plt.figure(figsize=(14, 6))
    
    # Nombre del archivo
    nombre_archivo = os.path.basename(imagen_path)
    
    # Layout de la figura
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Imagen original
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax1.set_title(f'{nombre_archivo}\n{"BINARIA" if tipo == "binaria" else "ESCALA DE GRISES"}', 
                  fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Histograma
    ax2 = fig.add_subplot(gs[0, 1:])
    valores_unicos = np.unique(img)
    
    if tipo == 'binaria':
        # Histograma para imagen binaria (solo 2 barras)
        conteos = [np.sum(img == val) for val in valores_unicos]
        colores = ['black' if val == 0 else 'white' for val in valores_unicos]
        bars = ax2.bar(valores_unicos, conteos, width=20, color=colores, 
                      edgecolor='blue', linewidth=2)
        ax2.set_xticks(valores_unicos)
        ax2.set_title('Histograma (Imagen Binaria)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Cantidad de píxeles')
        ax2.set_xlabel('Valor de intensidad')
        ax2.grid(True, alpha=0.3)
        
        # Añadir etiquetas de cantidad
        for bar, val, count in zip(bars, valores_unicos, conteos):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({count/img.size*100:.1f}%)',
                    ha='center', va='bottom', fontsize=9)
    else:
        # Histograma para imagen en escala de grises
        ax2.hist(img.ravel(), bins=256, range=(0, 256), color='steelblue', 
                edgecolor='black', alpha=0.7)
        ax2.set_title('Histograma (Escala de Grises)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Cantidad de píxeles')
        ax2.set_xlabel('Valor de intensidad')
        ax2.set_xlim(0, 255)
        ax2.grid(True, alpha=0.3)
    
    # 3. Tabla de estadísticas
    ax3 = fig.add_subplot(gs[1, 1:])
    ax3.axis('off')
    
    # Información estadística
    stats_data = [
        ['Tamaño', f'{img.shape[0]} × {img.shape[1]} px'],
        ['Total píxeles', f'{img.size:,}'],
        ['Valores únicos', f'{len(valores_unicos)}'],
        ['Mínimo', f'{img.min()}'],
        ['Máximo', f'{img.max()}'],
        ['Media', f'{img.mean():.2f}'],
        ['Desv. estándar', f'{img.std():.2f}'],
        ['Mediana', f'{np.median(img):.0f}']
    ]
    
    # Crear tabla
    table = ax3.table(cellText=stats_data, 
                     colLabels=['Estadística', 'Valor'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Estilo de la tabla
    for i in range(len(stats_data) + 1):
        if i == 0:
            table[(i, 0)].set_facecolor('#4CAF50')
            table[(i, 1)].set_facecolor('#4CAF50')
            table[(i, 0)].set_text_props(weight='bold', color='white')
            table[(i, 1)].set_text_props(weight='bold', color='white')
        else:
            if i % 2 == 0:
                table[(i, 0)].set_facecolor('#f0f0f0')
                table[(i, 1)].set_facecolor('#f0f0f0')
    
    ax3.set_title('Estadísticas Detalladas', fontsize=11, fontweight='bold', pad=20)
    
    plt.suptitle(f'Análisis de Imagen: {nombre_archivo}', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.show()



def verificar_directorio(directorio):
    """
    Verifica todas las imágenes en un directorio y muestra un resumen visual
    """
    print("\n" + "="*70)
    print("VERIFICACIÓN DE IMÁGENES EN DIRECTORIO")
    print("="*70)
    
    # Extensiones de imagen soportadas
    extensiones = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    # Buscar imágenes
    imagenes = []
    if os.path.exists(directorio):
        for archivo in os.listdir(directorio):
            if any(archivo.lower().endswith(ext) for ext in extensiones):
                imagenes.append(os.path.join(directorio, archivo))
    
    if not imagenes:
        print(f"❌ No se encontraron imágenes en {directorio}")
        return
    
    print(f"✅ Se encontraron {len(imagenes)} imagen(es)\n")
    
    # Contadores
    binarias = []
    grises = []
    tipos_dict = {}
    imagenes_data = []
    
    # Analizar cada imagen (sin mostrar gráficos individuales)
    for img_path in sorted(imagenes):
        tipo = analizar_imagen(img_path, mostrar_grafico=False)
        
        if tipo == 'binaria':
            binarias.append(os.path.basename(img_path))
        elif tipo == 'grises':
            grises.append(os.path.basename(img_path))
        
        tipos_dict[img_path] = tipo
        
        # Cargar imagen para el resumen visual
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            imagenes_data.append((img, os.path.basename(img_path), tipo))
    
    # Resumen en terminal
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)
    print(f"\n📊 Total de imágenes analizadas: {len(imagenes)}")
    print(f"\n🔲 Imágenes BINARIAS: {len(binarias)}")
    for img in binarias:
        print(f"   • {img}")
    
    print(f"\n🎨 Imágenes en ESCALA DE GRISES: {len(grises)}")
    for img in grises:
        print(f"   • {img}")
    
    # Crear visualización de resumen
    if imagenes_data:
        crear_resumen_visual(imagenes_data)
    
    return tipos_dict


def crear_resumen_visual(imagenes_data):
    """
    Crea un panel visual con todas las imágenes y sus histogramas
    """
    n_imagenes = len(imagenes_data)
    
    # Determinar el layout de la cuadrícula
    if n_imagenes <= 4:
        filas = 2
        cols = 2
    elif n_imagenes <= 6:
        filas = 2
        cols = 3
    elif n_imagenes <= 8:
        filas = 2
        cols = 4
    else:
        filas = 3
        cols = int(np.ceil(n_imagenes / 3))
    
    fig, axes = plt.subplots(filas, cols, figsize=(cols*4, filas*4.5))
    
    # Asegurarse de que axes sea un array 2D
    if n_imagenes == 1:
        axes = np.array([[axes]])
    elif filas == 1 or cols == 1:
        axes = axes.reshape(filas, cols)
    
    axes_flat = axes.flatten()
    
    for idx, (img, nombre, tipo) in enumerate(imagenes_data):
        if idx >= len(axes_flat):
            break
        
        ax = axes_flat[idx]
        
        # Dividir el subplot en dos partes (imagen arriba, histograma abajo)
        # Crear subgráficos manualmente
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=ax.get_subplotspec(), 
                                      hspace=0.3, height_ratios=[2, 1])
        
        ax.remove()
        ax_img = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
        
        # Mostrar imagen
        ax_img.imshow(img, cmap='gray', vmin=0, vmax=255)
        color_tipo = 'red' if tipo == 'binaria' else 'green'
        ax_img.set_title(f'{nombre}\n[{tipo.upper()}]', 
                        fontsize=9, fontweight='bold', color=color_tipo)
        ax_img.axis('off')
        
        # Mostrar histograma
        valores_unicos = np.unique(img)
        
        if tipo == 'binaria':
            conteos = [np.sum(img == val) for val in valores_unicos]
            colores = ['black' if val == 0 else 'lightgray' for val in valores_unicos]
            ax_hist.bar(valores_unicos, conteos, width=30, color=colores, 
                       edgecolor='red', linewidth=1.5)
            ax_hist.set_xticks(valores_unicos)
            ax_hist.set_xlim(-10, 265)
        else:
            ax_hist.hist(img.ravel(), bins=64, range=(0, 256), 
                        color='steelblue', edgecolor='black', alpha=0.7)
            ax_hist.set_xlim(0, 255)
        
        ax_hist.set_xlabel('Intensidad', fontsize=8)
        ax_hist.set_ylabel('Píxeles', fontsize=8)
        ax_hist.tick_params(labelsize=7)
        ax_hist.grid(True, alpha=0.3)
        
        # Añadir texto con estadísticas
        texto_stats = f'Min: {img.min()}, Max: {img.max()}\nMedia: {img.mean():.1f}'
        ax_hist.text(0.98, 0.98, texto_stats, 
                    transform=ax_hist.transAxes,
                    fontsize=7, verticalalignment='top', 
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Ocultar ejes sobrantes
    for idx in range(n_imagenes, len(axes_flat)):
        axes_flat[idx].remove()
    
    plt.suptitle('RESUMEN VISUAL - Todas las Imágenes', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()



def verificar_imagen_simple(imagen_path):
    """
    Versión simple que solo retorna el tipo
    """
    resultado = verificar_tipo_imagen(imagen_path)
    
    if resultado is None:
        return None
    
    tipo, _, _ = resultado
    return tipo


def main():
    """
    Función principal
    """
    print("="*70)
    print("VERIFICADOR DE TIPO DE IMAGEN")
    print("Detecta si una imagen es BINARIA o ESCALA DE GRISES")
    print("="*70)
    
    # Opción 1: Verificar directorio completo
    if os.path.exists('img'):
        verificar_directorio('img')
    else:
        print("\n❌ No se encontró el directorio 'img'")
    
    # Opción 2: Ejemplo de verificación individual
    print("\n" + "="*70)
    print("EJEMPLO DE USO PROGRAMÁTICO")
    print("="*70)
    print("\nEjemplo de código para usar en otros programas:")
    print("""
    from verificar_tipo_imagen import verificar_imagen_simple
    
    tipo = verificar_imagen_simple('img/mi_imagen.png')
    
    if tipo == 'binaria':
        print("Es una imagen binaria")
        # Aplicar procesamiento para imágenes binarias
    elif tipo == 'grises':
        print("Es una imagen en escala de grises")
        # Aplicar procesamiento para imágenes en grises
    """)


if __name__ == "__main__":
    main()
