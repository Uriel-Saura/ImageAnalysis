"""
Script de ejemplo para probar las funcionalidades de la Práctica 2
sin necesidad de usar la interfaz gráfica.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from generacion_ruido import aplicar_ruido_sal_pimienta, aplicar_ruido_gaussiano
from filtros_lineales import filtro_sobel, filtro_gaussiano, filtro_canny
from filtros_no_lineales import filtro_mediana


def ejemplo_basico():
    """
    Ejemplo básico de uso de las funcionalidades.
    """
    print("=" * 60)
    print("EJEMPLO DE USO - PRÁCTICA 2")
    print("=" * 60)
    
    # Crear una imagen de prueba
    print("\n1. Creando imagen de prueba...")
    imagen = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Dibujar algunas formas
    cv2.rectangle(imagen, (50, 50), (150, 150), (0, 0, 255), -1)
    cv2.circle(imagen, (250, 100), 50, (0, 255, 0), -1)
    cv2.line(imagen, (50, 250), (350, 250), (255, 0, 0), 3)
    
    print("   ✓ Imagen de prueba creada")
    
    # Aplicar ruido sal y pimienta
    print("\n2. Aplicando ruido sal y pimienta...")
    imagen_sp = aplicar_ruido_sal_pimienta(imagen, probabilidad=0.05)
    print("   ✓ Ruido sal y pimienta aplicado")
    
    # Aplicar filtro de mediana para remover el ruido
    print("\n3. Aplicando filtro de mediana para limpiar ruido...")
    imagen_limpia = filtro_mediana(imagen_sp, tamano_kernel=5)
    print("   ✓ Filtro de mediana aplicado")
    
    # Aplicar ruido gaussiano
    print("\n4. Aplicando ruido gaussiano...")
    imagen_gauss = aplicar_ruido_gaussiano(imagen, media=0, sigma=25)
    print("   ✓ Ruido gaussiano aplicado")
    
    # Aplicar filtro gaussiano para suavizar
    print("\n5. Aplicando filtro gaussiano para suavizar...")
    imagen_suave = filtro_gaussiano(imagen_gauss, tamano_kernel=5, sigma=1.0)
    print("   ✓ Filtro gaussiano aplicado")
    
    # Detectar bordes con Sobel
    print("\n6. Detectando bordes con Sobel...")
    bordes_sobel = filtro_sobel(imagen)
    print("   ✓ Bordes detectados con Sobel")
    
    # Detectar bordes con Canny
    print("\n7. Detectando bordes con Canny...")
    bordes_canny = filtro_canny(imagen, umbral1=100, umbral2=200)
    print("   ✓ Bordes detectados con Canny")
    
    # Visualizar resultados
    print("\n8. Generando visualización...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Convertir a RGB para visualización
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen_sp_rgb = cv2.cvtColor(imagen_sp, cv2.COLOR_BGR2RGB)
    imagen_limpia_rgb = cv2.cvtColor(imagen_limpia, cv2.COLOR_BGR2RGB)
    imagen_gauss_rgb = cv2.cvtColor(imagen_gauss, cv2.COLOR_BGR2RGB)
    imagen_suave_rgb = cv2.cvtColor(imagen_suave, cv2.COLOR_BGR2RGB)
    
    # Primera fila: Original y ruido sal y pimienta
    axes[0, 0].imshow(imagen_rgb)
    axes[0, 0].set_title('Imagen Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(imagen_sp_rgb)
    axes[0, 1].set_title('Ruido Sal y Pimienta')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(imagen_limpia_rgb)
    axes[0, 2].set_title('Filtro Mediana')
    axes[0, 2].axis('off')
    
    # Segunda fila: Ruido gaussiano
    axes[1, 0].imshow(imagen_rgb)
    axes[1, 0].set_title('Imagen Original')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(imagen_gauss_rgb)
    axes[1, 1].set_title('Ruido Gaussiano')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(imagen_suave_rgb)
    axes[1, 2].set_title('Filtro Gaussiano')
    axes[1, 2].axis('off')
    
    # Tercera fila: Detección de bordes
    axes[2, 0].imshow(imagen_rgb)
    axes[2, 0].set_title('Imagen Original')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(bordes_sobel, cmap='gray')
    axes[2, 1].set_title('Bordes Sobel')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(bordes_canny, cmap='gray')
    axes[2, 2].set_title('Bordes Canny')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    print("   ✓ Visualización generada")
    
    print("\n" + "=" * 60)
    print("EJEMPLO COMPLETADO")
    print("Mostrando visualización...")
    print("=" * 60)
    
    plt.show()


def ejemplo_con_imagen_real():
    """
    Ejemplo usando una imagen real del sistema.
    """
    print("\n" + "=" * 60)
    print("EJEMPLO CON IMAGEN REAL")
    print("=" * 60)
    print("\nPara usar este ejemplo:")
    print("1. Coloca una imagen en la carpeta 'img' del proyecto")
    print("2. Modifica la ruta en este script")
    print("3. Ejecuta el ejemplo")
    print("=" * 60)


if __name__ == "__main__":
    # Ejecutar ejemplo básico con imagen sintética
    ejemplo_basico()
    
    # Mostrar información sobre ejemplo con imagen real
    ejemplo_con_imagen_real()
