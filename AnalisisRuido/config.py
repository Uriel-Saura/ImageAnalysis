"""
Archivo de configuración para la Práctica 2.
Modifica estos valores para ajustar el comportamiento predeterminado de la aplicación.
"""

# ==================== CONFIGURACIÓN DE RUIDO ====================

# Ruido Sal y Pimienta
PROB_SAL_PIMIENTA_DEFAULT = 0.05  # Probabilidad por defecto (0.0 - 1.0)
PROB_SAL_PIMIENTA_MIN = 0.01      # Valor mínimo en la interfaz
PROB_SAL_PIMIENTA_MAX = 0.2       # Valor máximo en la interfaz

# Ruido Gaussiano
SIGMA_GAUSSIANO_DEFAULT = 25      # Sigma por defecto
SIGMA_GAUSSIANO_MIN = 5           # Valor mínimo en la interfaz
SIGMA_GAUSSIANO_MAX = 100         # Valor máximo en la interfaz
MEDIA_GAUSSIANO_DEFAULT = 0       # Media por defecto

# ==================== CONFIGURACIÓN DE FILTROS ====================

# Filtros Lineales
TAMANO_KERNEL_DEFAULT = 5         # Tamaño de kernel por defecto (debe ser impar)
TAMANO_KERNEL_MIN = 3             # Tamaño mínimo
TAMANO_KERNEL_MAX = 15            # Tamaño máximo

# Filtro Gaussiano
SIGMA_FILTRO_GAUSSIANO_DEFAULT = 1.0

# Filtro Bilateral
BILATERAL_D_DEFAULT = 9           # Diámetro del vecindario
BILATERAL_SIGMA_COLOR_DEFAULT = 75
BILATERAL_SIGMA_SPACE_DEFAULT = 75

# Detector Canny
CANNY_UMBRAL1_DEFAULT = 100       # Umbral bajo
CANNY_UMBRAL2_DEFAULT = 200       # Umbral alto

# Filtro Contraharmonic Mean
CONTRAHARMONIC_Q_DEFAULT = 1.5    # Parámetro Q (positivo elimina pimienta, negativo elimina sal)

# ==================== CONFIGURACIÓN DE INTERFAZ ====================

# Tamaño de la ventana principal
VENTANA_ANCHO = 1400
VENTANA_ALTO = 900

# Tamaño de los canvas de matplotlib
CANVAS_FIGURA_ANCHO = 5
CANVAS_FIGURA_ALTO = 4

# ==================== CONFIGURACIÓN DE VISUALIZACIÓN ====================

# Formato de guardado por defecto
FORMATO_GUARDADO_DEFAULT = ".png"

# DPI para figuras de matplotlib
DPI_MATPLOTLIB = 100

# Mostrar ejes en imágenes
MOSTRAR_EJES = False

# Colormap para imágenes en escala de grises
COLORMAP_GRISES = 'gray'

# ==================== CONFIGURACIÓN DE PROCESAMIENTO ====================

# Tipo de dato para cálculos intermedios
DTYPE_INTERMEDIO = 'float32'

# Método de padding para bordes
# Opciones: 'reflect', 'constant', 'replicate', 'wrap'
PADDING_METHOD = 'reflect'

# ==================== RUTAS ====================

# Carpeta de imágenes por defecto (relativa al proyecto principal)
CARPETA_IMAGENES_DEFAULT = "../img"

# Extensiones de archivo soportadas
EXTENSIONES_IMAGEN = [
    ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
    ("PNG", "*.png"),
    ("JPEG", "*.jpg *.jpeg"),
    ("BMP", "*.bmp"),
    ("TIFF", "*.tiff *.tif"),
    ("Todos los archivos", "*.*")
]

# ==================== MENSAJES ====================

# Mensajes de la interfaz
MSG_SIN_IMAGEN = "No se ha cargado ninguna imagen"
MSG_CARGAR_PRIMERO = "Primero carga una imagen"
MSG_ERROR_CARGA = "No se pudo cargar la imagen"
MSG_EXITO_CARGA = "Imagen cargada correctamente"
MSG_EXITO_GUARDADO = "Imagen guardada correctamente"
MSG_SIN_PROCESADA = "No hay imagen procesada para guardar"
MSG_RESTABLECER = "Imagen restablecida a la original"

# ==================== VALIDACIONES ====================

# Tamaño máximo de imagen permitido (en píxeles)
MAX_ANCHO_IMAGEN = 5000
MAX_ALTO_IMAGEN = 5000

# Advertencia si la imagen es muy grande
ADVERTIR_IMAGEN_GRANDE = True
UMBRAL_IMAGEN_GRANDE = 2000  # píxeles

# ==================== MODO DEBUG ====================

# Activar modo debug (muestra información adicional en consola)
DEBUG_MODE = False

# Mostrar tiempos de ejecución
MOSTRAR_TIEMPOS = False

# ==================== FUNCIONES DE UTILIDAD ====================

def validar_tamano_kernel(tamano):
    """Valida que el tamaño del kernel sea válido (impar y dentro del rango)."""
    if tamano < TAMANO_KERNEL_MIN or tamano > TAMANO_KERNEL_MAX:
        raise ValueError(f"Tamaño de kernel debe estar entre {TAMANO_KERNEL_MIN} y {TAMANO_KERNEL_MAX}")
    if tamano % 2 == 0:
        raise ValueError("Tamaño de kernel debe ser impar")
    return True


def validar_probabilidad(prob):
    """Valida que la probabilidad esté en el rango válido."""
    if prob < 0 or prob > 1:
        raise ValueError("La probabilidad debe estar entre 0 y 1")
    return True


def validar_sigma(sigma):
    """Valida que sigma sea positivo."""
    if sigma <= 0:
        raise ValueError("Sigma debe ser positivo")
    return True


def imprimir_configuracion():
    """Imprime la configuración actual."""
    print("=" * 60)
    print("CONFIGURACIÓN ACTUAL - PRÁCTICA 2")
    print("=" * 60)
    print("\nRuido:")
    print(f"  - Prob. Sal y Pimienta: {PROB_SAL_PIMIENTA_DEFAULT}")
    print(f"  - Sigma Gaussiano: {SIGMA_GAUSSIANO_DEFAULT}")
    print("\nFiltros:")
    print(f"  - Tamaño kernel: {TAMANO_KERNEL_DEFAULT}x{TAMANO_KERNEL_DEFAULT}")
    print(f"  - Umbrales Canny: {CANNY_UMBRAL1_DEFAULT}, {CANNY_UMBRAL2_DEFAULT}")
    print("\nInterfaz:")
    print(f"  - Tamaño ventana: {VENTANA_ANCHO}x{VENTANA_ALTO}")
    print(f"  - Modo debug: {DEBUG_MODE}")
    print("=" * 60)


if __name__ == "__main__":
    # Mostrar configuración al ejecutar este archivo
    imprimir_configuracion()
