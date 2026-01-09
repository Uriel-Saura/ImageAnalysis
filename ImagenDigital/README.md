# Procesamiento Básico de Imágenes Digitales

Aplicación con interfaz gráfica para operaciones fundamentales de procesamiento de imágenes.

## Funcionalidades

### 1. Conversión RGB a Escala de Grises
- Convierte imágenes a color a escala de grises
- Utiliza la conversión estándar de OpenCV
- Visualización lado a lado de original y resultado

### 2. Binarización de Imágenes

#### Umbral Fijo
- Control deslizante para ajustar el valor del umbral (0-255)
- Actualización en tiempo real del valor
- Píxeles por debajo del umbral → Negro (0)
- Píxeles por encima del umbral → Blanco (255)

#### Umbral Automático (Método de Otsu)
- Cálculo automático del umbral óptimo
- Minimiza la varianza intraclase
- Ideal para imágenes bimodales
- Muestra el valor del umbral calculado

### 3. Visualización de Histogramas
- Histograma de intensidad de la imagen original/grises
- Histograma de la imagen procesada
- Representación gráfica con matplotlib
- Permite analizar la distribución de intensidades

## Uso

### Ejecutar la aplicación:
```bash
python Main.py
```

### Flujo de trabajo:
1. **Cargar Imagen**: Abre una imagen desde el disco
2. **Convertir a Grises** (opcional): Convierte la imagen a escala de grises
3. **Aplicar Binarización**:
   - Ajustar umbral manualmente y aplicar
   - O usar umbral automático (Otsu)
4. **Mostrar Histogramas**: Visualiza la distribución de intensidades
5. **Guardar Resultado**: Guarda la imagen procesada

## Requisitos

- Python 3.7+
- OpenCV (cv2)
- NumPy
- Tkinter (incluido con Python)
- Pillow (PIL)
- Matplotlib

## Estructura de Archivos

```
ImagenDigital/
├── __init__.py                    # Inicialización del módulo
├── Main.py                        # Punto de entrada
├── procesamiento_basico.py        # Funciones de procesamiento
├── interfaz_imagen_digital.py     # Interfaz gráfica
└── README.md                      # Documentación
```

## Características Técnicas

- **Conversión a Grises**: Usa cv2.cvtColor con COLOR_BGR2GRAY
- **Binarización Fija**: cv2.threshold con THRESH_BINARY
- **Binarización Otsu**: cv2.threshold con THRESH_OTSU
- **Histogramas**: cv2.calcHist con 256 bins

## Ejemplos de Uso

### Conversión a Grises
```python
from procesamiento_basico import rgb_a_grises
imagen_gris = rgb_a_grises(imagen_color)
```

### Binarización con Umbral Fijo
```python
from procesamiento_basico import binarizacion_umbral_fijo
img_binaria, umbral = binarizacion_umbral_fijo(imagen, umbral=128)
```

### Binarización Automática
```python
from procesamiento_basico import binarizacion_umbral_otsu
img_binaria, umbral_calculado = binarizacion_umbral_otsu(imagen)
```

### Calcular Histograma
```python
from procesamiento_basico import calcular_histograma
histograma = calcular_histograma(imagen)
```
