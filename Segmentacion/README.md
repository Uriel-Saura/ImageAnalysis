# Módulo de Segmentación de Imágenes

Este módulo implementa diversas técnicas de segmentación, ecualización y ajuste de histograma para procesamiento de imágenes.

## Características Principales

### 1. Técnicas de Umbralización
- **Método de Otsu**: Determina un umbral óptimo minimizando la varianza intra-clase
- **Método de Entropía de Kapur**: Maximiza la entropía de las regiones segmentadas
- **Método del Mínimo de Histogramas**: Encuentra el mínimo entre dos picos del histograma
- **Método de la Media**: Utiliza la media de intensidades como umbral
- **Multiumbralización**: Segmenta la imagen en múltiples regiones
- **Umbralización por Banda**: Segmenta píxeles dentro de un rango específico
- **Umbralización Adaptativa**: Media y Gaussiana

### 2. Técnicas de Ecualización
- **Ecualización Uniforme**: Mejora contraste global
- **Ecualización Exponencial**: Resalta tonos oscuros
- **Ecualización Rayleigh**: Favorece regiones claras
- **Ecualización Hipercúbica**: Potencia diferencias extremas
- **Ecualización Logarítmica Hiperbólica**: Mejora detalles en sombras
- **CLAHE**: Ecualización adaptativa con limitación de contraste

### 3. Técnicas de Ajuste de Histograma
- **Función Potencia**: Ajuste no lineal de contraste
- **Corrección Gamma**: Control de brillo global
- **Desplazamiento**: Aumenta/disminuye brillo
- **Contracción**: Reduce contraste
- **Expansión**: Aumenta contraste
- **Transformación Logarítmica**: Expande valores oscuros

## Uso

### Interfaz Gráfica

```python
python Main.py
```

La interfaz permite:
- Cargar imágenes
- Aplicar diferentes técnicas de segmentación
- Comparar resultados de múltiples métodos
- Visualizar histogramas
- Guardar resultados

### Uso Programático

```python
from tecnicas_umbralizacion import metodo_otsu, metodo_entropia_kapur
from tecnicas_ecualizacion import ecualizacion_uniforme
import cv2

# Cargar imagen
imagen = cv2.imread('imagen.jpg')

# Segmentar
umbral, segmentada = metodo_otsu(imagen)

# Guardar
cv2.imwrite('resultado.jpg', segmentada)
```

## Estructura de Archivos

```
Segmentacion/
├── Main.py                        # Punto de entrada
├── interfaz_segmentacion.py      # Interfaz gráfica
├── tecnicas_umbralizacion.py     # Métodos de umbralización
├── tecnicas_ecualizacion.py      # Métodos de ecualización
├── tecnicas_ajuste.py            # Ajustes de histograma
├── preprocesamiento.py           # Funciones auxiliares
└── README.md                     # Este archivo
```

## Dependencias

- Python 3.7+
- OpenCV (cv2)
- NumPy
- Pillow (PIL)
- Matplotlib
- scikit-learn
- scipy
- tkinter (incluido en Python)

Instalar dependencias:
```bash
pip install opencv-python numpy pillow matplotlib scikit-learn scipy
```

## Casos de Uso

### Workflow Recomendado

1. **Cargar imagen** → Evaluar calidad visual
2. **Analizar histograma** → Identificar problemas de contraste/brillo
3. **Seleccionar técnica**:
   - Imágenes bimodales → Otsu
   - Múltiples regiones → Multiumbralización
   - Bajo contraste → Ecualización
   - Ajuste fino → Gamma/Potencia
4. **Comparar resultados** → Elegir mejor método
5. **Guardar resultado**

## Notas Técnicas

- Las imágenes se procesan en escala de grises para umbralización
- Los histogramas se calculan con 256 bins [0, 255]
- La interfaz muestra comparaciones lado a lado
- Todas las técnicas incluyen validación de parámetros
- Se soportan formatos: PNG, JPG, JPEG, BMP, TIF, TIFF
