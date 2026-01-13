# ImageAnalysis

Proyecto de análisis y procesamiento de imágenes digitales.

## Instalación

1. Crear y activar entorno virtual:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

```
ImageAnalysis/
├── AnalisisRuido/              # Análisis de ruido y filtros
│   └── Main.py
├── Final/                      # ★ Interfaz Principal Integrada ★
│   └── Main.py
├── Fourier/                    # Transformada de Fourier
│   └── Main.py
├── HeatMap/                    # Mapas de calor
│   └── Main.py
├── ImagenDigital/              # Procesamiento básico (RGB→Grises, Binarización, Histogramas)
│   └── Main.py
├── Morfologia/                 # Operaciones morfológicas
│   └── interfaz_morfologia.py
├── Operaciones/                # Operaciones con escalares, lógicas y aritméticas
│   └── Main.py
├── Proyecto/                   # ★ Reconocimiento de Texto (OCR) ★
│   └── Main.py
├── Segmentacion/               # Técnicas de segmentación
│   └── Main.py
└── img/                        # Imágenes de prueba
```

## Inicio Rápido

**Interfaz principal integrada (recomendado):**
```bash
python Final\Main.py
```

## Ejecutar Módulos Individuales

```bash
# Análisis de Ruido y Filtros
python AnalisisRuido\Main.py

# Transformada de Fourier
python Fourier\Main.py

# Mapas de Calor
python HeatMap\Main.py

# Procesamiento Básico de Imágenes Digitales
python ImagenDigital\Main.py

# Operaciones Morfológicas
python Morfologia\interfaz_morfologia.py

# Operaciones sobre Imágenes
python Operaciones\Main.py

# Técnicas de Segmentación
python Segmentacion\Main.py

# Reconocimiento de Texto (OCR)
python Proyecto\Main.py
```

## Módulo de Reconocimiento de Texto (OCR)

El nuevo módulo de OCR integra todas las técnicas de preprocesamiento del proyecto para mejorar la extracción de texto de imágenes.

### Características:

- **Preprocesamiento Inteligente**: Aplica automáticamente las mejores técnicas según el tipo de documento
- **Evaluación de Calidad**: Analiza contraste, nitidez, ruido e iluminación
- **Múltiples Perfiles**: Documentos escaneados, fotos digitales, capturas de pantalla, texto manuscrito
- **Motor OCR Flexible**: Soporte para Tesseract y EasyOCR
- **Exportación**: Guarda resultados en TXT, JSON, HTML o CSV
- **Visualización**: Muestra bounding boxes del texto detectado

### Instalación de Tesseract OCR:

1. **Windows**: Descargar de [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. **Linux**: `sudo apt-get install tesseract-ocr tesseract-ocr-spa`
3. **Mac**: `brew install tesseract tesseract-lang`

### Pipeline de Preprocesamiento:

El sistema aplica técnicas del proyecto según el tipo de documento:

1. **Conversión a escala de grises** (ImagenDigital)
2. **Reducción de ruido** (AnalisisRuido: filtros mediana, bilateral)
3. **Mejora de contraste** (Segmentacion: CLAHE, ecualización)
4. **Binarización** (ImagenDigital/Segmentacion: Otsu, adaptativa)
5. **Operaciones morfológicas** (Morfologia: apertura, cierre)

### Ejemplo de Uso:

```python
from Proyecto.motor_ocr import MotorOCR
from Proyecto.preprocesamiento_ocr import PipelinePreprocesamientoOCR
import cv2

# Cargar imagen
imagen = cv2.imread('documento.jpg')

# Preprocesar
pipeline = PipelinePreprocesamientoOCR()
imagen_procesada, _ = pipeline.procesar_con_perfil(imagen, 'documentos_escaneados')

# Extraer texto
motor = MotorOCR(motor='tesseract', idioma='spa')
resultados = motor.extraer_texto(imagen_procesada)
print(resultados['texto'])
```
