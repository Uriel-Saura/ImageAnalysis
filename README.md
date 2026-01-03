# ImageAnalysis

Proyecto de análisis de imágenes con operaciones morfológicas y transformada de Fourier.

## Requisitos

- Python 3.12+

## Instalación

1. Crear entorno virtual:
```bash
python -m venv .venv
```

2. Activar entorno virtual:
- Windows (PowerShell):
```bash
.venv\Scripts\Activate.ps1
```
- Windows (CMD):
```bash
.venv\Scripts\activate.bat
```
- Linux/Mac:
```bash
source .venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

```
ImageAnalysis/
├── Fourier/                    # Módulo de Transformada de Fourier
│   ├── Main.py                # Punto de entrada
│   ├── interfaz_fourier.py    # Interfaz gráfica
│   ├── logica_fft.py          # Lógica de FFT
│   ├── filtros_fourier.py     # Filtros (pasa-bajas, pasa-altas)
│   ├── analisis_filtros.py    # Análisis de efectos de filtrado
│   └── visualizador.py        # Visualización de imágenes
├── HeatMap/                   # Módulo de Mapas de Calor
│   ├── Main.py
│   ├── interfaz_heatmap.py
│   └── logica_heatmap.py
├── Morfologia/                # Módulo de Morfología
│   ├── interfaz_morfologia.py
│   ├── mostrar_todas_operaciones.py
│   └── verificar_tipo_imagen.py
├── Practica2/                 # Módulo de Ruido y Filtros
│   ├── Main.py                # Punto de entrada
│   ├── interfaz_practica2.py  # Interfaz gráfica completa
│   ├── generacion_ruido.py    # Generación de ruido (sal y pimienta, gaussiano)
│   ├── filtros_lineales.py    # Filtros paso altas y bajas
│   ├── filtros_no_lineales.py # Filtros de orden
│   ├── visualizador.py        # Funciones de visualización
│   ├── ejemplo_uso.py         # Script de ejemplo
│   ├── README.md              # Documentación del módulo
│   └── GUIA_USUARIO.md        # Guía detallada de usuario
├── img/                       # Carpeta de imágenes
└── requirements.txt           # Dependencias del proyecto
```

## Uso

### Módulo de Morfología
```bash
python Morfologia\interfaz_morfologia.py
```

### Módulo de Fourier
```bash
python Fourier\interfaz_fourier.py
```

### Módulo de HeatMap
```bash
python HeatMap\Main.py
```

### Módulo Práctica 2 (Ruido y Filtros)
```bash
cd Practica2
python Main.py
```

O ejecutar el ejemplo:
```bash
cd Practica2
python ejemplo_uso.py
```

## Funcionalidades

### Morfología
- Erosión, Dilatación, Apertura, Cierre
- Frontera, Adelgazamiento, Hit-or-Miss
- Esqueleto, Gradientes morfológicos
- Top Hat, Black Hat, Filtro combinado

### Fourier
- Transformada de Fourier (FFT)
- Visualización de magnitud y fase
- Filtros pasa-bajas: Ideal, Gaussiano, Butterworth
- Filtros pasa-altas: Ideal, Gaussiano, Butterworth
- Análisis de efectos del filtrado (MSE, PSNR, SSIM)

### HeatMap
- Generación de mapas de calor
- Visualización de distribución de intensidades
- Análisis de zonas de interés

### Práctica 2 - Ruido y Filtros
#### Generación de Ruido:
- Ruido sal y pimienta (configurable)
- Ruido gaussiano (media y sigma ajustables)
- Visualización de histogramas

#### Filtros Lineales Paso Altas (Detección de Bordes):
- Operadores de primer orden: Sobel, Prewitt, Roberts, Kirsch, Canny
- Operadores de segundo orden: Laplacianos (clásico, 8 vecinos, direccionales)

#### Filtros Lineales Paso Bajas (Suavizado):
- Filtro promediador (blur)
- Filtro promediador pesado
- Filtro gaussiano
- Filtro bilateral (preserva bordes)

#### Filtros No Lineales (de Orden):
- Filtros básicos: Mediana, Moda, Máximo, Mínimo
- Filtros avanzados: Mediana adaptativa, Contraharmonic Mean, Mediana ponderada
