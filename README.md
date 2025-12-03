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
├── Morfologia/                # Módulo de Morfología
│   ├── interfaz_morfologia.py
│   ├── mostrar_todas_operaciones.py
│   └── verificar_tipo_imagen.py
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
