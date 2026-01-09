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
```
