"""
Main - Pipeline Detallado OCR Paso a Paso
Ejecutar: python Proyecto/Main_Pipeline_Detallado.py
"""

import sys
import os

# Configurar ruta
directorio_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if directorio_raiz not in sys.path:
    sys.path.insert(0, directorio_raiz)

from Proyecto.interfaz_pipeline_detallado import main

if __name__ == "__main__":
    main()
