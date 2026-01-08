"""
Punto de entrada principal para la Práctica 2: Generación de Ruido y Aplicación de Filtros.

Este módulo inicia la interfaz gráfica de usuario para la práctica 2.

Funcionalidades incluidas:
- Generación de ruido (sal y pimienta, gaussiano)
- Filtros lineales paso altas (detección de bordes)
- Filtros lineales paso bajas (suavizado)
- Filtros no lineales (de orden)
- Visualización comparativa
"""

from interfaz_practica2 import iniciar_interfaz


def main():
    """
    Función principal que inicia la aplicación.
    """
    print("=" * 60)
    print("PRÁCTICA 2 - GENERACIÓN DE RUIDO Y APLICACIÓN DE FILTROS")
    print("=" * 60)
    print("\nIniciando interfaz gráfica...")
    print("\nFuncionalidades disponibles:")
    print("  ✓ Generación de ruido sal y pimienta")
    print("  ✓ Generación de ruido gaussiano")
    print("  ✓ Filtros paso altas (Sobel, Prewitt, Roberts, Kirsch, Canny)")
    print("  ✓ Filtros Laplacianos (clásico, 8 vecinos, direccionales)")
    print("  ✓ Filtros paso bajas (promediador, gaussiano, bilateral)")
    print("  ✓ Filtros no lineales (mediana, moda, máximo, mínimo)")
    print("  ✓ Visualización comparativa e histogramas")
    print("\n" + "=" * 60)
    
    # Iniciar interfaz gráfica
    iniciar_interfaz()


if __name__ == "__main__":
    main()
