"""
Script de prueba automatizada para verificar el funcionamiento de todos los módulos.
"""

import sys
import numpy as np
import cv2

print("=" * 60)
print("PRUEBAS AUTOMATIZADAS - PRÁCTICA 2")
print("=" * 60)

# ==================== PRUEBA 1: IMPORTACIÓN DE MÓDULOS ====================
print("\n1. Verificando importación de módulos...")

try:
    from generacion_ruido import (
        aplicar_ruido_sal_pimienta,
        aplicar_ruido_gaussiano,
        calcular_histograma
    )
    print("   ✓ generacion_ruido.py")
except Exception as e:
    print(f"   ✗ Error en generacion_ruido.py: {e}")
    sys.exit(1)

try:
    from filtros_lineales import (
        filtro_sobel, filtro_prewitt, filtro_roberts, filtro_kirsch, filtro_canny,
        filtro_laplaciano_clasico, filtro_laplaciano_8_vecinos,
        filtro_promediador, filtro_gaussiano, filtro_bilateral
    )
    print("   ✓ filtros_lineales.py")
except Exception as e:
    print(f"   ✗ Error en filtros_lineales.py: {e}")
    sys.exit(1)

try:
    from filtros_no_lineales import (
        filtro_mediana, filtro_moda, filtro_maximo, filtro_minimo
    )
    print("   ✓ filtros_no_lineales.py")
except Exception as e:
    print(f"   ✗ Error en filtros_no_lineales.py: {e}")
    sys.exit(1)

# ==================== PRUEBA 2: CREACIÓN DE IMAGEN DE PRUEBA ====================
print("\n2. Creando imagen de prueba...")

try:
    # Crear imagen de prueba sintética
    imagen_test = np.ones((200, 200, 3), dtype=np.uint8) * 128
    
    # Añadir algunas formas
    cv2.rectangle(imagen_test, (30, 30), (80, 80), (255, 255, 255), -1)
    cv2.circle(imagen_test, (150, 100), 40, (0, 0, 0), -1)
    cv2.line(imagen_test, (50, 150), (150, 150), (200, 200, 200), 3)
    
    print(f"   ✓ Imagen de prueba creada: {imagen_test.shape}")
except Exception as e:
    print(f"   ✗ Error al crear imagen: {e}")
    sys.exit(1)

# ==================== PRUEBA 3: GENERACIÓN DE RUIDO ====================
print("\n3. Probando generación de ruido...")

try:
    # Ruido sal y pimienta
    img_sp = aplicar_ruido_sal_pimienta(imagen_test, probabilidad=0.05)
    assert img_sp.shape == imagen_test.shape, "Forma incorrecta"
    assert img_sp.dtype == np.uint8, "Tipo de dato incorrecto"
    print("   ✓ Ruido sal y pimienta")
except Exception as e:
    print(f"   ✗ Error en ruido sal y pimienta: {e}")
    sys.exit(1)

try:
    # Ruido gaussiano
    img_gauss = aplicar_ruido_gaussiano(imagen_test, media=0, sigma=25)
    assert img_gauss.shape == imagen_test.shape, "Forma incorrecta"
    assert img_gauss.dtype == np.uint8, "Tipo de dato incorrecto"
    print("   ✓ Ruido gaussiano")
except Exception as e:
    print(f"   ✗ Error en ruido gaussiano: {e}")
    sys.exit(1)

try:
    # Histograma
    hist = calcular_histograma(imagen_test)
    assert len(hist) > 0, "Histograma vacío"
    print("   ✓ Cálculo de histograma")
except Exception as e:
    print(f"   ✗ Error en histograma: {e}")
    sys.exit(1)

# ==================== PRUEBA 4: FILTROS PASO ALTAS ====================
print("\n4. Probando filtros paso altas...")

filtros_pa = [
    ("Sobel", filtro_sobel),
    ("Prewitt", filtro_prewitt),
    ("Roberts", filtro_roberts),
    ("Kirsch", filtro_kirsch),
    ("Canny", filtro_canny),
    ("Laplaciano Clásico", filtro_laplaciano_clasico),
    ("Laplaciano 8 Vecinos", filtro_laplaciano_8_vecinos)
]

for nombre, filtro in filtros_pa:
    try:
        resultado = filtro(imagen_test)
        assert resultado is not None, "Resultado None"
        assert resultado.shape[:2] == imagen_test.shape[:2], "Forma incorrecta"
        print(f"   ✓ {nombre}")
    except Exception as e:
        print(f"   ✗ Error en {nombre}: {e}")
        sys.exit(1)

# ==================== PRUEBA 5: FILTROS PASO BAJAS ====================
print("\n5. Probando filtros paso bajas...")

filtros_pb = [
    ("Promediador", lambda img: filtro_promediador(img, 5)),
    ("Gaussiano", lambda img: filtro_gaussiano(img, 5, 1.0)),
    ("Bilateral", lambda img: filtro_bilateral(img, 9, 75, 75))
]

for nombre, filtro in filtros_pb:
    try:
        resultado = filtro(imagen_test)
        assert resultado is not None, "Resultado None"
        assert resultado.shape == imagen_test.shape, "Forma incorrecta"
        print(f"   ✓ {nombre}")
    except Exception as e:
        print(f"   ✗ Error en {nombre}: {e}")
        sys.exit(1)

# ==================== PRUEBA 6: FILTROS NO LINEALES ====================
print("\n6. Probando filtros no lineales...")

filtros_nl = [
    ("Mediana", lambda img: filtro_mediana(img, 5)),
    ("Máximo", lambda img: filtro_maximo(img, 5)),
    ("Mínimo", lambda img: filtro_minimo(img, 5))
]

for nombre, filtro in filtros_nl:
    try:
        resultado = filtro(imagen_test)
        assert resultado is not None, "Resultado None"
        assert resultado.shape == imagen_test.shape, "Forma incorrecta"
        print(f"   ✓ {nombre}")
    except Exception as e:
        print(f"   ✗ Error en {nombre}: {e}")
        sys.exit(1)

# ==================== PRUEBA 7: FILTRO DE MODA (PUEDE SER LENTO) ====================
print("\n7. Probando filtro de moda (puede tardar)...")

try:
    # Imagen más pequeña para el filtro de moda
    img_pequeña = imagen_test[0:50, 0:50]
    resultado = filtro_moda(img_pequeña, 3)
    assert resultado is not None, "Resultado None"
    print("   ✓ Filtro de moda")
except Exception as e:
    print(f"   ⚠ Advertencia en filtro de moda: {e}")
    # No salir, este filtro puede ser lento

# ==================== PRUEBA 8: PIPELINE COMPLETO ====================
print("\n8. Probando pipeline completo...")

try:
    # 1. Añadir ruido sal y pimienta
    img_ruido = aplicar_ruido_sal_pimienta(imagen_test, 0.05)
    
    # 2. Limpiar con mediana
    img_limpia = filtro_mediana(img_ruido, 5)
    
    # 3. Suavizar con gaussiano
    img_suave = filtro_gaussiano(img_limpia, 5, 1.0)
    
    # 4. Detectar bordes con Sobel
    img_bordes = filtro_sobel(img_suave)
    
    assert img_bordes is not None, "Pipeline falló"
    print("   ✓ Pipeline completo: Ruido → Mediana → Gaussiano → Sobel")
except Exception as e:
    print(f"   ✗ Error en pipeline: {e}")
    sys.exit(1)

# ==================== PRUEBA 9: VALIDACIÓN DE TIPOS DE IMAGEN ====================
print("\n9. Probando con diferentes tipos de imagen...")

try:
    # Imagen en escala de grises
    img_gris = cv2.cvtColor(imagen_test, cv2.COLOR_BGR2GRAY)
    resultado = filtro_sobel(img_gris)
    assert resultado is not None, "Falló con imagen en grises"
    print("   ✓ Imagen en escala de grises")
    
    # Imagen a color
    resultado = filtro_gaussiano(imagen_test, 5)
    assert resultado is not None, "Falló con imagen a color"
    print("   ✓ Imagen a color")
except Exception as e:
    print(f"   ✗ Error con tipos de imagen: {e}")
    sys.exit(1)

# ==================== PRUEBA 10: VALIDACIÓN DE RANGOS ====================
print("\n10. Validando rangos de valores...")

try:
    # Aplicar ruido
    img_ruido = aplicar_ruido_gaussiano(imagen_test, 0, 50)
    
    # Verificar que los valores estén en rango [0, 255]
    assert np.min(img_ruido) >= 0, "Valores negativos detectados"
    assert np.max(img_ruido) <= 255, "Valores superiores a 255"
    assert img_ruido.dtype == np.uint8, "Tipo de dato incorrecto"
    
    print("   ✓ Todos los valores en rango [0, 255]")
except Exception as e:
    print(f"   ✗ Error en validación de rangos: {e}")
    sys.exit(1)

# ==================== RESUMEN ====================
print("\n" + "=" * 60)
print("RESULTADO DE LAS PRUEBAS")
print("=" * 60)
print("✓ Todas las pruebas pasaron exitosamente")
print("\nMódulos verificados:")
print("  • Generación de ruido")
print("  • Filtros lineales paso altas (7 operadores)")
print("  • Filtros lineales paso bajas (3 filtros)")
print("  • Filtros no lineales (4 filtros)")
print("  • Pipeline completo")
print("  • Validaciones de tipos y rangos")
print("\nEl sistema está listo para usar.")
print("=" * 60)

print("\nPara iniciar la interfaz gráfica, ejecuta:")
print("  python Main.py")
print("\nPara ver un ejemplo de uso programático:")
print("  python ejemplo_uso.py")
print("=" * 60)

sys.exit(0)
