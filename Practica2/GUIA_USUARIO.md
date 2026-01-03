# Guía de Usuario - Práctica 2

## Tabla de Contenidos
1. [Instalación](#instalación)
2. [Inicio Rápido](#inicio-rápido)
3. [Guía Detallada](#guía-detallada)
4. [Ejemplos de Código](#ejemplos-de-código)
5. [Solución de Problemas](#solución-de-problemas)

---

## Instalación

### Paso 1: Instalar Dependencias

Desde la raíz del proyecto `ImageAnalysis`, ejecuta:

```bash
pip install -r requirements.txt
```

### Paso 2: Verificar Instalación

```bash
python -c "import cv2, numpy, matplotlib, scipy; print('Todas las dependencias instaladas correctamente')"
```

---

## Inicio Rápido

### Método 1: Interfaz Gráfica (Recomendado)

```bash
cd Practica2
python Main.py
```

### Método 2: Script de Ejemplo

```bash
cd Practica2
python ejemplo_uso.py
```

---

## Guía Detallada

### 1. Generación de Ruido

#### Ruido Sal y Pimienta

Este tipo de ruido afecta píxeles aleatorios, convirtiéndolos en valores extremos (255 o 0).

**Parámetros:**
- `probabilidad`: Porcentaje de píxeles afectados (0.01 - 0.2)
  - Bajo (0.01-0.05): Ruido ligero
  - Medio (0.05-0.10): Ruido moderado
  - Alto (0.10-0.20): Ruido severo

**Cuándo usar:**
- Simular defectos en sensores CCD/CMOS
- Pruebas de robustez de algoritmos
- Evaluación de filtros de mediana

**Mejor filtro para remover:** Filtro de Mediana

#### Ruido Gaussiano

Añade ruido basado en una distribución normal (campana de Gauss).

**Parámetros:**
- `media`: Valor central de la distribución (generalmente 0)
- `sigma`: Desviación estándar (5-100)
  - Bajo (5-20): Ruido sutil
  - Medio (20-50): Ruido visible
  - Alto (50-100): Ruido muy notable

**Cuándo usar:**
- Simular ruido térmico en sensores
- Simular ruido de cuantización
- Pruebas de filtros de suavizado

**Mejor filtro para remover:** Filtro Gaussiano o Bilateral

---

### 2. Filtros Paso Altas (Detección de Bordes)

#### Operadores de Primer Orden

Estos operadores calculan la primera derivada de la imagen para detectar cambios de intensidad.

**Sobel**
- **Ventajas:** Buen balance entre detección y reducción de ruido
- **Uso recomendado:** Detección general de bordes
- **Sensibilidad al ruido:** Media

**Prewitt**
- **Ventajas:** Similar a Sobel, más simple computacionalmente
- **Uso recomendado:** Cuando se necesita velocidad
- **Sensibilidad al ruido:** Media

**Roberts**
- **Ventajas:** Muy rápido, kernel 2x2
- **Uso recomendado:** Bordes diagonales
- **Sensibilidad al ruido:** Alta
- **Desventajas:** Más sensible al ruido

**Kirsch**
- **Ventajas:** Detecta 8 direcciones, muy robusto
- **Uso recomendado:** Bordes con dirección específica
- **Sensibilidad al ruido:** Baja
- **Desventajas:** Más lento (8 convoluciones)

**Canny**
- **Ventajas:** Mejor detector de bordes, bordes finos y continuos
- **Uso recomendado:** Cuando se necesita máxima calidad
- **Sensibilidad al ruido:** Baja (incluye suavizado gaussiano)
- **Parámetros:**
  - `umbral1`: Umbral inferior (50-100)
  - `umbral2`: Umbral superior (150-250)

#### Operadores de Segundo Orden (Laplacianos)

Calculan la segunda derivada, sensibles a cambios rápidos de intensidad.

**Laplaciano Clásico**
- Máscara: 4 vecinos
- Uso: Detección general de bordes

**Laplaciano 8 Vecinos**
- Máscara: 8 vecinos (incluye diagonales)
- Uso: Detección más sensible

**Laplacianos Direccionales**
- **Horizontal:** Detecta bordes horizontales
- **Vertical:** Detecta bordes verticales
- **Diagonal Principal:** Detecta bordes en diagonal \
- **Diagonal Secundaria:** Detecta bordes en diagonal /

**Nota:** Los Laplacianos son muy sensibles al ruido. Se recomienda aplicar un filtro gaussiano primero.

---

### 3. Filtros Paso Bajas (Suavizado)

#### Filtro Promediador (Blur)

**Descripción:** Promedia todos los píxeles en la vecindad.

**Parámetros:**
- `tamano_kernel`: 3, 5, 7, 9, 11, etc. (impar)

**Efectos:**
- Kernel pequeño (3x3): Suavizado ligero
- Kernel mediano (5x5-7x7): Suavizado moderado
- Kernel grande (9x9+): Suavizado fuerte, puede difuminar detalles

**Uso recomendado:**
- Reducir ruido gaussiano ligero
- Pre-procesamiento rápido

#### Filtro Promediador Pesado

**Descripción:** Promedio con pesos, da más importancia al centro.

**Ventajas:**
- Preserva mejor los detalles que el promediador simple
- Menos difuminado

**Uso recomendado:**
- Cuando se necesita balance entre suavizado y preservación

#### Filtro Gaussiano

**Descripción:** Usa una distribución gaussiana para los pesos.

**Parámetros:**
- `tamano_kernel`: Tamaño del kernel (impar)
- `sigma`: Desviación estándar (0.5-3.0)

**Ventajas:**
- Suavizado natural
- Preserva mejor los bordes que el promediador
- Matemáticamente óptimo

**Uso recomendado:**
- Reducir ruido gaussiano
- Pre-procesamiento para detección de bordes
- Aplicaciones que requieren calidad

#### Filtro Bilateral

**Descripción:** Suaviza pero preserva bordes usando pesos espaciales y de intensidad.

**Parámetros:**
- `d`: Diámetro del vecindario (5-9)
- `sigma_color`: Sigma en espacio de color (50-150)
- `sigma_space`: Sigma en espacio coordenado (50-150)

**Ventajas:**
- **Excelente preservación de bordes**
- Reduce ruido sin difuminar contornos

**Uso recomendado:**
- Cuando es crítico preservar bordes
- Fotografía computacional
- Procesamiento de rostros

**Desventajas:**
- Más lento que otros filtros

---

### 4. Filtros No Lineales (de Orden)

#### Filtro de Mediana

**Descripción:** Reemplaza cada píxel con la mediana de su vecindad.

**Parámetros:**
- `tamano_kernel`: 3, 5, 7, etc.

**Ventajas:**
- **Excelente para ruido sal y pimienta**
- Preserva bordes
- No introduce nuevos valores

**Uso recomendado:**
- Remover ruido sal y pimienta
- Cuando se deben preservar bordes

**Nota:** Kernel 5x5 es óptimo en la mayoría de casos

#### Filtro de Moda

**Descripción:** Reemplaza con el valor más frecuente en la vecindad.

**Ventajas:**
- Bueno para imágenes con áreas uniformes
- Preserva valores dominantes

**Uso recomendado:**
- Imágenes con colores discretos
- Después de segmentación

**Desventajas:**
- Lento en imágenes con muchos valores únicos

#### Filtro de Máximo

**Descripción:** Reemplaza con el valor máximo en la vecindad (dilación).

**Efectos:**
- Expande regiones claras
- Reduce regiones oscuras

**Uso recomendado:**
- Remover ruido pimienta (píxeles oscuros)
- Operaciones morfológicas

#### Filtro de Mínimo

**Descripción:** Reemplaza con el valor mínimo en la vecindad (erosión).

**Efectos:**
- Expande regiones oscuras
- Reduce regiones claras

**Uso recomendado:**
- Remover ruido sal (píxeles claros)
- Operaciones morfológicas

---

### 5. Filtros Avanzados (Opcionales)

#### Mediana Adaptativa

**Descripción:** Ajusta el tamaño del kernel según las características locales.

**Ventajas:**
- Mejor preservación de detalles
- Adaptativo al nivel de ruido

**Uso recomendado:**
- Ruido sal y pimienta variable

#### Contraharmonic Mean

**Descripción:** Media contraarmónica con parámetro Q.

**Parámetros:**
- `Q > 0`: Elimina ruido pimienta
- `Q < 0`: Elimina ruido sal
- `Q = 0`: Equivale a media aritmética

**Uso recomendado:**
- Cuando se conoce el tipo de ruido dominante

#### Mediana Ponderada

**Descripción:** Mediana con pesos según distancia al centro.

**Ventajas:**
- Mejor preservación del píxel central
- Menos difuminado

**Uso recomendado:**
- Alternativa a la mediana estándar

---

## Ejemplos de Código

### Ejemplo 1: Pipeline Completo de Procesamiento

```python
import cv2
from Practica2 import (
    aplicar_ruido_sal_pimienta,
    filtro_mediana,
    filtro_canny
)

# 1. Cargar imagen
imagen = cv2.imread('foto.jpg')

# 2. Añadir ruido
imagen_ruido = aplicar_ruido_sal_pimienta(imagen, probabilidad=0.05)

# 3. Limpiar ruido
imagen_limpia = filtro_mediana(imagen_ruido, tamano_kernel=5)

# 4. Detectar bordes
bordes = filtro_canny(imagen_limpia, umbral1=100, umbral2=200)

# 5. Guardar resultados
cv2.imwrite('resultado_limpio.jpg', imagen_limpia)
cv2.imwrite('resultado_bordes.jpg', bordes)
```

### Ejemplo 2: Comparar Múltiples Filtros

```python
import cv2
import matplotlib.pyplot as plt
from Practica2 import (
    filtro_sobel,
    filtro_prewitt,
    filtro_roberts,
    filtro_canny
)

imagen = cv2.imread('foto.jpg')

# Aplicar diferentes detectores
sobel = filtro_sobel(imagen)
prewitt = filtro_prewitt(imagen)
roberts = filtro_roberts(imagen)
canny = filtro_canny(imagen)

# Visualizar
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].imshow(sobel, cmap='gray')
axes[0, 0].set_title('Sobel')
axes[0, 1].imshow(prewitt, cmap='gray')
axes[0, 1].set_title('Prewitt')
axes[1, 0].imshow(roberts, cmap='gray')
axes[1, 0].set_title('Roberts')
axes[1, 1].imshow(canny, cmap='gray')
axes[1, 1].set_title('Canny')
plt.show()
```

### Ejemplo 3: Análisis de Ruido con Histogramas

```python
import cv2
import matplotlib.pyplot as plt
from Practica2 import aplicar_ruido_gaussiano, calcular_histograma

imagen = cv2.imread('foto.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar ruido
imagen_ruido = aplicar_ruido_gaussiano(imagen, media=0, sigma=25)

# Calcular histogramas
hist_original = calcular_histograma(imagen)
hist_ruido = calcular_histograma(imagen_ruido)

# Visualizar
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(hist_original[0])
plt.title('Histograma Original')

plt.subplot(1, 2, 2)
plt.plot(hist_ruido[0])
plt.title('Histograma con Ruido Gaussiano')

plt.show()
```

---

## Solución de Problemas

### Problema: "ModuleNotFoundError: No module named 'cv2'"

**Solución:**
```bash
pip install opencv-python
```

### Problema: "ModuleNotFoundError: No module named 'scipy'"

**Solución:**
```bash
pip install scipy
```

### Problema: La interfaz no se muestra

**Solución:**
- Verifica que tkinter esté instalado (viene con Python en Windows/Mac)
- En Linux: `sudo apt-get install python3-tk`

### Problema: Filtros muy lentos

**Soluciones:**
- Reduce el tamaño del kernel
- Usa imágenes más pequeñas para pruebas
- Evita el filtro de moda con kernels grandes
- Los filtros opcionales (mediana adaptativa, moda) son más lentos

### Problema: La imagen se ve muy oscura después del filtro

**Solución:**
- Algunos filtros (como Laplacianos) pueden producir valores negativos
- Los valores se recortan a [0, 255]
- Considera normalizar el resultado

### Problema: Los bordes detectados son muy ruidosos

**Solución:**
1. Aplica un filtro gaussiano antes del detector de bordes
2. Ajusta los umbrales del detector Canny
3. Reduce el ruido de la imagen original primero

---

## Tips y Mejores Prácticas

### Para Ruido

1. **Sal y Pimienta:** Usa filtro de mediana (kernel 5x5)
2. **Gaussiano:** Usa filtro gaussiano o bilateral
3. **Mixto:** Aplica mediana primero, luego gaussiano

### Para Detección de Bordes

1. Siempre pre-procesa con filtro gaussiano
2. Usa Canny para mejor calidad
3. Usa Sobel para rapidez
4. Experimenta con diferentes umbrales

### Para Suavizado

1. Empieza con kernels pequeños (3x3, 5x5)
2. Usa bilateral si los bordes son importantes
3. Usa gaussiano para propósito general
4. Evita kernels muy grandes (>11x11)

### General

1. Guarda resultados intermedios para comparación
2. Usa imágenes de prueba pequeñas primero
3. Documenta los parámetros que funcionan mejor
4. Compara visualmente diferentes filtros

---

## Recursos Adicionales

### Documentación
- OpenCV: https://docs.opencv.org/
- NumPy: https://numpy.org/doc/
- Matplotlib: https://matplotlib.org/

### Lecturas Recomendadas
- "Digital Image Processing" - Gonzalez & Woods
- "Computer Vision: Algorithms and Applications" - Richard Szeliski

---

**Última actualización:** Diciembre 2025
