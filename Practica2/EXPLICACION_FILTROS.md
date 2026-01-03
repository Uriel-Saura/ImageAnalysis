# Guía Completa de Filtros de Procesamiento de Imágenes

Este documento explica en detalle todos los filtros lineales implementados en el sistema, incluyendo su teoría, aplicaciones y código.

## Tabla de Contenidos

1. [Filtros Paso Altas (High-Pass Filters)](#filtros-paso-altas)
   - [Operadores de Primer Orden](#operadores-de-primer-orden)
   - [Operadores de Segundo Orden](#operadores-de-segundo-orden)
2. [Filtros Paso Bajas (Low-Pass Filters)](#filtros-paso-bajas)
3. [Comparación de Filtros](#comparación-de-filtros)

---

## Filtros Paso Altas

Los filtros paso altas permiten el paso de las altas frecuencias de una imagen, lo que resulta en el realce de bordes y detalles. Son fundamentales para la detección de bordes y cambios abruptos de intensidad.

### Operadores de Primer Orden

Estos operadores calculan la primera derivada de la imagen, siendo sensibles a cambios graduales de intensidad.

#### 1. Filtro Sobel

**Teoría:**
El operador Sobel es uno de los más populares para detección de bordes. Utiliza dos kernels de 3x3 que calculan las derivadas parciales en direcciones horizontal (Gx) y vertical (Gy).

**Kernels:**
```
Gx = [-1  0  1]      Gy = [-1 -1 -1]
     [-2  0  2]           [ 0  0  0]
     [-1  0  1]           [ 1  1  1]
```

**Características:**
- Suaviza la imagen antes de calcular el gradiente
- Robusto al ruido
- Excelente para bordes bien definidos
- La magnitud del gradiente se calcula como: $G = \sqrt{Gx^2 + Gy^2}$

**Código:**
```python
def filtro_sobel(imagen):
    """
    Aplica el operador Sobel para detección de bordes.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Operadores Sobel
    sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitud del gradiente
    magnitud = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitud = np.uint8(np.clip(magnitud, 0, 255))
    
    return magnitud
```

**Aplicaciones:**
- Detección de bordes en imágenes con ruido moderado
- Preprocesamiento para segmentación
- Análisis de texturas

---

#### 2. Filtro Prewitt

**Teoría:**
Similar a Sobel, pero con pesos uniformes. Calcula el gradiente usando convolución con kernels direccionales.

**Kernels:**
```
Gx = [-1  0  1]      Gy = [-1 -1 -1]
     [-1  0  1]           [ 0  0  0]
     [-1  0  1]           [ 1  1  1]
```

**Características:**
- Pesos uniformes (no pondera el centro como Sobel)
- Más simple computacionalmente
- Sensible al ruido que Sobel
- Bueno para bordes con transiciones más suaves

**Código:**
```python
def filtro_prewitt(imagen):
    """
    Aplica el operador Prewitt para detección de bordes.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Máscaras Prewitt
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=np.float32)
    
    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]], dtype=np.float32)
    
    prewitt_x = cv2.filter2D(imagen, cv2.CV_64F, kernel_x)
    prewitt_y = cv2.filter2D(imagen, cv2.CV_64F, kernel_y)
    
    magnitud = np.sqrt(prewitt_x**2 + prewitt_y**2)
    magnitud = np.uint8(np.clip(magnitud, 0, 255))
    
    return magnitud
```

**Aplicaciones:**
- Detección de bordes en imágenes con poco ruido
- Análisis de orientación de gradientes

---

#### 3. Filtro Roberts

**Teoría:**
El operador más simple de los gradientes. Utiliza kernels de 2x2 que calculan diferencias diagonales, aproximando el gradiente en direcciones de 45° y 135°.

**Kernels:**
```
Gx = [ 1  0]      Gy = [ 0  1]
     [ 0 -1]           [-1  0]
```

**Características:**
- Kernels más pequeños (2x2)
- Muy rápido computacionalmente
- Muy sensible al ruido
- Ideal para imágenes de alta calidad sin ruido
- Detecta bordes diagonales eficientemente

**Código:**
```python
def filtro_roberts(imagen):
    """
    Aplica el operador Roberts para detección de bordes.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Máscaras Roberts
    kernel_x = np.array([[1, 0],
                         [0, -1]], dtype=np.float32)
    
    kernel_y = np.array([[0, 1],
                         [-1, 0]], dtype=np.float32)
    
    roberts_x = cv2.filter2D(imagen, cv2.CV_64F, kernel_x)
    roberts_y = cv2.filter2D(imagen, cv2.CV_64F, kernel_y)
    
    magnitud = np.sqrt(roberts_x**2 + roberts_y**2)
    magnitud = np.uint8(np.clip(magnitud, 0, 255))
    
    return magnitud
```

**Aplicaciones:**
- Imágenes médicas de alta calidad
- Detección rápida en sistemas en tiempo real
- Análisis de bordes diagonales

---

#### 4. Filtro Kirsch

**Teoría:**
Operador direccional que utiliza 8 máscaras (una por cada dirección de la brújula: N, NE, E, SE, S, SW, W, NW). Toma el máximo de todas las direcciones para obtener la magnitud del borde.

**Kernels (8 direcciones):**
```
N:  [ 5  5  5]    NE: [-3  5  5]    E:  [-3 -3  5]
    [-3  0 -3]        [-3  0  5]        [-3  0  5]
    [-3 -3 -3]        [-3 -3 -3]        [-3 -3  5]

SE: [-3 -3 -3]    S:  [-3 -3 -3]    SW: [-3 -3 -3]
    [-3  0  5]        [-3  0 -3]        [ 5  0 -3]
    [-3  5  5]        [ 5  5  5]        [ 5  5 -3]

W:  [ 5 -3 -3]    NW: [ 5  5 -3]
    [ 5  0 -3]        [ 5  0 -3]
    [ 5 -3 -3]        [-3 -3 -3]
```

**Características:**
- Detecta bordes en 8 direcciones
- Proporciona información direccional del borde
- Más robusto para bordes en cualquier orientación
- Computacionalmente más costoso (8 convoluciones)

**Código:**
```python
def filtro_kirsch(imagen):
    """
    Aplica el operador Kirsch para detección de bordes.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Las 8 máscaras direccionales de Kirsch
    kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
    ]
    
    # Aplicar todas las máscaras y tomar el máximo
    resultados = []
    for kernel in kernels:
        resultado = cv2.filter2D(imagen, cv2.CV_64F, kernel)
        resultados.append(resultado)
    
    magnitud = np.maximum.reduce(resultados)
    magnitud = np.uint8(np.clip(magnitud, 0, 255))
    
    return magnitud
```

**Aplicaciones:**
- Detección de bordes con orientación específica
- Análisis de patrones direccionales
- Reconocimiento de formas

---

#### 5. Filtro Canny

**Teoría:**
Considerado el detector de bordes óptimo. Utiliza un proceso multi-etapa:
1. Suavizado gaussiano para reducir ruido
2. Cálculo del gradiente con Sobel
3. Supresión de no-máximos
4. Umbralización con histéresis (dos umbrales)

**Características:**
- Detección óptima de bordes (criterios de Canny)
- Bordes delgados de un píxel
- Buena localización
- Única respuesta por borde
- Dos umbrales: bajo y alto para histéresis

**Código:**
```python
def filtro_canny(imagen, umbral1=100, umbral2=200):
    """
    Aplica el detector de bordes Canny.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    bordes = cv2.Canny(imagen, umbral1, umbral2)
    
    return bordes
```

**Parámetros:**
- `umbral1`: Umbral bajo para histéresis (típicamente 100)
- `umbral2`: Umbral alto para histéresis (típicamente 200)
- Relación recomendada: umbral2 = 2 × umbral1

**Aplicaciones:**
- Detección precisa de bordes
- Segmentación de objetos
- Análisis de formas
- Preprocesamiento para visión artificial

---

### Operadores de Segundo Orden

Estos operadores calculan la segunda derivada de la imagen, siendo sensibles a cambios rápidos de intensidad (cruces por cero).

#### 6. Laplaciano Clásico

**Teoría:**
Calcula la segunda derivada isotrópica de la imagen: $\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$

**Kernel:**
```
[ 0  1  0]
[ 1 -4  1]
[ 0  1  0]
```

**Características:**
- Isotrópico (independiente de la dirección)
- Detecta bordes y puntos aislados
- Muy sensible al ruido
- Respuesta de doble borde (positivo y negativo)
- No proporciona información direccional

**Código:**
```python
def filtro_laplaciano_clasico(imagen):
    """
    Aplica el operador Laplaciano con máscara clásica.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Máscara clásica del Laplaciano
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    
    laplaciano = cv2.filter2D(imagen, cv2.CV_64F, kernel)
    laplaciano = np.uint8(np.clip(np.abs(laplaciano), 0, 255))
    
    return laplaciano
```

**Aplicaciones:**
- Realce de bordes
- Detección de cambios rápidos
- Mejora de detalles finos

---

#### 7. Laplaciano 8 Vecinos

**Teoría:**
Considera los 8 vecinos del píxel central (4 ortogonales + 4 diagonales).

**Kernel:**
```
[ 1  1  1]
[ 1 -8  1]
[ 1  1  1]
```

**Características:**
- Más completo que el clásico
- Considera conexiones diagonales
- Mayor sensibilidad a detalles
- Más sensible al ruido

**Código:**
```python
def filtro_laplaciano_8_vecinos(imagen):
    """
    Aplica el operador Laplaciano con máscara de 8 vecinos.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Máscara de 8 vecinos
    kernel = np.array([[1, 1, 1],
                       [1, -8, 1],
                       [1, 1, 1]], dtype=np.float32)
    
    laplaciano = cv2.filter2D(imagen, cv2.CV_64F, kernel)
    laplaciano = np.uint8(np.clip(np.abs(laplaciano), 0, 255))
    
    return laplaciano
```

**Aplicaciones:**
- Detección de bordes en todas direcciones
- Realce de detalles finos
- Análisis de texturas

---

#### 8. Laplaciano Direccional Horizontal

**Teoría:**
Detecta cambios horizontales en la imagen (bordes verticales).

**Kernel:**
```
[ 0  0  0]
[ 1 -2  1]
[ 0  0  0]
```

**Código:**
```python
def filtro_laplaciano_horizontal(imagen):
    """
    Aplica el operador Laplaciano direccional horizontal.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    kernel = np.array([[0, 0, 0],
                       [1, -2, 1],
                       [0, 0, 0]], dtype=np.float32)
    
    laplaciano = cv2.filter2D(imagen, cv2.CV_64F, kernel)
    laplaciano = np.uint8(np.clip(np.abs(laplaciano), 0, 255))
    
    return laplaciano
```

**Aplicaciones:**
- Detección de líneas verticales
- Análisis de patrones horizontales

---

#### 9. Laplaciano Direccional Vertical

**Teoría:**
Detecta cambios verticales en la imagen (bordes horizontales).

**Kernel:**
```
[ 0  1  0]
[ 0 -2  0]
[ 0  1  0]
```

**Código:**
```python
def filtro_laplaciano_vertical(imagen):
    """
    Aplica el operador Laplaciano direccional vertical.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    kernel = np.array([[0, 1, 0],
                       [0, -2, 0],
                       [0, 1, 0]], dtype=np.float32)
    
    laplaciano = cv2.filter2D(imagen, cv2.CV_64F, kernel)
    laplaciano = np.uint8(np.clip(np.abs(laplaciano), 0, 255))
    
    return laplaciano
```

**Aplicaciones:**
- Detección de líneas horizontales
- Análisis de patrones verticales

---

#### 10. Laplaciano Diagonal Principal

**Teoría:**
Detecta cambios en la diagonal principal (↘).

**Kernel:**
```
[ 1  0  0]
[ 0 -2  0]
[ 0  0  1]
```

**Código:**
```python
def filtro_laplaciano_diagonal_principal(imagen):
    """
    Aplica el operador Laplaciano direccional diagonal principal.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    kernel = np.array([[1, 0, 0],
                       [0, -2, 0],
                       [0, 0, 1]], dtype=np.float32)
    
    laplaciano = cv2.filter2D(imagen, cv2.CV_64F, kernel)
    laplaciano = np.uint8(np.clip(np.abs(laplaciano), 0, 255))
    
    return laplaciano
```

**Aplicaciones:**
- Detección de líneas diagonales (↘)
- Análisis de patrones oblicuos

---

#### 11. Laplaciano Diagonal Secundaria

**Teoría:**
Detecta cambios en la diagonal secundaria (↙).

**Kernel:**
```
[ 0  0  1]
[ 0 -2  0]
[ 1  0  0]
```

**Código:**
```python
def filtro_laplaciano_diagonal_secundaria(imagen):
    """
    Aplica el operador Laplaciano direccional diagonal secundaria.
    """
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    kernel = np.array([[0, 0, 1],
                       [0, -2, 0],
                       [1, 0, 0]], dtype=np.float32)
    
    laplaciano = cv2.filter2D(imagen, cv2.CV_64F, kernel)
    laplaciano = np.uint8(np.clip(np.abs(laplaciano), 0, 255))
    
    return laplaciano
```

**Aplicaciones:**
- Detección de líneas diagonales (↙)
- Análisis de patrones oblicuos inversos

---

## Filtros Paso Bajas

Los filtros paso bajas suavizan la imagen eliminando altas frecuencias (ruido y detalles finos). Son útiles para reducir ruido y preparar imágenes para procesamiento posterior.

### 12. Filtro Promediador

**Teoría:**
El filtro más simple. Reemplaza cada píxel por el promedio de sus vecinos en una ventana de tamaño especificado.

**Kernel (ejemplo 3x3):**
```
[1/9  1/9  1/9]
[1/9  1/9  1/9]
[1/9  1/9  1/9]
```

**Características:**
- Muy simple y rápido
- Reduce ruido pero difumina bordes
- Todos los píxeles tienen el mismo peso
- Tamaño del kernel ajustable

**Código:**
```python
def filtro_promediador(imagen, tamano_kernel=5):
    """
    Aplica un filtro promediador (blur) simple.
    """
    resultado = cv2.blur(imagen, (tamano_kernel, tamano_kernel))
    return resultado
```

**Aplicaciones:**
- Reducción de ruido básica
- Preprocesamiento rápido
- Suavizado general

---

### 13. Filtro Promediador Pesado

**Teoría:**
Similar al promediador, pero con pesos mayores hacia el centro, dando más importancia al píxel central y sus vecinos cercanos.

**Kernel:**
```
[1/16  2/16  1/16]
[2/16  4/16  2/16]
[1/16  2/16  1/16]
```

**Características:**
- Mejor preservación del píxel central
- Suavizado más controlado
- Transición más suave que el promediador simple

**Código:**
```python
def filtro_promediador_pesado(imagen):
    """
    Aplica un filtro promediador con pesos.
    """
    # Kernel con pesos hacia el centro
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32) / 16
    
    if len(imagen.shape) == 3:
        resultado = cv2.filter2D(imagen, -1, kernel)
    else:
        resultado = cv2.filter2D(imagen, -1, kernel)
    
    return resultado
```

**Aplicaciones:**
- Suavizado moderado
- Reducción de ruido preservando estructura

---

### 14. Filtro Gaussiano

**Teoría:**
Utiliza la distribución gaussiana para asignar pesos. Los pesos disminuyen con la distancia según la función:

$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

**Características:**
- Mejor filtro de suavizado lineal
- Preserva mejor los bordes que el promediador
- Parámetro σ (sigma) controla el suavizado
- Separable (puede aplicarse en 1D dos veces)
- Teóricamente óptimo para reducir ruido gaussiano

**Código:**
```python
def filtro_gaussiano(imagen, tamano_kernel=5, sigma=1.0):
    """
    Aplica un filtro gaussiano para suavizado.
    """
    resultado = cv2.GaussianBlur(imagen, (tamano_kernel, tamano_kernel), sigma)
    return resultado
```

**Parámetros:**
- `tamano_kernel`: Tamaño de la ventana (debe ser impar)
- `sigma`: Desviación estándar de la gaussiana
  - σ pequeño: poco suavizado
  - σ grande: mucho suavizado

**Aplicaciones:**
- Reducción de ruido gaussiano
- Preprocesamiento para Canny
- Escalas múltiples (pirámides)

---

### 15. Filtro Bilateral

**Teoría:**
Filtro no lineal que preserva bordes mientras suaviza. Considera tanto la distancia espacial como la similitud de intensidad:

$$BF[I]_p = \frac{1}{W_p} \sum_{q \in S} G_{\sigma_s}(\|p-q\|) \cdot G_{\sigma_r}(|I_p - I_q|) \cdot I_q$$

Donde:
- $G_{\sigma_s}$ es la función gaussiana espacial
- $G_{\sigma_r}$ es la función gaussiana de rango (intensidad)

**Características:**
- **No lineal**: no puede expresarse como convolución simple
- Preserva bordes mientras suaviza regiones homogéneas
- Más lento que filtros lineales
- Dos parámetros sigma: espacial y de intensidad
- Excelente para mantener detalles importantes

**Código:**
```python
def filtro_bilateral(imagen, d=9, sigma_color=75, sigma_space=75):
    """
    Aplica un filtro bilateral que preserva bordes.
    """
    resultado = cv2.bilateralFilter(imagen, d, sigma_color, sigma_space)
    return resultado
```

**Parámetros:**
- `d`: Diámetro del vecindario de píxeles
- `sigma_color`: Filtro sigma en el espacio de color (mayor = más colores se mezclan)
- `sigma_space`: Filtro sigma en el espacio de coordenadas (mayor = más píxeles lejanos se influyen)

**Aplicaciones:**
- Reducción de ruido preservando bordes
- Suavizado de piel en fotografía
- Preprocesamiento para segmentación
- Efectos artísticos (cartoon, pintura)

---

## Comparación de Filtros

### Filtros Paso Altas - Detección de Bordes

| Filtro | Tamaño Kernel | Direccional | Sensibilidad Ruido | Velocidad | Mejor Uso |
|--------|---------------|-------------|-------------------|-----------|-----------|
| **Roberts** | 2×2 | Diagonal | Muy Alta | Muy Rápido | Imágenes de alta calidad |
| **Prewitt** | 3×3 | H/V | Alta | Rápido | Bordes suaves |
| **Sobel** | 3×3 | H/V | Media | Rápido | Uso general, robusto |
| **Kirsch** | 3×3 | 8 direcciones | Media | Lento | Análisis direccional |
| **Canny** | Multi-etapa | Todas | Baja | Medio | Detección precisa óptima |
| **Laplaciano** | 3×3 | Isotrópico | Muy Alta | Rápido | Realce de detalles |

### Filtros Paso Bajas - Suavizado

| Filtro | Preserva Bordes | Tiempo Cómputo | Calidad | Mejor Uso |
|--------|-----------------|----------------|---------|-----------|
| **Promediador** | No | Muy Rápido | Básica | Suavizado rápido simple |
| **Promediador Pesado** | Parcial | Rápido | Media | Balance velocidad-calidad |
| **Gaussiano** | Parcial | Rápido | Alta | Reducción ruido gaussiano |
| **Bilateral** | **Sí** | Lento | Muy Alta | Preservación de bordes |

### Recomendaciones de Uso

#### Para Detección de Bordes:
1. **Imagen con ruido**: Canny o Sobel
2. **Imagen limpia**: Roberts (más rápido)
3. **Análisis direccional**: Kirsch
4. **Realce general**: Laplaciano + Gaussiano (LoG)

#### Para Suavizado:
1. **Rapidez crítica**: Promediador
2. **Calidad general**: Gaussiano
3. **Preservar bordes**: Bilateral
4. **Balance**: Promediador pesado

### Técnicas Combinadas

#### Laplaciano de Gaussiana (LoG)
Combina suavizado gaussiano con Laplaciano para reducir sensibilidad al ruido:

```python
def log_filter(imagen, tamano_kernel=5, sigma=1.0):
    # Suavizar primero
    suavizada = cv2.GaussianBlur(imagen, (tamano_kernel, tamano_kernel), sigma)
    
    # Aplicar Laplaciano
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    
    resultado = cv2.filter2D(suavizada, cv2.CV_64F, kernel)
    return np.uint8(np.clip(np.abs(resultado), 0, 255))
```

#### Diferencia de Gaussianas (DoG)
Aproximación eficiente del LoG:

```python
def dog_filter(imagen, sigma1=1.0, sigma2=2.0):
    g1 = cv2.GaussianBlur(imagen, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(imagen, (0, 0), sigma2)
    return cv2.subtract(g1, g2)
```

---

## Ejemplo de Uso Completo

```python
import cv2
import numpy as np
from filtros_lineales import *

# Cargar imagen
imagen = cv2.imread('imagen.jpg')

# Convertir a escala de grises si es necesario
if len(imagen.shape) == 3:
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
else:
    gris = imagen.copy()

# --- FILTROS PASO ALTAS ---
# Detección de bordes con diferentes operadores
sobel = filtro_sobel(imagen)
prewitt = filtro_prewitt(imagen)
roberts = filtro_roberts(imagen)
kirsch = filtro_kirsch(imagen)
canny = filtro_canny(imagen, umbral1=100, umbral2=200)

# Laplacianos
lap_clasico = filtro_laplaciano_clasico(imagen)
lap_8 = filtro_laplaciano_8_vecinos(imagen)
lap_horizontal = filtro_laplaciano_horizontal(imagen)
lap_vertical = filtro_laplaciano_vertical(imagen)

# --- FILTROS PASO BAJAS ---
# Suavizado con diferentes técnicas
promedio = filtro_promediador(imagen, tamano_kernel=5)
promedio_pesado = filtro_promediador_pesado(imagen)
gaussiano = filtro_gaussiano(imagen, tamano_kernel=5, sigma=1.5)
bilateral = filtro_bilateral(imagen, d=9, sigma_color=75, sigma_space=75)

# Mostrar resultados
cv2.imshow('Original', imagen)
cv2.imshow('Sobel', sobel)
cv2.imshow('Canny', canny)
cv2.imshow('Gaussiano', gaussiano)
cv2.imshow('Bilateral', bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## Consideraciones Importantes

### Manejo de Bordes
Todos los filtros requieren manejar los bordes de la imagen. OpenCV ofrece varios métodos:
- `BORDER_CONSTANT`: Relleno con valor constante
- `BORDER_REPLICATE`: Replica el píxel del borde
- `BORDER_REFLECT`: Reflejo de borde
- `BORDER_WRAP`: Envoltura circular

### Normalización
Es importante normalizar los resultados para visualización:
```python
resultado = np.uint8(np.clip(resultado, 0, 255))
```

### Conversión a Escala de Grises
La mayoría de filtros de detección de bordes trabajan mejor en escala de grises:
```python
if len(imagen.shape) == 3:
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
```

### Optimización
- Use funciones nativas de OpenCV cuando estén disponibles (más rápidas)
- Considere el tamaño del kernel vs calidad necesaria
- Para procesamiento en tiempo real, prefiera filtros más simples

---

## Referencias Matemáticas

### Gradiente
$$\nabla f = \left[ \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right]$$

$$|\nabla f| = \sqrt{\left(\frac{\partial f}{\partial x}\right)^2 + \left(\frac{\partial f}{\partial y}\right)^2}$$

### Laplaciano
$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

### Convolución
$$(f * g)(x, y) = \sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} f(i, j) \cdot g(x-i, y-j)$$

---

## Conclusión

Este documento cubre los filtros lineales más importantes en procesamiento de imágenes. Cada filtro tiene sus ventajas y aplicaciones específicas:

- **Para principiantes**: Comience con Sobel y Gaussiano
- **Para producción**: Use Canny para bordes y Bilateral para suavizado
- **Para experimentar**: Pruebe Kirsch y los Laplacianos direccionales
- **Para velocidad**: Roberts y Promediador

La clave está en entender las características de cada filtro y seleccionar el apropiado según:
1. Tipo de imagen (con/sin ruido)
2. Objetivo (detección de bordes, suavizado, realce)
3. Restricciones de tiempo (velocidad vs calidad)
4. Requisitos de precisión

¡Experimente con diferentes parámetros para obtener los mejores resultados en sus aplicaciones específicas!
