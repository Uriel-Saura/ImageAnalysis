# Documentación Técnica - Práctica 2

## Resumen de Implementación

Este documento describe los detalles técnicos de la implementación de los algoritmos de generación de ruido y filtros.

---

## 1. Generación de Ruido

### 1.1 Ruido Sal y Pimienta

**Algoritmo:**
```
Para cada píxel (x, y):
    Generar número aleatorio r en [0, 1]
    Si r < probabilidad/2:
        píxel = 255 (sal - blanco)
    Si r > 1 - probabilidad/2:
        píxel = 0 (pimienta - negro)
```

**Implementación:**
- Usa `numpy.random.random()` para generar matriz de valores aleatorios
- Operaciones vectorizadas para eficiencia
- Complejidad: O(n×m) donde n×m es el tamaño de la imagen

**Características:**
- Afecta píxeles de forma independiente
- No introduce valores intermedios
- Simulación realista de defectos en sensores

---

### 1.2 Ruido Gaussiano

**Algoritmo:**
```
Para cada píxel (x, y):
    ruido = Normal(media, sigma²)
    píxel_nuevo = clip(píxel_original + ruido, 0, 255)
```

**Implementación:**
- Usa `numpy.random.normal()` para generar distribución normal
- Parámetros: media (μ) y desviación estándar (σ)
- Recorte de valores fuera del rango [0, 255]
- Complejidad: O(n×m)

**Características:**
- Distribución normal: P(x) = (1/(σ√(2π))) × e^(-(x-μ)²/(2σ²))
- Simula ruido térmico y de cuantización
- Valores continuos en todo el rango

---

## 2. Filtros Lineales Paso Altas

### 2.1 Operador Sobel

**Kernels:**
```
Gx = [-1  0  1]      Gy = [-1 -2 -1]
     [-2  0  2]           [ 0  0  0]
     [-1  0  1]           [ 1  2  1]
```

**Magnitud del Gradiente:**
```
G = √(Gx² + Gy²)
```

**Características:**
- Aproximación de primera derivada
- Kernel 3×3 con suavizado integrado
- Buena respuesta a bordes en todas direcciones

---

### 2.2 Operador Prewitt

**Kernels:**
```
Gx = [-1  0  1]      Gy = [-1 -1 -1]
     [-1  0  1]           [ 0  0  0]
     [-1  0  1]           [ 1  1  1]
```

**Características:**
- Similar a Sobel, pesos uniformes
- Más simple computacionalmente
- Sensibilidad similar al ruido que Sobel

---

### 2.3 Operador Roberts

**Kernels:**
```
Gx = [ 1  0]         Gy = [ 0  1]
     [ 0 -1]              [-1  0]
```

**Características:**
- Kernel 2×2 (el más pequeño)
- Rápido de calcular
- Mejor respuesta a bordes diagonales
- Más sensible al ruido

---

### 2.4 Operador Kirsch

**Descripción:**
Usa 8 máscaras direccionales para detectar bordes en todas las orientaciones.

**Máscaras (ejemplo - dirección Norte):**
```
[ 5  5  5]
[-3  0 -3]
[-3 -3 -3]
```

**Algoritmo:**
```
Para cada píxel:
    Aplicar las 8 máscaras
    Tomar el máximo valor absoluto
```

**Características:**
- Muy robusto
- Detecta dirección del borde
- Más costoso (8 convoluciones)

---

### 2.5 Detector de Bordes Canny

**Algoritmo Multi-Etapa:**

1. **Suavizado Gaussiano:**
   - Reduce ruido antes de la detección

2. **Cálculo de Gradientes:**
   - Usa operador Sobel
   - Calcula magnitud y dirección

3. **Supresión No-Máxima:**
   - Adelgaza bordes a un píxel de ancho
   - Preserva solo máximos locales en dirección del gradiente

4. **Umbralización con Histéresis:**
   - Umbral alto (T2): bordes fuertes
   - Umbral bajo (T1): bordes débiles
   - Conecta bordes débiles a fuertes

**Ventajas:**
- Bordes finos y bien localizados
- Buena detección con poco ruido
- Reducción de falsos positivos

**Parámetros Recomendados:**
- T1: 100 (umbral bajo)
- T2: 200 (umbral alto)
- Relación T2/T1 ≈ 2:1 o 3:1

---

### 2.6 Operador Laplaciano

**Kernel Clásico (4 vecinos):**
```
[ 0  1  0]
[ 1 -4  1]
[ 0  1  0]
```

**Kernel 8 Vecinos:**
```
[ 1  1  1]
[ 1 -8  1]
[ 1  1  1]
```

**Máscaras Direccionales:**

Horizontal:
```
[ 0  0  0]
[ 1 -2  1]
[ 0  0  0]
```

Vertical:
```
[ 0  1  0]
[ 0 -2  0]
[ 0  1  0]
```

Diagonal Principal:
```
[ 1  0  0]
[ 0 -2  0]
[ 0  0  1]
```

Diagonal Secundaria:
```
[ 0  0  1]
[ 0 -2  0]
[ 1  0  0]
```

**Características:**
- Operador de segunda derivada: ∇²f = ∂²f/∂x² + ∂²f/∂y²
- Isotrópico (respuesta igual en todas direcciones)
- Muy sensible al ruido
- Detecta cambios rápidos de intensidad

---

## 3. Filtros Lineales Paso Bajas

### 3.1 Filtro Promediador

**Kernel (ejemplo 3×3):**
```
[1/9  1/9  1/9]
[1/9  1/9  1/9]
[1/9  1/9  1/9]
```

**Fórmula:**
```
f'(x,y) = (1/n²) × Σ f(x+i, y+j)
```

**Características:**
- Promedio simple de la vecindad
- Suavizado uniforme
- Puede causar difuminado excesivo
- Complejidad: O(n×m×k²) donde k es tamaño del kernel

---

### 3.2 Filtro Promediador Pesado

**Kernel (ejemplo 3×3):**
```
[1/16  2/16  1/16]
[2/16  4/16  2/16]
[1/16  2/16  1/16]
```

**Características:**
- Pesos decrecientes desde el centro
- Preserva mejor los detalles
- Menos difuminado que promedio simple

---

### 3.3 Filtro Gaussiano

**Función Gaussiana 2D:**
```
G(x,y) = (1/(2πσ²)) × e^(-(x²+y²)/(2σ²))
```

**Kernel (ejemplo 5×5, σ=1.0):**
```
[0.003  0.013  0.022  0.013  0.003]
[0.013  0.059  0.097  0.059  0.013]
[0.022  0.097  0.159  0.097  0.022]
[0.013  0.059  0.097  0.059  0.013]
[0.003  0.013  0.022  0.013  0.003]
```

**Características:**
- Suavizado natural
- Pesos basados en distribución gaussiana
- Separable: G(x,y) = G(x) × G(y)
- Parámetro σ controla el nivel de suavizado
- Matemáticamente óptimo (minimiza producto espacio-frecuencia)

**Propiedades:**
- Transformada de Fourier de una Gaussiana es otra Gaussiana
- No introduce artefactos (ripples)
- Preserva mejor los bordes que promediador simple

---

### 3.4 Filtro Bilateral

**Fórmula:**
```
f'(x,y) = (1/W) × Σ f(xi,yi) × Gs(||p-pi||) × Gr(|f(p)-f(pi)|)

Donde:
- Gs: Gaussiana espacial (distancia geométrica)
- Gr: Gaussiana de rango (diferencia de intensidad)
- W: Factor de normalización
```

**Componentes:**

1. **Peso Espacial:**
   ```
   ws(x,y) = e^(-(x²+y²)/(2σs²))
   ```

2. **Peso de Intensidad:**
   ```
   wr(Δf) = e^(-Δf²/(2σr²))
   ```

**Características:**
- Filtro no lineal edge-preserving
- Combina cercanía espacial y similitud de intensidad
- Preserva bordes mientras suaviza regiones uniformes
- Más costoso computacionalmente

**Parámetros:**
- σ_space: controla tamaño del vecindario
- σ_color: controla cuánto puede variar la intensidad

---

## 4. Filtros No Lineales (de Orden)

### 4.1 Filtro de Mediana

**Algoritmo:**
```
Para cada píxel (x, y):
    Extraer ventana W de tamaño k×k centrada en (x,y)
    Ordenar valores en W
    píxel_salida = mediana(W)
```

**Ejemplo (ventana 3×3):**
```
Ventana:        Ordenado:       Mediana:
[3 5 1]        [1 2 3]
[7 2 9]   →    [3 5 7]    →    5
[4 6 8]        [6 8 9]
```

**Características:**
- No lineal
- Robusto a outliers
- Excelente para ruido sal y pimienta
- Preserva bordes
- Complejidad: O(n×m×k²log(k))

**Ventajas:**
- No introduce nuevos valores de intensidad
- No desplaza bordes
- Efectivo contra ruido impulsivo

---

### 4.2 Filtro de Moda

**Algoritmo:**
```
Para cada píxel (x, y):
    Extraer ventana W
    Contar frecuencia de cada valor
    píxel_salida = valor más frecuente
```

**Características:**
- Preserva valores dominantes
- Útil en imágenes con regiones uniformes
- Puede ser lento con muchos valores únicos
- No efectivo con ruido gaussiano

---

### 4.3 Filtro de Máximo (Dilación)

**Algoritmo:**
```
Para cada píxel (x, y):
    píxel_salida = max{f(x+i, y+j) | (i,j) ∈ W}
```

**Efectos:**
- Expande regiones brillantes
- Reduce regiones oscuras
- Elimina ruido pimienta (píxeles negros)

**Aplicaciones:**
- Operaciones morfológicas
- Reducción de ruido pimienta

---

### 4.4 Filtro de Mínimo (Erosión)

**Algoritmo:**
```
Para cada píxel (x, y):
    píxel_salida = min{f(x+i, y+j) | (i,j) ∈ W}
```

**Efectos:**
- Expande regiones oscuras
- Reduce regiones brillantes
- Elimina ruido sal (píxeles blancos)

**Aplicaciones:**
- Operaciones morfológicas
- Reducción de ruido sal

---

### 4.5 Filtro de Mediana Adaptativa

**Algoritmo:**
```
Para cada píxel (x, y):
    kernel_size = kernel_inicial
    Mientras kernel_size <= kernel_max:
        Calcular mediana en ventana de tamaño kernel_size
        Si píxel != mediana Y |píxel - mediana| > umbral:
            píxel_salida = mediana
            break
        kernel_size += 2
    Si no reemplazado:
        píxel_salida = píxel_original
```

**Características:**
- Ajusta tamaño de ventana dinámicamente
- Mejor preservación de detalles
- Más efectivo con ruido variable
- Más lento que mediana estándar

---

### 4.6 Filtro Contraharmonic Mean

**Fórmula:**
```
f'(x,y) = Σ f(x,y)^(Q+1) / Σ f(x,y)^Q
```

**Parámetros:**
- Q > 0: Elimina ruido pimienta
- Q < 0: Elimina ruido sal
- Q = 0: Media aritmética
- Q = -1: Media armónica

**Características:**
- Efectivo contra ruido impulsivo específico
- Requiere conocimiento del tipo de ruido
- Puede introducir artefactos si Q mal elegido

---

### 4.7 Filtro de Mediana Ponderada

**Algoritmo:**
```
Para cada píxel (x, y):
    Extraer ventana W
    Asignar pesos según distancia al centro
    Crear lista ponderada (replicar valores según peso)
    píxel_salida = mediana(lista_ponderada)
```

**Matriz de Pesos (ejemplo 3×3):**
```
[1  2  1]
[2  4  2]
[1  2  1]
```

**Características:**
- Variante de mediana que favorece píxeles centrales
- Mejor preservación del píxel original
- Menos difuminado

---

## 5. Análisis de Complejidad

### Complejidad Temporal

| Filtro | Complejidad | Observaciones |
|--------|-------------|---------------|
| Ruido S&P | O(n×m) | Operaciones vectorizadas |
| Ruido Gaussiano | O(n×m) | Operaciones vectorizadas |
| Sobel/Prewitt | O(n×m) | Kernel fijo 3×3 |
| Roberts | O(n×m) | Kernel fijo 2×2 |
| Kirsch | O(8×n×m) | 8 convoluciones |
| Canny | O(n×m) | Múltiples pasos optimizados |
| Laplaciano | O(n×m) | Kernel fijo |
| Promediador | O(n×m×k²) | k = tamaño kernel |
| Gaussiano | O(n×m×k) | Separable |
| Bilateral | O(n×m×k²) | No separable |
| Mediana | O(n×m×k²log k) | Ordenamiento |
| Moda | O(n×m×k²) | Conteo de frecuencias |
| Máximo/Mínimo | O(n×m×k²) | Búsqueda de extremo |

Donde:
- n×m: dimensiones de la imagen
- k: tamaño del kernel

---

## 6. Métricas de Calidad

### 6.1 Error Cuadrático Medio (MSE)

```
MSE = (1/(n×m)) × Σ [f(x,y) - f'(x,y)]²
```

### 6.2 Relación Señal-Ruido de Pico (PSNR)

```
PSNR = 10 × log₁₀(MAX²/MSE)
```

Donde MAX = 255 para imágenes de 8 bits.

**Interpretación:**
- PSNR > 40 dB: Excelente calidad
- 30-40 dB: Buena calidad
- 20-30 dB: Calidad aceptable
- < 20 dB: Baja calidad

### 6.3 Índice de Similitud Estructural (SSIM)

```
SSIM(x,y) = [l(x,y)^α × c(x,y)^β × s(x,y)^γ]
```

Donde:
- l(x,y): comparación de luminancia
- c(x,y): comparación de contraste
- s(x,y): comparación de estructura

**Interpretación:**
- SSIM = 1: Imágenes idénticas
- SSIM → 0: Imágenes muy diferentes

---

## 7. Recomendaciones de Uso

### Para Diferentes Tipos de Ruido

| Tipo de Ruido | Filtro Recomendado | Alternativa |
|---------------|-------------------|-------------|
| Sal y Pimienta | Mediana (5×5) | Mediana Adaptativa |
| Gaussiano | Gaussiano | Bilateral |
| Mixto | Mediana + Gaussiano | Bilateral |
| Pimienta solo | Máximo | Contraharmonic (Q>0) |
| Sal solo | Mínimo | Contraharmonic (Q<0) |

### Para Detección de Bordes

| Objetivo | Filtro Recomendado | Configuración |
|----------|-------------------|---------------|
| Máxima calidad | Canny | T1=100, T2=200 |
| Rapidez | Sobel | Kernel 3×3 |
| Bordes direccionales | Kirsch | 8 máscaras |
| Segunda derivada | Laplaciano 8-vecinos | Suavizar primero |

---

## 8. Consideraciones de Implementación

### Manejo de Bordes

**Métodos implementados:**
1. **Reflect:** Refleja píxeles en el borde
2. **Constant:** Rellena con valor constante (0)
3. **Replicate:** Replica píxeles del borde

### Optimizaciones

1. **Operaciones Vectorizadas:**
   - Uso de NumPy para evitar loops en Python

2. **Separabilidad:**
   - Filtro Gaussiano implementado como dos pases 1D

3. **Uso de OpenCV:**
   - Funciones optimizadas en C++ para operaciones comunes

4. **Tipo de Datos:**
   - Cálculos en float32/float64
   - Conversión a uint8 al final

---

## Referencias

1. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.

2. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications* (2nd ed.). Springer.

3. Canny, J. (1986). A Computational Approach to Edge Detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-8(6), 679-698.

4. Tomasi, C., & Manduchi, R. (1998). Bilateral filtering for gray and color images. *Sixth International Conference on Computer Vision*.

5. OpenCV Documentation: https://docs.opencv.org/

6. NumPy Documentation: https://numpy.org/doc/

---

**Autor:** ImageAnalysis Project  
**Fecha:** Diciembre 2025  
**Versión:** 1.0.0
