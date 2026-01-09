# Operaciones sobre Imágenes

Aplicación con interfaz gráfica para realizar operaciones matemáticas y lógicas sobre imágenes.

## Funcionalidades

### 1. Operaciones con Escalares
Operaciones que se aplican a cada píxel de una imagen con un valor constante:

- **Suma Escalar**: Aumenta el brillo sumando un valor a cada píxel
- **Resta Escalar**: Reduce el brillo restando un valor a cada píxel
- **Multiplicación Escalar**: Amplifica la intensidad multiplicando por un factor
- **División Escalar**: Reduce la intensidad dividiendo por un factor

**Rango del escalar**: -255 a 255

### 2. Operaciones Lógicas
Operaciones bit a bit entre dos imágenes (o sobre una imagen):

- **AND**: Realiza AND lógico entre píxeles de ambas imágenes
  - Útil para crear máscaras
  - Resultado: Solo píxeles comunes brillantes
  
- **OR**: Realiza OR lógico entre píxeles de ambas imágenes
  - Combina regiones brillantes de ambas imágenes
  
- **XOR**: Realiza XOR lógico entre píxeles de ambas imágenes
  - Detecta diferencias entre imágenes
  - Resultado: Solo píxeles diferentes
  
- **NOT**: Invierte los píxeles de una imagen
  - Negativo de la imagen
  - Solo requiere una imagen

### 3. Operaciones Aritméticas
Operaciones píxel a píxel entre dos imágenes:

- **Suma Ponderada**: Combina dos imágenes con pesos ajustables
  - Permite crear transiciones y mezclas
  - Control independiente de peso para cada imagen (0.0 a 1.0)
  
- **Resta**: Resta píxel a píxel (Imagen1 - Imagen2)
  - Detección de cambios
  - Útil para segmentación por diferencia
  
- **Multiplicación**: Multiplica píxeles de ambas imágenes
  - Máscara de región de interés
  - Efecto de oscurecimiento selectivo
  
- **División**: Divide píxeles (Imagen1 / Imagen2)
  - Normalización
  - Corrección de iluminación
  
- **Diferencia Absoluta**: |Imagen1 - Imagen2|
  - Detección de cambios sin signo
  - Comparación de imágenes

## Uso

### Ejecutar la aplicación:
```bash
python Main.py
```

### Flujo de trabajo:

#### Operaciones con Escalares:
1. **Cargar Imagen 1**: Abre la imagen a procesar
2. **Ajustar valor escalar**: Usa el control deslizante
3. **Aplicar operación**: Elige suma, resta, multiplicación o división
4. **Ver resultado**: Se muestra en la tercera columna

#### Operaciones Lógicas:
1. **Cargar Imagen 1 y 2**: Abre ambas imágenes (excepto para NOT)
2. **Aplicar operación lógica**: AND, OR, XOR o NOT
3. **Ver resultado**: Operación bit a bit visualizada

#### Operaciones Aritméticas:
1. **Cargar Imagen 1 y 2**: Abre ambas imágenes
2. **Ajustar pesos** (para suma ponderada)
3. **Aplicar operación**: Suma, resta, multiplicación, división o diferencia
4. **Ver resultado**: Combinación de ambas imágenes

## Requisitos

- Python 3.7+
- OpenCV (cv2)
- NumPy
- Tkinter (incluido con Python)
- Pillow (PIL)

## Estructura de Archivos

```
Operaciones/
├── __init__.py                      # Inicialización del módulo
├── Main.py                          # Punto de entrada
├── procesamiento_operaciones.py     # Funciones de procesamiento
├── interfaz_operaciones.py          # Interfaz gráfica
└── README.md                        # Documentación
```

## Características Técnicas

### Operaciones con Escalares
- **Saturación**: Los valores se mantienen entre 0-255 (clip)
- **Conversión automática**: Las imágenes RGB se convierten a grises

### Operaciones Lógicas
- **Bit a bit**: Operaciones sobre representación binaria
- **Redimensionamiento**: Imagen 2 se ajusta al tamaño de Imagen 1
- **cv2.bitwise_and/or/xor/not**: Funciones optimizadas de OpenCV

### Operaciones Aritméticas
- **Normalización**: Multiplicación normalizada por 255
- **Protección**: División protegida contra división por cero
- **Suma ponderada**: cv2.addWeighted para mejor rendimiento

## Ejemplos de Uso

### Operaciones con Escalares
```python
# Aumentar brillo
resultado = suma_escalar(imagen, 50)

# Reducir contraste
resultado = multiplicacion_escalar(imagen, 0.5)
```

### Operaciones Lógicas
```python
# Crear máscara
mascara = operacion_and(imagen1, imagen2)

# Combinar regiones
union = operacion_or(imagen1, imagen2)

# Invertir imagen
invertida = operacion_not(imagen)
```

### Operaciones Aritméticas
```python
# Mezcla 50-50
mezcla = suma_imagenes(img1, img2, 0.5, 0.5)

# Detectar cambios
cambios = diferencia_absoluta(img1, img2)

# Máscara de ROI
roi = multiplicacion_imagenes(imagen, mascara)
```

## Aplicaciones Comunes

- **Ajuste de brillo/contraste**: Operaciones con escalares
- **Detección de movimiento**: Diferencia absoluta entre frames
- **Transiciones de video**: Suma ponderada con pesos variables
- **Segmentación por color**: Operaciones lógicas con máscaras
- **Corrección de iluminación**: División de imágenes
- **Efectos artísticos**: Multiplicación y mezcla de imágenes
