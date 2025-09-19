### 🧠 Práctica 6: Introducción a Redes Neuronales

**Objetivos de la Práctica** 🎯

  * Conocer los *frameworks* clave de *Deep Learning*: **TensorFlow y Keras**.
  * Comprender los **principios básicos** de una neurona artificial.
  * Entender el rol de las **funciones de activación** y de **costo**.
  * Visualizar el proceso de **retropropagación** (*backpropagation*) y **optimización de parámetros**.
  * Implementar una red neuronal simple desde cero con **NumPy** para consolidar los conceptos.

**Duración aproximada:**
- 60 minutos.

**Tabla de ayuda:**

Para la ejecución del código ingresar a https://colab.research.google.com/ 

### **1. Introducción a *Frameworks* Clave: TensorFlow y Keras**

**TensorFlow** es una potente librería de código abierto para *machine learning* y *deep learning*. **Keras** es una API de alto nivel que se ejecuta sobre TensorFlow, diseñada para hacer que la construcción de modelos sea más rápida y sencilla. Juntos, permiten prototipar y desplegar redes neuronales con facilidad.

#### **Ejercicio:**

Crea un modelo de red neuronal simple con Keras para clasificar el conjunto de datos de Iris.

```python
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Cargar y preprocesar los datos
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir las etiquetas a one-hot encoding
y_train_one_hot = keras.utils.to_categorical(y_train, num_classes=3)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes=3)

# 1. Construir un modelo secuencial (ajustado para evitar la advertencia)
modelo = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

# 2. Compilar el modelo
modelo.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# 3. Entrenar el modelo
modelo.fit(X_train, y_train_one_hot, epochs=50, verbose=0)

# 4. Evaluar el modelo
loss, accuracy = modelo.evaluate(X_test, y_test_one_hot, verbose=0)
print(f"Precisión del modelo en el conjunto de prueba: {accuracy*100:.2f}%")
```

**Reto:** Cambia el número de neuronas en la capa oculta (`Dense`) del modelo de **8 a 16** y el número de épocas de **50 a 100**. ¿Mejora o empeora la precisión del modelo?

```python
# Pista de código para el reto:
# Pista: Modifica el argumento 'units' en la primera capa Dense y el argumento 'epochs' en la función fit().

modelo = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(16, activation='relu'),
    # ...
])
# ...
modelo.fit(X_train, y_train_one_hot, epochs=100, verbose=0)
# Luego, ejecuta y compara los resultados.
```

-----

### **2. La Neurona Artificial: Fundamentos**

Una neurona artificial es la unidad fundamental de una red neuronal. Recibe una o más entradas, las combina con un conjunto de **pesos** (*weights*), les aplica un **sesgo** (*bias*), y luego pasa el resultado a través de una **función de activación** para producir una salida.

#### **Ejercicio:**

Simula una neurona artificial simple con NumPy.

```python
import numpy as np

# Datos de entrada
entradas = np.array([1.5, 2.0, 3.0])

# Parámetros del modelo (pesos y sesgo)
pesos = np.array([0.5, -0.2, 0.8])
sesgo = 0.1

# 1. Calcular la suma ponderada (input * weights + bias)
suma_ponderada = np.dot(entradas, pesos) + sesgo
print(f"Suma ponderada: {suma_ponderada:.2f}")

# 2. Aplicar una función de activación (ej. Función ReLU)
def relu(x):
    return np.maximum(0, x)

salida = relu(suma_ponderada)
print(f"Salida de la neurona: {salida:.2f}")
```

**Reto:** Cambia la función de activación a la función **sigmoide** y recalcula la salida.

```python
# Pista de código para el reto:
# Pista: La función sigmoide se define como 1 / (1 + e^-x). En NumPy, puedes usar np.exp() para la exponenciación.

def sigmoide(x):
    # Tu código aquí para retornar la fórmula de la sigmoide
    pass

# Luego llama a tu nueva función con la suma_ponderada
# nueva_salida = sigmoide(suma_ponderada)
```

-----

### **3. *Backpropagation* y Optimización de Parámetros**

El proceso de *backpropagation* es el corazón del entrenamiento de una red neuronal. Consiste en calcular el **gradiente de la función de costo** con respecto a los pesos y sesgos, y luego usar ese gradiente para **actualizar los parámetros** del modelo y minimizar el error.

#### **Ejercicio:**

Simula una actualización de peso y sesgo para una neurona simple usando el gradiente.

```python
import numpy as np

# Valores iniciales
entrada = 2.0
peso = 0.5
sesgo = 0.1
tasa_aprendizaje = 0.01
valor_real = 1.0

# 1. Calcular la salida (predicción) y el error
prediccion = entrada * peso + sesgo
error = prediccion - valor_real

# 2. Calcular el gradiente (derivada del error con respecto a los parámetros)
gradiente_peso = error * entrada
gradiente_sesgo = error * 1  # Derivada del sesgo

# 3. Actualizar los parámetros
nuevo_peso = peso - tasa_aprendizaje * gradiente_peso
nuevo_sesgo = sesgo - tasa_aprendizaje * gradiente_sesgo

print(f"Peso inicial: {peso:.2f}, Nuevo peso: {nuevo_peso:.2f}")
print(f"Sesgo inicial: {sesgo:.2f}, Nuevo sesgo: {nuevo_sesgo:.2f}")
```

**Reto:** En el ejercicio anterior, realiza un segundo paso de **retropropagación** para ver cómo cambian los parámetros de nuevo.

```python
# Pista de código para el reto:
# Pista: Usa las variables 'nuevo_peso' y 'nuevo_sesgo' como punto de partida para el segundo paso.

# 1. Calcular la nueva predicción con los nuevos parámetros
# nueva_prediccion = entrada * nuevo_peso + nuevo_sesgo
# 2. Calcular el nuevo error
# nuevo_error = nueva_prediccion - valor_real
# 3. Calcular los nuevos gradientes
# nuevos_gradientes_peso = nuevo_error * entrada
# ...
# 4. Actualizar los parámetros por segunda vez
# peso_final = nuevo_peso - tasa_aprendizaje * nuevos_gradientes_peso
# ...
```

-----

### **4. Implementación de una Red Neuronal Simple**

Para cerrar la práctica, se implementa una red neuronal de una sola capa desde cero con NumPy, integrando los conceptos de la neurona artificial y *backpropagation*.

#### **Ejercicio:**

Construye una clase `RedNeuronal` para entrenar y predecir datos.

```python
import numpy as np

# Clases para la práctica
class RedNeuronal:
    def __init__(self, n_entradas, n_salidas):
        self.pesos = np.random.randn(n_entradas, n_salidas)
        self.sesgo = np.zeros((1, n_salidas))

    def predecir(self, X):
        return self.forward(X)

    def forward(self, X):
        self.suma_ponderada = np.dot(X, self.pesos) + self.sesgo
        return 1 / (1 + np.exp(-self.suma_ponderada)) # Activación Sigmoide

    def entrenar(self, X, y, tasa_aprendizaje, epochs):
        for _ in range(epochs):
            prediccion = self.forward(X)
            # Calcular el error
            error = y - prediccion
            # Backpropagation
            d_prediccion = prediccion * (1 - prediccion) # Derivada de la Sigmoide
            gradiente_pesos = np.dot(X.T, error * d_prediccion)
            gradiente_sesgo = np.sum(error * d_prediccion, axis=0, keepdims=True)
            # Actualizar pesos y sesgos
            self.pesos += tasa_aprendizaje * gradiente_pesos
            self.sesgo += tasa_aprendizaje * gradiente_sesgo

# Generar datos de ejemplo
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]]) # Problema XOR

# Entrenar la red neuronal
red_neuronal = RedNeuronal(n_entradas=2, n_salidas=1)
red_neuronal.entrenar(X_train, y_train, tasa_aprendizaje=0.1, epochs=1000)

# Predecir y mostrar resultados
print("Resultados de la predicción:")
print(red_neuronal.predecir(X_train).round(2))
```

**Reto:** Modifica el ejercicio para que resuelva el problema de clasificación **AND** en lugar del problema **XOR**.

```python
# Pista de código para el reto:
# Pista: Solo necesitas cambiar el vector `y_train` para que coincida con la tabla de verdad del operador AND.
# La tabla de verdad de AND es:
# (0, 0) -> 0
# (0, 1) -> 0
# (1, 0) -> 0
# (1, 1) -> 1

# Tu código aquí
```
