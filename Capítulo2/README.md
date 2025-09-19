He reformateado el texto a un formato Markdown estándar y claro para que el código se muestre correctamente en GitHub. Los títulos, subtítulos, listas y bloques de código están ahora correctamente estructurados para una fácil lectura y copia.

-----

# 🤖 Preprocesamiento de Datos y Análisis Exploratorio

¡Bienvenido a la segunda práctica\! Aquí te centrarás en los pasos cruciales para preparar los datos antes de construir un modelo de *machine learning*.

-----

## 🎯 Práctica 2.1: Limpieza, Transformación y *Feature Engineering*

### **Objetivos**

  * Comprender y aplicar técnicas de **limpieza de datos** para manejar valores nulos y atípicos.
  * Realizar **transformaciones** esenciales como el escalado de datos numéricos y la codificación de variables categóricas.
  * Crear nuevas variables (*features*) a través del ***Feature Engineering*** para mejorar el rendimiento de los modelos.

### **1. Limpieza de Datos: Nulos y *Outliers***

Antes de analizar los datos, es vital asegurarse de que estén limpios. Los **valores nulos** (`NaN`) y los **valores atípicos** (*outliers*) pueden sesgar los resultados.

#### **Ejercicio 1: Gestión de Valores Nulos**

Identifica los valores nulos en el *DataFrame* y usa la imputación por la media para rellenarlos.

```python
import pandas as pd
import numpy as np

# Datos de ejemplo con valores nulos
data = {'Edad': [25, 30, np.nan, 45, 30, 45],
        'Ventas': [150, 200, 180, 220, np.nan, 190],
        'Región': ['Norte', 'Sur', 'Norte', 'Sur', 'Norte', 'Norte']}
df_ejemplo = pd.DataFrame(data)

print("DataFrame con valores nulos:")
print(df_ejemplo)
print("-" * 30)

# Calcular la media de la columna 'Ventas'
media_ventas = df_ejemplo['Ventas'].mean()
print(f"Media de la columna 'Ventas': {media_ventas}")

# Imputar valores nulos con la media
df_ejemplo['Ventas'] = df_ejemplo['Ventas'].fillna(media_ventas)

print("DataFrame después de la imputación:")
print(df_ejemplo)
```

**Reto:** En el *DataFrame* anterior, identifica los nulos en la columna `Edad` y usa la **imputación por la mediana** para rellenarlos. Explica brevemente por qué la mediana puede ser una mejor opción que la media.

```python
# Pista de Código para el Reto:
# Pista 1: El método .median() te dará la mediana de una columna.
# Pista 2: El método .fillna() es el mismo que se usó para las ventas.
# Pista 3: La mediana es más robusta frente a valores atípicos.

# Tu código aquí
```

-----

### **2. Transformación de Datos: Escalado y Codificación**

Para que los modelos de *machine learning* funcionen correctamente, los datos a menudo deben ser transformados. El **escalado** pone las variables en la misma escala, mientras que la **codificación** convierte variables categóricas en números.

#### **Ejercicio 2: Escalado de Datos Numéricos**

Usa el `StandardScaler` de Scikit-learn para escalar las columnas `Ventas` y `Edad`.

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Datos de ejemplo
data = {'Edad': [25, 30, 35, 45, 30, 45],
        'Ventas': [150, 200, 180, 220, 250, 190]}
df_transformacion = pd.DataFrame(data)

# Inicializar el escalador
scaler = StandardScaler()

# Escalar las columnas numéricas
df_scaled = scaler.fit_transform(df_transformacion[['Edad', 'Ventas']])

# Convertir el resultado a un DataFrame para visualizar
df_scaled = pd.DataFrame(df_scaled, columns=['Edad_escalada', 'Ventas_escaladas'])

print("DataFrame después del escalado:")
print(df_scaled)
```

**Reto:** Codifica la columna `Región` usando ***One-Hot Encoding*** para convertir las categorías en columnas numéricas. Explica por qué esta técnica es útil para el *machine learning*.

```python
# Pista de Código para el Reto:
# Pista 1: Pandas tiene una función muy útil para esto: pd.get_dummies().
# Pista 2: La técnica de One-Hot Encoding crea una nueva columna por cada categoría.
# Pista 3: Los modelos de ML no pueden trabajar directamente con texto.

# Tu código aquí
```

-----

### **3. *Feature Engineering* Básico**

El ***Feature Engineering*** es el proceso de crear nuevas variables a partir de las existentes. Una buena *feature* puede mejorar significativamente el rendimiento del modelo.

#### **Ejercicio 3: Creación de Variables Derivadas**

Crea una nueva columna llamada `VentaPorEdad` que sea el resultado de dividir `Ventas` entre `Edad`.

```python
import pandas as pd

# DataFrame con datos limpios
data = {'Edad': [25, 30, 35, 45, 30, 45],
        'Ventas': [150, 200, 180, 220, 250, 190]}
df_fe = pd.DataFrame(data)

# Crear la nueva feature
df_fe['VentaPorEdad'] = df_fe['Ventas'] / df_fe['Edad']

print("DataFrame con la nueva variable:")
print(df_fe)
```

**Reto:** A partir de la columna `Edad`, crea una nueva *feature* categórica llamada `GrupoEdad` con las siguientes categorías: `'Joven'` (menor a 35), y `'Adulto'` (35 o más).

```python
# Pista de Código para el Reto:
# Pista 1: Puedes usar el método .apply() de Pandas con una función lambda.
# Pista 2: El método .apply() se ejecuta sobre cada elemento de la serie.
# Pista 3: La sintaxis para la función lambda es "lambda x: ...".

# Tu código aquí
```

-----

### **4. Reto Final de Código: Ciclo de Preprocesamiento Completo** 💡

**Descripción del Problema:**
Tienes un conjunto de datos desordenado. Tu objetivo es aplicar todo lo aprendido en esta práctica para prepararlo para un modelo de *machine learning*.

**Tarea:**

1.  **Limpieza:** Imputa los valores nulos de la columna `Puntuacion` con el valor 0.
2.  **Transformación:** Escala la columna `Puntuacion`.
3.  ***Feature Engineering*:** Crea una nueva variable llamada `Puntuacion_log` aplicando el logaritmo natural (`np.log()`) a la columna `Puntuacion`.

<!-- end list -->

```python
# Pista de Código para el Reto:
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Datos de ejemplo
datos_reto = {'ID_Usuario': [1, 2, 3, 4, 5],
              'Puntuacion': [100, 250, np.nan, 500, 150]}
df_reto = pd.DataFrame(datos_reto)

# Pista 1: Usa .fillna(0) para la imputación.
# Pista 2: Usa MinMaxScaler() en lugar de StandardScaler() para este reto.
# Pista 3: El logaritmo se aplica a una columna completa.

# Tu código aquí
```

-----

## 📊 Práctica 2.2: Análisis Exploratorio y Preparación de Datos

### **Objetivos**

  * Realizar un **análisis exploratorio de datos (EDA)** utilizando visualizaciones.
  * Comprender la importancia de dividir los datos en conjuntos de **entrenamiento y prueba**.
  * Conocer el concepto de **validación cruzada** para evaluar modelos de manera robusta.

### **1. Análisis Exploratorio con Visualizaciones**

El **Análisis Exploratorio de Datos (EDA)** es un paso clave para entender las características de un conjunto de datos. Las visualizaciones nos permiten identificar patrones, tendencias y la distribución de las variables.

#### **Ejercicio:**

Usa un conjunto de datos simple para visualizar la relación entre `Edad` y `VentaTotal`, diferenciando a los clientes por su `Estado` (Activo/Inactivo).

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Datos de ejemplo
data = {'Edad': [25, 30, 45, 50, 28, 35, 60],
        'VentaTotal': [150, 200, 180, 220, 250, 190, 300],
        'Estado': ['Activo', 'Activo', 'Inactivo', 'Activo', 'Inactivo', 'Activo', 'Inactivo']}
df_exploracion = pd.DataFrame(data)

# Crear un gráfico de dispersión para visualizar la relación
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Edad', y='VentaTotal', hue='Estado', data=df_exploracion, s=100)
plt.title('Venta Total vs. Edad por Estado de Cliente')
plt.xlabel('Edad')
plt.ylabel('Venta Total')
plt.show()
```

**Reto:** Crea un **histograma** que muestre la distribución de la `VentaTotal` y un **gráfico de caja** (*boxplot*) que visualice la distribución de las ventas para cada `Estado`.

```python
# Pista de Código para el Reto:
# Pista 1: Usa sns.histplot() o plt.hist() para el histograma.
# Pista 2: Usa sns.boxplot() para el gráfico de caja.

# Tu código aquí
```

-----

### **2. Separación *Train/Test* y Validación Cruzada**

Antes de entrenar un modelo, debemos dividir nuestros datos para evaluar su rendimiento de forma objetiva.

  * **División *Train/Test***: Separa el conjunto de datos en dos partes. El **conjunto de entrenamiento** se usa para que el modelo aprenda, y el **conjunto de prueba** se usa para evaluar su rendimiento en datos que nunca ha visto. Esto previene el sobreajuste (*overfitting*).
  * **Validación Cruzada (*Cross-Validation*)**: Es una técnica más robusta para evaluar un modelo. En lugar de una sola división, el conjunto de datos se divide en `k` particiones (*folds*). El modelo se entrena `k` veces, usando un *fold* diferente como conjunto de prueba en cada iteración. El rendimiento final es el promedio de todas las evaluaciones. Esto reduce la varianza de la evaluación.

#### **Ejercicio:**

Divide el *DataFrame* en un conjunto de entrenamiento y uno de prueba usando una proporción de 80/20.

```python
from sklearn.model_selection import train_test_split
import pandas as pd

# Datos de ejemplo
data = {'Edad': [25, 30, 45, 50, 28, 35, 60],
        'VentaTotal': [150, 200, 180, 220, 250, 190, 300],
        'Estado': ['Activo', 'Activo', 'Inactivo', 'Activo', 'Inactivo', 'Activo', 'Inactivo']}
df_exploracion = pd.DataFrame(data)
X = df_exploracion[['Edad', 'VentaTotal']] # Features (variables de entrada)
y = df_exploracion['Estado'] # Target (variable a predecir)

# Dividir los datos 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Forma del conjunto de entrenamiento (X_train):", X_train.shape)
print("Forma del conjunto de prueba (X_test):", X_test.shape)
```

**Reto:** Realiza una validación cruzada de 3 *folds* utilizando un clasificador `LogisticRegression` sobre el *DataFrame* `X` e `y` definidos en el ejercicio. Imprime el promedio de la precisión de la validación cruzada.

```python
# Pista de Código para el Reto:
# Pista 1: Importa cross_val_score y LogisticRegression.
# Pista 2: Define el modelo y luego usa cross_val_score.
# Pista 3: cross_val_score(modelo, X, y, cv=3, scoring='accuracy').

# Tu código aquí
```
