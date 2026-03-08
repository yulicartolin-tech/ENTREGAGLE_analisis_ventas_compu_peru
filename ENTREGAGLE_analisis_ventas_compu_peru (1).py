# ============================================================
#  ENTREGABLE TRABAJO FINAL
# Análisis de Datos con NumPy, Pandas y Matplotlib
# Tema: Ventas de Suministros de Cómputo en el Perú
#
# Autor: alicia mallma cartolin
# Curso: modulos y paquetes para machine learning con python
# PROFESOR: SAUL SNEIDER CHAVEZ CHICO
# Herramientas: Pandas, NumPy, Matplotlib
# ============================================================

# ===============================
# IMPORTACIÓN DE LIBRERÍAS
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# ============================================================
# 1. CREACIÓN DEL DATASET
# ============================================================

print("\n=================================================")
print("DATASET DE VENTAS DE SUMINISTROS DE COMPUTO")
print("=================================================\n")

datos = {
    "Producto": ["Laptop", "Mouse", "Teclado", "Monitor", "Impresora"],
    "Precio": [3500, 80, 150, 900, 600],
    "Cantidad": [3, 10, 7, 4, 2],
    "Ciudad": ["Lima", "Arequipa", "Lima", "Cusco", "Trujillo"]
}

# Convertir a DataFrame
df = pd.DataFrame(datos)

print("DataFrame original:\n")
print(df)


# ============================================================
# 2. MANIPULACIÓN DE DATOS CON PANDAS
# ============================================================

print("\n=================================================")
print("MANIPULACIÓN DE DATOS")
print("=================================================\n")

# Crear columna Ventas
df["Ventas"] = df["Precio"] * df["Cantidad"]

print("DataFrame con columna Ventas:\n")
print(df)


# ============================================================
# 3. ESTADÍSTICAS
# ============================================================

print("\n=================================================")
print("ESTADÍSTICAS DE VENTAS")
print("=================================================\n")

promedio = df["Ventas"].mean()
maximo = df["Ventas"].max()
minimo = df["Ventas"].min()
total = df["Ventas"].sum()

print("Promedio de ventas:", promedio)
print("Venta máxima:", maximo)
print("Venta mínima:", minimo)
print("Total de ventas:", total)


# ============================================================
# 4. FILTRADO DE DATOS
# ============================================================

print("\n=================================================")
print("FILTROS DE DATOS")
print("=================================================\n")

print("Ventas realizadas en Lima:\n")
print(df[df["Ciudad"] == "Lima"])

print("\nProductos con ventas mayores a 1000:\n")
print(df[df["Ventas"] > 1000])

print("\nProductos con cantidad mayor a 5:\n")
print(df[df["Cantidad"] > 5])


# ============================================================
# 5. CÁLCULOS CON NUMPY
# ============================================================

print("\n=================================================")
print("CÁLCULOS CON NUMPY")
print("=================================================\n")

ventas_array = np.array(df["Ventas"])

media = np.mean(ventas_array)
desviacion = np.std(ventas_array)
maximo_np = np.max(ventas_array)
minimo_np = np.min(ventas_array)

print("Media:", media)
print("Desviación estándar:", desviacion)
print("Valor máximo:", maximo_np)
print("Valor mínimo:", minimo_np)


# ============================================================
# 6. VISUALIZACIÓN DE DATOS
# ============================================================

print("\nGenerando gráficos...\n")

# Gráfico de Barras
plt.figure()
plt.bar(df["Producto"], df["Ventas"])
plt.title("Ventas por Producto")
plt.xlabel("Producto")
plt.ylabel("Ventas")
plt.show()


# Gráfico de Línea
plt.figure()
plt.plot(df["Producto"], df["Cantidad"], marker="o")
plt.title("Cantidad Vendida por Producto")
plt.xlabel("Producto")
plt.ylabel("Cantidad")
plt.show()


# Gráfico de Pastel
ventas_ciudad = df.groupby("Ciudad")["Ventas"].sum()

plt.figure()
plt.pie(ventas_ciudad, labels=ventas_ciudad.index, autopct="%1.1f%%")
plt.title("Distribución de Ventas por Ciudad")
plt.show()


# ============================================================
# 7. ANÁLISIS DE RESULTADOS
# ============================================================

print("\n=================================================")
print("ANÁLISIS DE RESULTADOS")
print("=================================================\n")

producto_top = df.loc[df["Ventas"].idxmax(), "Producto"]
ciudad_top = df.groupby("Ciudad")["Ventas"].sum().idxmax()

print("1. Producto con mayores ventas:", producto_top)
print("2. Ciudad con mayor volumen de ventas:", ciudad_top)
print("3. Promedio de ventas:", promedio)

if desviacion > 1000:
    print("4. Existe alta variación en las ventas.")
else:
    print("4. Las ventas son relativamente estables.")

print("5. Producto recomendado para promocionar:", producto_top)


# ============================================================
# 8. GENERAR DATASET MÁS GRANDE
# ============================================================

print("\n=================================================")
print("GENERACIÓN DE DATASET MÁS GRANDE")
print("=================================================\n")

productos = [
"Laptop","Mouse","Teclado","Monitor","Tablet",
"Impresora","Webcam","Audifonos","Router","Disco SSD"
]

ciudades = ["Lima","Cusco","Arequipa","Trujillo","Piura"]

dataset = {
    "Producto": [],
    "Precio": [],
    "Cantidad": [],
    "Ciudad": []
}

for i in range(30):

    dataset["Producto"].append(random.choice(productos))
    dataset["Precio"].append(random.randint(50,4000))
    dataset["Cantidad"].append(random.randint(1,15))
    dataset["Ciudad"].append(random.choice(ciudades))

df_grande = pd.DataFrame(dataset)

df_grande["Ventas"] = df_grande["Precio"] * df_grande["Cantidad"]

print("Dataset ampliado:\n")
print(df_grande)


# ============================================================
# 9. GRÁFICOS ADICIONALES
# ============================================================

# Histograma
plt.figure()
plt.hist(df_grande["Ventas"], bins=10)
plt.title("Histograma de Ventas")
plt.xlabel("Ventas")
plt.ylabel("Frecuencia")
plt.show()


# Gráfico de Dispersión
plt.figure()
plt.scatter(df_grande["Precio"], df_grande["Cantidad"])
plt.title("Relación Precio vs Cantidad")
plt.xlabel("Precio")
plt.ylabel("Cantidad")
plt.show()