import pandas as pd
import os

# Ruta donde tienes guardados todos los archivos CSV
folder_path = '/Users/Stikets/Desktop/Axel/python/Data'

# Lista para guardar los DataFrames
all_matches = []

# Cargar cada archivo CSV
for year in range(1968, 2025):  # hasta 2024 inclusive
    filename = f"atp_matches_{year}.csv"
    full_path = os.path.join(folder_path, filename)
    print("archivo encontrado: " + filename)
    
    if os.path.exists(full_path):
        print("leyendo: " + filename)
        df = pd.read_csv(full_path)
        df['year'] = year  # para saber de qué año viene cada fila
        all_matches.append(df)
        print("leido y añadido: " + filename)
    else:
        print(f"⚠️ Archivo no encontrado: {filename}")

# Combinar todo en un solo DataFrame
print("combinando todo en uno ...")
df_all = pd.concat(all_matches, ignore_index=True)

# Guardar el DataFrame combinado en un CSV
output_path = os.path.join(folder_path, "atp_matches_1968_2024_completo.csv")
df_all.to_csv(output_path, index=False)
print("Archivo guardado en:", output_path)

# Mostrar forma y columnas para verificar
print("Número total de partidos:", df_all.shape[0])
print("Columnas:", df_all.columns.tolist())
