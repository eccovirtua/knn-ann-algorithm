# # migrate_to_mongo.py
# import os
# import pandas as pd
# import numpy as np
# from pymongo import MongoClient
# from dotenv import load_dotenv
# from pathlib import Path

# # 1. Configuración
# load_dotenv()
# MONGO_URI = os.getenv("MONGODB_URI")
# BASE_DIR = Path(__file__).resolve().parent.parent
# VECT_DIR = BASE_DIR / "data" / "vectorized"

# # 2. Cargar datos actuales (Lo mismo que hace tu API ahora)
# print("Cargando datos locales...")
# items_df = pd.read_parquet(VECT_DIR / "items.parquet").reset_index(drop=True)
# embeds = np.load(VECT_DIR / "items_embeds.npz")["embeddings"] 

# # 3. Conexión a Mongo
# client = MongoClient(MONGO_URI)
# db = client.get_database()
# collection = db["items_col"] # Nueva colección para items + vectores

# # 4. Unir y Subir
# print(f"Migrando {len(items_df)} items a MongoDB...")
# batch = []
# BATCH_SIZE = 1000

# # Limpiar colección anterior si existe (opcional, ten cuidado en prod)
# collection.delete_many({}) 

# for index, row in items_df.iterrows():
#     # Convertir fila de Pandas a Diccionario Python estándar
#     doc = row.to_dict()
    
#     # Limpiar valores NaN/Nat de Pandas para que Mongo no falle
    
#     for k, v in doc.items():
#         # 1. Si es un array de Numpy, convertir a lista de Python (Mongo odia Numpy)
#         if isinstance(v, np.ndarray):
#             v = v.tolist()
#             doc[k] = v
        
#         # 2. Si es una lista o tupla, saltamos la verificación de NaN (no es un valor nulo)
#         if isinstance(v, (list, tuple)):
#             continue 

#         # 3. Ahora sí, es seguro verificar NaN para valores simples
#         if pd.isna(v):
#             doc[k] = None
    
        
        
            
#     # --- LA CLAVE: Agregar el vector ---
#     # Convertimos numpy array a lista de floats normal
#     vector = embeds[index].tolist()
#     doc["embedding"] = vector 
    
#     # Asegurar que itemId sea string para búsquedas rápidas
#     doc["itemId"] = str(doc["itemId"])
    
#     batch.append(doc)
    
#     if len(batch) >= BATCH_SIZE:
#         collection.insert_many(batch)
#         batch = []
#         print(f"Insertados {index} items...")

# if batch:
#     collection.insert_many(batch)

# print("¡Migración Completada! Ahora crea el índice en Atlas.")
import numpy as np
from pathlib import Path

# Asegúrate de que la ruta apunte a donde tienes tu archivo .npz
# Según tu código anterior, parece estar en data/vectorized/
path = Path("../data/vectorized/items_embeds.npz") 

try:
    data = np.load(path)
    embeds = data["embeddings"]
    print("-" * 30)
    print(f"✅ TAMAÑO TOTAL: {embeds.shape}")
    print(f"👉 TUS DIMENSIONES SON: {embeds.shape[1]}")  # <--- ESTE ES EL NÚMERO
    print("-" * 30)
except FileNotFoundError:
    print("❌ No encontré el archivo. Verifica la ruta.")