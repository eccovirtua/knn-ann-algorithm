# services/recs_api.py
import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from pathlib import Path
from typing import List, Tuple
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import UTC
from enum import Enum
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import logging

from starlette.middleware.cors import CORSMiddleware

logger = logging.getLogger("recs_api")
logging.basicConfig(level=logging.INFO)

#cargar el .env y conectar a mi atlasdb(mongo)xddddddddddd
load_dotenv()


MONGO_URI = os.getenv("MONGODB_URI")
if not MONGO_URI:
    raise RuntimeError("MONGODB_URI is not defined")

mongo = MongoClient(MONGO_URI)
db = mongo.get_database()             # base de datos por defecto de tu URI
feedback_col = db["feedback"]         # colección para almacenar feedback






app = FastAPI(title="Recommendation Service")

# Seguridad JWT
JWT_SECRET = "N2wwJveBGKL6f8iWIL7nx+Cl0rMoJUWpyCfsbu+7mHQ="
security = HTTPBearer() #lee el header Authorization: bearer <token>

def get_user_id_from_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    extrae user_id desde el JWT.
    Se inyecta en endpoints con `Depends(get_user_id_from_jwt)`.
    """
    token = credentials.credentials #extrae el string del token
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        # decode verifica la firma y decodifica el payload
        return payload.get("userId") or payload.get("sub")
        # Devuelve userId (o sub) del payload

    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")
        #revisar ya que esta fallando autenticación en cada nueva instancia de aplicacion






        #CARGA DE DATOS (ITEMS)

#Rutas a la carpeta que contiene items.parquet y los vectores
BASE_DIR = Path(__file__).resolve().parents[1]
VECT_DIR = BASE_DIR / "data" / "vectorized"


#Carga catálogo de ítems en un DataFrame
items_df = pd.read_parquet(VECT_DIR / "items.parquet")

#Construye un diccionario: item_id → índice (fila) en el DataFrame
movieid_to_index = {
    row["itemId"]: idx
    for idx, row in items_df.iterrows()
}

# 2. Carga embeddings
embeds = np.load(VECT_DIR / "items_embeds.npz")["embeds"]

# Construye y carga el índice Annoy para búsqueda rápida de vecinos
dim = embeds.shape[1]
ann_index = AnnoyIndex(dim, metric="angular")
ann_index.load(str(VECT_DIR / "items_index.ann"))



#  Pydantic models
class RecItem(BaseModel):
    """Representa un ítem recomendado o seed."""
    item_id: str
    title: str
    distance: float
    image_url: str | None = None

class RecommendRequest(BaseModel):
    """Payload para `/recommend`."""
    item_id: str
    top_n: int = 5

class RecommendResponse(BaseModel):
    """Respuesta de `/recommend`."""
    item_id: str
    recommendations: list[RecItem]

class FeedbackRequest(BaseModel):
    """Payload para `/feedback/{domain}`."""
    item_id: str
    feedback: int    # -1 para rechazo, +1 para aceptación

class SeedResponse(BaseModel):
    """Respuesta de `/seed/{domain}` y `/feedback/{domain}`."""
    seed_item: RecItem


# ───────────────────────────────── Helpers para Mongo ───────────────────────────────── #

def save_feedback(user_id: str, domain: str, item_id: str, feedback: int):
    """Upsert: si existe actualiza feedback+ts(timestamp), si no existe lo inserta."""
    now = datetime.now(UTC)
    feedback_col.update_one(
        {"user_id": user_id, "domain": domain, "item_id": item_id},
        {"$set": {"feedback": feedback, "ts": now}},
        upsert=True
    )
    feedback_col.create_index([("user_id", 1), ("domain", 1), ("item_id", 1)], unique=True)

def get_history(user_id: str, domain: str) -> List[Tuple[str,int]]:
    """Recupera listado ordenado de (item_id, feedback)."""
    docs = feedback_col.find({"user_id": user_id, "domain": domain}).sort("ts", 1)
    return [(d["item_id"], d["feedback"]) for d in docs]

def clear_history(user_id: str, domain: str):
    """Borra todos los feedbacks de ese user y dominio."""
    feedback_col.delete_many({"user_id": user_id, "domain": domain})





###### Endpoints ########
@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """
    Devuelve las `top_n` recomendaciones más cercanas al item_id dado,
    usando Annoy y el catálogo completo (sin filtrar por dominio).
    """


    # 1. Validar que el item exista
    if req.item_id not in movieid_to_index:
        raise HTTPException(404, f"item_id '{req.item_id}' no encontrado")

    # 2. Consultar Annoy para obtener vecinos
    idx = movieid_to_index[req.item_id]
    neigh_idxs, dists = ann_index.get_nns_by_item(
        idx, req.top_n + 1, include_distances=True
    )



    # Construir la lista de RecItem (omitimos el primero que es el mismo seed([1:])
    recs: List[RecItem] = []
    for n_idx, dist in zip(neigh_idxs[1:], dists[1:]):
        row = items_df.iloc[n_idx]
        recs.append(RecItem(
            item_id=row["itemId"],
            title=row["title"],
            distance=dist,
            image_url=row.get("image_url", None)
        ))

    return RecommendResponse(item_id=req.item_id, recommendations=recs)




#dominios permitidos, entiendase por dominio, tipos de contenido multimedia
class Domain(str, Enum):
    movie = "movie"
    book = "book"
    music = "music"




#funcion helper para evitar el codigo duplicado
def generate_new_seed(domain: str) -> RecItem:
    candidates = items_df[items_df["domain"] == domain]
    if candidates.empty:
        raise HTTPException(404, "no hay items para el dominio")
    row = candidates.sample(1).iloc[0]
    return RecItem(
        item_id=row["itemId"],
        title=row["title"],
        distance=0.0,
        image_url=row.get["image_url", None]
    )


#Decide una lógica muy simple para el primer seed (p. ej. un ítem aleatorio del dominio solicitado).
@app.get("/seed/{domain}", response_model=SeedResponse)
def get_initial_seed(
        domain: Domain,
        user_id: str = Depends(get_user_id_from_jwt)
):
    """
    Si el usuario ya tiene historial en este domain, devuelve la última seed mostrada.
    Si no tiene historial, escoge una seed aleatoria, la persiste con feedback=0 y la devuelve.
    IMPORTANTE: NO borra historial aquí (reset lo hace explícitamente).
    """
    dom = domain.value

    # Recupera historial ordenado; si existe, la última tupla corresponde a la seed más reciente
    history = get_history(user_id, dom)
    if history:
        last_item_id, _ = history[-1]  # get_history está ordenado por timestamp ascendente
        # localiza esa fila en items_df
        if last_item_id in movieid_to_index:
            row = items_df.loc[movieid_to_index[last_item_id]]
            seed = RecItem(item_id=row["itemId"], title=row["title"],
                           distance=0.0, image_url=row.get("image_url", None))
            return SeedResponse(seed_item=seed)
        else:
            # fallback: si el item del historial no se encuentra en el DF, borrar historial y crear seed nueva
            clear_history(user_id, dom)

    # No hay historial → crear seed aleatoria y guardarla con feedback 0 (neutra)
    seed = generate_new_seed(dom)
    save_feedback(user_id, dom, seed.item_id, 0)
    return SeedResponse(seed_item=seed)


@app.post("/feedback/{domain}", response_model=SeedResponse)
def handle_feedback(
        domain: Domain,
        req: FeedbackRequest,
        user_id: str = Depends(get_user_id_from_jwt)
):
    """
       Registra el feedback (positivo o negativo) para el item dado,
       luego calcula y retorna el siguiente seed según feedback y dominio.
       """

    domain_str = domain.value

    #guardar feedback real
    save_feedback(user_id, domain_str, req.item_id, req.feedback)


    # Calcula la siguiente seed
    new_seed = compute_next_seed(user_id, domain_str)
    if new_seed is None:
        # # si se agotó el dominio -> devolvemos semilla inicial limpia
        # row = items_df[items_df["domain"] == dom].sample(1).iloc[0]
        # reset = RecItem(item_id=row["itemId"], title=row["title"], distance=0.0,
        #                 image_url=row.get("image_url"))
        # return SeedResponse(seed_item=reset)
        raise HTTPException(404, "No se pudo generar nuevo ítem semilla")

    last =  feedback_col.find_one(
        {"user_id": user_id, "domain": domain_str},
        sort=[("ts", -1)]
    )
    if not last or last.get("item_id") != new_seed.item_id:
        save_feedback(user_id, domain_str, new_seed.item_id, 0)

    return SeedResponse(seed_item=new_seed)

# CORS: limitar en producción a orígenes que controles (ej. la URL de tu app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # cambiar "*" por la URL de tu cliente en producción
    allow_credentials=True,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset/{domain}", response_model=SeedResponse)
def reset_recs(
    domain: Domain,
    user_id: str = Depends(get_user_id_from_jwt)
):
    # """
    # Borra t0do el historial del user para este domain y devuelve una seed inicial limpia.
    # """
    dom = domain.value
    # Borra el historial en mongo
    clear_history(user_id, dom)

    # Escoge una semilla aleatoria del dominio
    seed = generate_new_seed(dom)
    save_feedback(user_id, dom, seed.item_id, 0)
    return SeedResponse(seed_item=seed)


def compute_next_seed(user_id: str, domain: str) -> RecItem | None:
    """
    Devuelve la siguiente semilla basada en el historial persistido (Mongo).
    - Usa el último feedback positivo (si existe) y explora sus vecinos ampliados.
    - Evita items ya mostrados (shown = positivos + negativos + neutrales).
    - Si no hay vecinos válidos, elige uno aleatorio del dominio fuera de 'shown'.
    - Retorna None si no quedan items.
    """
    # Leer el historial (get_history devuelve ordenado por ts ascendente)
    history = get_history(user_id, domain)
    logger.info("compute_next_seed user=%s domain=%s history_len=%d", user_id, domain, len(history))

    # Conjuntos/listas útiles
    shown = {item for item, _ in history if item}            # todos los mostrados (positivos, negativos y neutrales)
    positives = [item for item, fb in history if fb > 0]     # LISTA preservando orden cronológico

    # 1) Si hay positivos, tomar el último (más reciente) y explorar vecinos
    if positives:
        base = positives[-1]
        logger.info("Base positiva encontrada: %s (searching neighbors)", base)

        idx = movieid_to_index.get(base)
        if idx is not None:
            # pedir muchos vecinos para tener más opciones y evitar loops
            K = 50
            neigh_idxs, _ = ann_index.get_nns_by_item(idx, K, include_distances=True)

            # recorremos los vecinos (skip self en neigh_idxs[0]) y devolvemos el primer candidato válido
            for neigh_idx in neigh_idxs[1:]:
                row = items_df.iloc[neigh_idx]
                candidate_id = row["itemId"]
                candidate_domain = row["domain"]

                if candidate_domain == domain and candidate_id not in shown:
                    logger.info("Vecino seleccionado: %s (domain=%s)", candidate_id, candidate_domain)
                    return RecItem(
                        item_id=candidate_id,
                        title=row["title"],
                        distance=0.0,
                        image_url=row.get("image_url", None)
                    )
        else:
            logger.warning("Base positiva %s no encontrada en movieid_to_index", base)

    # 2) Si no hay positivos o no encontramos vecino válido, elegir aleatorio fuera de 'shown'
    candidates = items_df[items_df["domain"] == domain]
    candidates = candidates[~candidates["itemId"].isin(shown)]
    if not candidates.empty:
        row = candidates.sample(1).iloc[0]
        logger.info("Seed aleatoria seleccionada: %s", row["itemId"])
        return RecItem(
            item_id=row["itemId"],
            title=row["title"],
            distance=0.0,
            image_url=row.get("image_url", None)
        )

    # 3) Si ya se vieron todos los items
    logger.info("No quedan candidatos para user=%s domain=%s (todos mostrados)", user_id, domain)
    return None