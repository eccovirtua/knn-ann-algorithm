# services/recs_api.py
import os
from dotenv import load_dotenv
import re
from fastapi import Header, Response, Query
from bson import ObjectId
import sys
from firebase_admin import credentials
import firebase_admin
from firebase_admin import auth
import logging
from enum import Enum
from uuid import uuid4
from typing import List, Tuple, Optional, Dict, Any
from dotenv import load_dotenv
import math
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone
import random
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
from services import tmdb_api
from services.tmdb_api import fetch_movie_poster
from services.lastfm_api import get_album_art
from services.lastfm_api import PLACEHOLDER
from pydantic import BaseModel
from datetime import datetime, timezone


# ---------- logging ----------
logger = logging.getLogger("recs_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
logger.addHandler(handler)

# ---------- config & env ----------
load_dotenv()
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "ClaveSecreta123")
JWT_SECRET = os.getenv("JWT_SECRET", "N2wwJveBGKL6f8iWIL7nx+Cl0rMoJUWpyCfsbu+7mHQ=")
MONGO_URI = os.getenv("MONGODB_URI")
if not MONGO_URI:
    raise RuntimeError("MONGODB_URI is not defined")

SESSION_DAILY_LIMIT = 9999 # Límite diario
# ---------- app ----------
app = FastAPI(title="Recommendation Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
# monta el router con prefijo /external
app.include_router(tmdb_api.router, prefix="/external", tags=["External APIs"])
security = HTTPBearer()

# ---------- Mongo (colecciones) ----------
client = AsyncIOMotorClient(MONGO_URI)
db = client.get_database()
feedback_col = db.get_collection("feedback")
sessions_col = db.get_collection("sessions")
session_feedback_col = db.get_collection("session_feedback")
user_lists_col = db.get_collection("user_lists")
user_favorites_col = db.get_collection("user_favorites")


# índices defensivos
try:
    user_favorites_col.create_index([("user_id", 1), ("item_id", 1)], unique=True) # <-- NUEVO ÍNDICE
except Exception as e:
    logger.warning("No se pudo crear índices: %s", e)
try:
    feedback_col.create_index([("user_id", 1), ("domain", 1), ("item_id", 1)], unique=True)
    sessions_col.create_index([("session_id", 1)], unique=True)
except Exception as e:
    logger.warning("No se pudo crear índices: %s", e)

try:
    user_lists_col.create_index([("user_id", 1), ("name", 1)], unique=True)  # <-- AÑADIR ESTA
except Exception as e:
    logger.warning("No se pudo crear índices: %s", e)

try:
    sessions_col.create_index([("user_id", 1), ("created_date_utc", 1)]) # <-- AÑADIR ESTA
except Exception as e:
    logger.warning("No se pudo crear índices: %s", e)


items_col = db.get_collection("items_col") 
try:
    # Vital para buscar un item rápido por su ID (usado en detalles y recomendaciones)
    items_col.create_index("itemId", unique=True)
    
    # Vital para filtrar por 'movie', 'book', 'music' rápidamente
    items_col.create_index("domain")
    
    # Vital para tu buscador por texto (regex) en search_items
    items_col.create_index("title") 
except Exception as e:
    logger.warning("No se pudo crear índices para items_col: %s", e)

# ---------- modelos Pydantic ----------
class RecItem(BaseModel):
    item_id: str
    title: str
    distance: float
    image_url: Optional[str] = None

class FavoriteStatusResponse(BaseModel):
    item_id: str
    is_favorite: bool

class UserUsageStatus(BaseModel):
    daily_limit: int
    sessions_today: int
    remaining_today: int

class RecommendRequest(BaseModel):
    item_id: str
    top_n: int = 5
class RecommendResponse(BaseModel):
    item_id: str
    recommendations: List[RecItem]
class FeedbackRequest(BaseModel):
    item_id: str
    feedback: int  # -1 rechazo, +1 like, 0 neutro
class SeedResponse(BaseModel):
    seed_item: Optional[RecItem] = None  # permitir None cuando la sesión termina
class SessionCreateResponse(BaseModel):
    session_id: str
    seed: Optional[RecItem] = None
    is_finished: bool = False
class SessionStateResponse(BaseModel):
    session_id: str
    domain: str
    last_item: Optional[RecItem]
    iterations: int
    limit: int
    finished: bool

class SearchResultItem(BaseModel):
    item_id: str
    title: str
    domain: str # Good to know the type in search results
    image_url: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchResultItem]

class ItemDetailResponse(RecItem): # Inherit from RecItem to include basic fields
    # Add any other specific details you want to show, if available in items_df
    # For example:
    genres: Optional[List[str]] = None
    year: Optional[str] = None
    # Add 'artist' for music, 'director' for movies etc. if they exist in items_df
    # Example for music:
    artist: Optional[str] = None
    # Example specific field from your dataset (if it exists)
    google_avg_rating: Optional[int] = None
    imdb_score: Optional[float] = None
    listeners: Optional[int] = None
class FinalListResponse(BaseModel):
    recommendations: List[RecItem]
    session_avg_quality: float = 0.0
class SeedResponseWithSessionId(BaseModel):
    session_id: str
    seed_item: Optional[RecItem] = None
class TimeStats(BaseModel):
    hours_interacting: float = 0.0
    hours_from_final_recs: float = 0.0
class DomainStats(BaseModel):
    total_sessions: int = 0
    finished_sessions: int = 0
    total_items_shown: int = 0
    items_liked: int = 0
    items_rejected: int = 0
    final_recs_generated: int = 0
    avg_quality_score: float = 0.0
    time_stats: TimeStats
class UserDashboardStats(BaseModel):
    total_sessions: int
    finished_sessions: int
    total_items_interacted: int
    total_items_liked: int
    total_items_rejected: int
    total_final_recs_generated: int
    total_avg_quality_score: float = 0.0
    total_time_stats: TimeStats
    domain_stats: Dict[str, DomainStats]
class Domain(str, Enum):
    movie = "movie"
    book = "book"
    music = "music"

class ListCreateRequest(BaseModel):
    name: str
    icon_name: str
    color_hex: str

class ListUpdateRequest(BaseModel):
    icon_name: str
    color_hex: str
    name: str

class ItemAddRequest(BaseModel):
    item_id: str

# Modelo para la info BÁSICA de la lista (para mostrar en la app)
class UserListBasic(BaseModel):
    list_id: str
    name: str
    item_count: int
    icon_name: str  # SÍ está definido aquí
    color_hex: str
    is_archived: bool = False

# Modelo para la lista COMPLETA (cuando el usuario entra a verla)
class UserListDetail(UserListBasic):
    # Reutilizamos SearchResultItem para mostrar los items
    items: List[SearchResultItem]

class UserProfileRequest(BaseModel):
    age: int
    name: str = ""

# 1. Definimos el modelo de respuesta (lo que espera Android)
class UserLookupResponse(BaseModel):
    email: str

class UserCreate(BaseModel):
    username: str
    email: str
    age: int
    firebaseUid: str
    profile_picture: Optional[str] = None


class UserProfileResponse(BaseModel):
    firebaseUid: str
    name: str              # En tu DB es "name", mantengamos la consistencia
    email: str             # Útil tenerlo
    age: int
    profile_picture: Optional[str] = None
    country: Optional[str] = "International" # Valor por defecto si no existe en DB
    cover_image: Optional[str] = None
    show_age: bool = True

class UserUpdateRequest(BaseModel):
    country: Optional[str] = None
    show_age: Optional[bool] = None
    profile_picture: Optional[str] = None
    cover_image: Optional[str] = None

def _safe_float(value) -> Optional[float]:
    """Convierte de forma segura a float, o devuelve None si falla o es NaN."""
    if pd.isna(value): # Esto captura None, np.nan, etc.
        return None
    try:
        f_val = float(value)
        # Comprueba de nuevo si la conversión resultó en nan (ej. si era "nan")
        if math.isnan(f_val):
            return None
        return f_val
    except (ValueError, TypeError):
        return None

def _safe_int(value) -> Optional[int]:
    """Convierte de forma segura a int, o devuelve None si falla o es NaN."""
    if pd.isna(value): # Esto captura None, np.nan, etc.
        return None
    try:
        # Usamos float() primero para manejar valores como "5.0" y nan
        f_val = float(value)
        if math.isnan(f_val):
            return None
        return int(f_val)
    except (ValueError, TypeError):
        return None


# 1. Definimos las posibles rutas donde podría estar el archivo
# - Opción A: En local (raíz del proyecto)
# - Opción B: En Cloud Run (montado desde Secret Manager)
POSSIBLE_PATHS = [
    "./recommender-ec605-firebase-adminsdk-fbsvc-3249973bce.json", # Local
    "/secrets/firebase-creds"  # Cloud Run (Ruta absoluta del montaje)
]

cred_path = None

# Buscamos dónde existe el archivo
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        cred_path = path
        break

try:
    firebase_admin.get_app()
except ValueError:
    if cred_path:
        # Si encontramos el archivo (Local o Nube), lo usamos
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        print(f"Firebase inicializado usando: {cred_path}")
    else:
        # Si no hay archivo, intentamos la identidad por defecto (puede que falle si hay conflicto de proyectos)
        print("No se encontró JSON de credenciales. Usando Default Credentials.")
        firebase_admin.initialize_app()

# Tu función validadora
async def get_outsystems_user(
    x_api_key: str = Header(...),
    x_user_id: str = Header(...)
) -> str:
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Contraseña incorrecta")
    
    if not x_user_id:
        raise HTTPException(status_code=400, detail="Falta el usuario")
        
    return x_user_id

async def _get_list_and_map_to_basic(list_id: str) -> UserListBasic:
    """Helper para buscar una lista por ID y devolver el modelo básico."""
    updated_doc = await user_lists_col.find_one({"_id": ObjectId(list_id)})

    if not updated_doc:
        # Esto es un fallback por seguridad
        raise HTTPException(status_code=404, detail="Lista no encontrada tras actualización")

    return UserListBasic(
        list_id=str(updated_doc["_id"]),
        name=updated_doc.get("name"),
        icon_name=updated_doc.get("icon_name", "default"),
        color_hex=updated_doc.get("color_hex", "#FFFFFF"),
        item_count=len(updated_doc.get("items", []))
    )

async def get_current_user_uid(auth_creds: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verifica el ID Token enviado desde Android usando Firebase Auth.
    """
    token = auth_creds.credentials
    try:
        # Firebase valida la firma, expiración y emisor por ti
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token['uid']
        # El 'uid' es el identificador único y persistente del usuario en Firebase
        return uid

    except Exception as err:
        logger.error(f"Error de autenticación Firebase: {err}")
        raise HTTPException(
            status_code=401,
            detail="Token de Firebase inválido o expirado"
        )

async def _set_list_archive_status(list_id: str, user_id: str, archive: bool) -> UserListBasic:
    """Helper para cambiar el estado de archivo de una lista."""
    try:
        result = await user_lists_col.update_one(
            {"_id": ObjectId(list_id), "user_id": user_id},
            {"$set": {"is_archived": archive}}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="ID de lista inválido")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Lista no encontrada o no pertenece al usuario")

    # Reutilizamos el helper de mapeo que creamos en la pregunta anterior
    return await _get_list_and_map_to_basic(list_id)

def _col(df: pd.Series, name: str, default):
    try:
        return df.get(name, default)
    except(ValueError, TypeError):
        return default

# ---------- Mongo helpers (global feedback) ----------
def save_feedback(user_id: str, domain: str, item_id: str, feedback: int):
    """Upsert simple: guarda último feedback y timestamp (persistente)."""
    now = datetime.now(timezone.utc)
    feedback_col.update_one(
        {"user_id": user_id, "domain": domain, "item_id": item_id},
        {"$set": {"feedback": feedback, "ts": now}},
        upsert=True,
    )
async def get_history(user_id: str, domain: str) -> List[Tuple[str, int]]:
    # Motor: cursor.to_list(length=None) para traer todo
    cursor = feedback_col.find({"user_id": user_id, "domain": domain}).sort("ts", 1)
    docs = await cursor.to_list(length=None)
    return [(d["item_id"], d["feedback"]) for d in docs]

def clear_history(user_id: str, domain: str):
    feedback_col.delete_many({"user_id": user_id, "domain": domain})

# ---------- Sessions helpers ----------
SESSION_ITER_LIMIT = 10
K_VECINOS = 50
CONSIDER_LAST_POSITIVES = 3
EXPLORATION_SAMPLE = 200
TARGET_FINAL_N = 20
DIVERSITY_JACCARD_THRESHOLD = 0.60
ALPHA_SIM = 0.60    # similitud (desde Annoy / distancia)
BETA_POP = 0.30     # popularidad / base_score
GAMMA_IMDB = 0.10
DELTA_NOVELTY = 0.40

async def row_to_recitem(doc:dict, distance: float = 0.0) -> RecItem:
    item_id = doc.get("item_id") or doc.get("itemId")
    domain = doc.get("domain")
    title = doc.get("title", "")
    image_url = doc.get("image_url") # Leemos la URL del dataset

    if domain == "music" and image_url == "PLACEHOLDER": # Asumiendo que era un string
        image_url = None

    # --- Movies (TMDB) ---
    if domain == "movie" and not image_url:
        title = doc.get("title", "")
        clean_title = title.split("(")[0].strip()
        
        # ¡AQUÍ ESTÁ LA MAGIA! Borramos el try/except de asyncio
        # y simplemente le decimos que "espere" (await) el resultado.
        image_url = await fetch_movie_poster(clean_title)

    # --- Music (Last.fm) ---
    elif domain == "music" and not image_url:
        artist = doc.get("artist", "")
        track = doc.get("title", "")
        item_id = doc.get("item_id") or doc.get("itemId") or ""
        
        if not artist and item_id.startswith("lf-") and "_" in item_id:
            try:
                parts = item_id.replace("lf-", "", 1).split("_", 1)
                artist = parts[0].strip()
                track = parts[1].strip()
            except Exception as err:
                print(f"⚠️ Error extrayendo artista/track desde item_id: {err}")
                
        # 🧩 Fallback: intentar dividir el título por guion
        if not artist and "-" in track:
            parts = track.split("-", 1)
            artist = parts[0].strip()
            track = parts[1].strip()
            
        # 🧩 Consultar imagen del álbum en Last.fm
        if artist and track:
            try:
                # Nota: Si tu función 'get_album_art' también está definida con 'async def'
                # en tu código, deberías ponerle un 'await' aquí (await get_album_art(...)).
                # Si es una función normal (def), déjala tal cual está aquí:
                image_url = get_album_art(artist, track)
            except Exception as err:
                print(f"⚠️ Error obteniendo imagen de Last.fm: {err}")
                image_url = None
                
    # --- Fallback general ---
    if not image_url:
        image_url = "https://placehold.co/300x450?text=No+Image"

    return RecItem(
        item_id=item_id,
        title=title,
        distance=distance,
        image_url=image_url
    )

def _genres_to_set(genres):
    """Normaliza la columna genres (lista o string 'a|b')."""
    if genres is None:
        return set()
    if isinstance(genres, list):
        return {str(g).strip().lower() for g in genres if str(g).strip()}
    if isinstance(genres, str):
        return {g.strip().lower() for g in genres.split("|") if g.strip()}
    return set()

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni > 0 else 0.0

def _get_popularity(row: pd.Series) -> float:
    # popularidad general (fallbacks robustos)
    if "popularity" in row and row["popularity"] is not None:
        try:
            return float(row["popularity"])
        except (ValueError, TypeError):  # Capturamos solo los errores de conversión de tipo
            pass
    # respaldos por dominio
    if "base_score" in row and row["base_score"] is not None:
        try:
            return float(row["base_score"])
        except (ValueError, TypeError):
            pass
    if "imdb_score" in row and row["imdb_score"] is not None:
        return float(row["imdb_score"]) / 10.0
    return 0.0

def _rating(row: pd.Series) -> float:
    # rating por dominio (robusto)
    for k in ["imdb_score", "rating", "google_rating"]:
        if k in row and row[k] is not None:
            try:
                val = float(row[k])
                # normalizar imdb a 0..5 si viene en 0..10
                if k == "imdb_score":
                    return val / 2.0
                return val
            except(ValueError, TypeError):
                continue
    return 0.0

def _rating_count(row: pd.Series) -> int:
    for k in ["rating_count", "votes", "vote_count"]:
        if k in row and row[k] is not None:
            try:
                return int(row[k])
            except(ValueError, TypeError):
                continue
    return 0

# Core algorithm
async def generate_new_seed(domain: str) -> RecItem:
    pipeline = [
        {"$match": {"domain": domain}},
        {"$sample": {"size": 1}}
    ]
    # Motor: aggregate devuelve un cursor asíncrono
    cursor = items_col.aggregate(pipeline)
    results = await cursor.to_list(length=1)

    if not results:
        raise HTTPException(404, "No hay items para el dominio")

    return await row_to_recitem(results[0], distance=0.0)


async def compute_next_seed(domain: str, user_id: str = None, session_history: List[Tuple[str, int]] = None) -> \
Optional[RecItem]:
    """
    Calcula el siguiente item semilla.
    Puede usar un historial pasado explícitamente O buscarlo en BD usando user_id.
    """
    history = session_history

    # Si no nos dan historial, pero sí usuario, lo buscamos
    if history is None and user_id:
        history = await get_history(user_id, domain)

    if not history:
        history = []

    shown = {item for item, _ in history}

    # 1. Buscar el último ítem positivo (Like)
    last_positive_id = None
    for item_id, feedback in reversed(history):
        if feedback > 0:
            last_positive_id = item_id
            break

    query_vector = None

    # 2. Obtener vector del último like (si existe)
    if last_positive_id:
        seed_doc = await items_col.find_one({"itemId": last_positive_id}, {"embedding": 1})
        if seed_doc and "embedding" in seed_doc:
            query_vector = seed_doc["embedding"]

    # 3. Construir Pipeline
    pipeline = []

    if query_vector:
        # A) Búsqueda Vectorial
        pipeline.append({
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 50,
                "limit": 10
            }
        })
    else:
        # B) Random del dominio (Cold start o sin likes recientes)
        pipeline.append({"$match": {"domain": domain}})
        pipeline.append({"$sample": {"size": 10}})

    # 4. Filtrar vistos y dominio
    pipeline.append({
        "$match": {
            "domain": domain,
            "itemId": {"$nin": list(shown)}
        }
    })

    pipeline.append({"$project": {"embedding": 0}})
    pipeline.append({"$limit": 1})

    cursor = items_col.aggregate(pipeline)
    results = await cursor.to_list(length=1)

    if results:
        return await row_to_recitem(results[0], distance=0.0)

    return None

def _calculate_quality_score(doc: dict, domain: str) -> float:
    """Calcula el score de calidad basado en el dominio."""
    score = 0.0
    try:
        if domain == "movie":
            # Normalizar de 1-10 a 0-5
            # float() lanzará ValueError si el string no es un número
            score = float(doc.get("imdb_score") or 0) / 2.0
        elif domain == "book":
            # Google books suele ser 1-5
            score = float(doc.get("google_avg_rating") or 0)
        elif domain == "music":
            # Logaritmo de listeners para normalizar popularidad
            listeners = float(doc.get("listeners") or 1)
            # Aseguramos que listeners sea al menos 1 para evitar log10(0) o error matemático
            if listeners < 1:
                listeners = 1.0
            score = min(5.0, math.log10(listeners) / 1.6)

    except (ValueError, TypeError):
        score = 0.0

    return round(score, 2)


async def build_final_grid(session_id: str, user_id: str, domain: str,
                           history: List[Tuple[str, int]], target_n: int = 20) -> List[RecItem]:
    shown_ids = [iid for iid, _ in history]
    # Filtramos solo los IDs que tuvieron feedback positivo
    positive_ids = [iid for iid, fb in history if fb > 0]

    final_docs = []

    # 1. Recomendaciones Vectoriales (Si hay likes previos)
    if positive_ids:
        last_like = positive_ids[-1]
        seed_doc = await items_col.find_one({"itemId": last_like}, {"embedding": 1})

        if seed_doc and "embedding" in seed_doc:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": seed_doc["embedding"],
                        "numCandidates": 100,  # Aumentado para tener variedad
                        "limit": 12  # Intentamos llenar un 60% con recomendaciones inteligentes
                    }
                },
                {"$match": {"domain": domain, "itemId": {"$nin": shown_ids}}},
                {"$project": {"embedding": 0}}
            ]
            cursor = items_col.aggregate(pipeline)
            final_docs.extend(await cursor.to_list(length=None))

    # 2. Relleno "Joyas Ocultas" / Random
    # Calculamos cuántos faltan para llegar a target_n
    current_ids = [d["itemId"] for d in final_docs] + shown_ids
    needed = target_n - len(final_docs)

    if needed > 0:
        fill_pipeline = [
            {"$match": {"domain": domain, "itemId": {"$nin": current_ids}}},
            {"$sample": {"size": needed}},
            {"$project": {"embedding": 0}}
        ]
        cursor = items_col.aggregate(fill_pipeline)
        final_docs.extend(await cursor.to_list(length=None))

    # 3. Procesar resultados finales
    # Recortar si nos pasamos y mezclar para que no salgan ordenados por similitud exacta siempre
    final_docs = final_docs[:target_n]
    random.shuffle(final_docs)

    final_rec_items = []
    final_items_to_save = []

    for doc in final_docs:
        # Convertir a objeto Pydantic
        rec_item = await row_to_recitem(doc)

        # Calcular Score usando la función auxiliar
        score = _calculate_quality_score(doc, domain)

        # Preparar datos para guardar
        item_data = rec_item.model_dump()
        item_data["quality_score"] = score

        final_rec_items.append(rec_item)
        final_items_to_save.append(item_data)

    # Calcular promedio de la sesión
    avg_quality = 0.0
    if final_items_to_save:
        avg_quality = sum(x["quality_score"] for x in final_items_to_save) / len(final_items_to_save)

    # 4. Guardar en Mongo (Update asíncrono)
    await sessions_col.update_one(
        {"session_id": session_id, "user_id": user_id},
        {"$set": {
            "final_grid": final_items_to_save,
            "session_avg_quality_score": round(avg_quality, 4)
        }}
    )

    return final_rec_items
# Constantes para estimación de tiempo en HORAS
TIME_ESTIMATES = {
    "movie": 1.75,  # 1 h 45 m en promedio por película
    "book": 6.0,  # 6h en promedio por libro
    "music": 0.058,  # 3.5 minutos en promedio por canción
    "interaction_seconds": 30 / 3600  # 30 segundos por interacción, convertido a horas
}
# --- HELPER NECESARIO ---
async def get_session(session_id: str) -> dict:
    """Helper asíncrono para buscar sesión."""
    return await sessions_col.find_one({"session_id": session_id})

async def get_user_dashboard_stats(user_id: str) -> UserDashboardStats:
    # 1. Pipeline de Agregación (El pipeline JSON se mantiene idéntico)
    pipeline = [
        {'$match': {'user_id': user_id}},
        {
            '$addFields': {
                'history_processed': {
                    '$map': {
                        'input': {'$ifNull': ['$history', []]},
                        'as': 'item',
                        'in': {
                            'feedback_value': {
                                '$let': {
                                    'vars': {'feedback_target': '$$item.1'},
                                    'in': {
                                        '$cond': [
                                            {'$eq': [{'$type': '$$feedback_target'}, 'object']},
                                            {'$toInt': {'$let': {
                                                'vars': {'feedback_obj': {'$arrayElemAt': [{'$objectToArray': '$$feedback_target'}, 0]}},
                                                'in': '$$feedback_obj.v'
                                            }}},
                                            {'$cond': [
                                                {'$in': [{'$type': '$$feedback_target'}, ['array', 'null', 'undefined']]},
                                                0,
                                                {'$toInt': '$$feedback_target'}
                                            ]}
                                        ]
                                    }
                                }
                            },
                            'item_id': {'$arrayElemAt': ['$$item', 0]}
                        }
                    }
                }
            }
        },
        {
            '$addFields': {
                'avg_grid_score': {
                    '$cond': [
                        {'$and': ['$finished', {'$gt': [{'$size': {'$ifNull': ['$final_grid', []]}}, 0]}]},
                        {'$avg': '$final_grid.quality_score'},
                        None
                    ]
                }
            }
        },
        {
            '$group': {
                '_id': '$domain',
                'total_sessions': {'$sum': 1},
                'finished_sessions': {'$sum': {'$cond': ['$finished', 1, 0]}},
                'total_items_shown': {'$sum': {'$size': {'$ifNull': ['$history_processed', []]}}},
                'items_liked': {
                    '$sum': {
                        '$size': {
                            '$filter': {
                                'input': {'$ifNull': ['$history_processed', []]},
                                'as': 'item',
                                'cond': {'$eq': ['$$item.feedback_value', 1]}
                            }
                        }
                    }
                },
                'items_rejected': {
                    '$sum': {
                        '$size': {
                            '$filter': {
                                'input': {'$ifNull': ['$history_processed', []]},
                                'as': 'item',
                                'cond': {'$eq': ['$$item.feedback_value', -1]}
                            }
                        }
                    }
                },
                'final_recs_generated': {
                    '$sum': {
                        '$cond': ['$finished', {'$size': {'$ifNull': ['$final_grid', []]}}, 0]
                    }
                },
                'sum_of_avg_scores': {'$sum': '$avg_grid_score'},
                'sessions_with_scores': {'$sum': {'$cond': ['$avg_grid_score', 1, 0]}}
            }
        }
    ]

    # --- CAMBIO MOTOR: Ejecución asíncrona del pipeline ---
    cursor = sessions_col.aggregate(pipeline)
    results = await cursor.to_list(length=None)

    # El procesamiento de datos en Python se mantiene igual (CPU bound)
    domain_stats_map: Dict[str, DomainStats] = {
        "movie": DomainStats(time_stats=TimeStats()),
        "book": DomainStats(time_stats=TimeStats()),
        "music": DomainStats(time_stats=TimeStats())
    }

    total_sum_scores = 0.0
    total_sessions_with_scores = 0

    total_stats = {
        'total_sessions': 0, 'finished_sessions': 0, 'total_items_interacted': 0,
        'total_items_liked': 0, 'total_items_rejected': 0, 'total_final_recs_generated': 0,
        'total_hours_interacting': 0.0, 'total_hours_from_final_recs': 0.0
    }

    for doc in results:
        domain = doc['_id']
        if domain in domain_stats_map:
            hours_interacting = doc.get('total_items_shown', 0) * TIME_ESTIMATES['interaction_seconds']
            hours_from_recs = doc.get('final_recs_generated', 0) * TIME_ESTIMATES.get(domain, 0)

            sum_of_avg_scores = doc.get('sum_of_avg_scores')
            if sum_of_avg_scores is None or not math.isfinite(sum_of_avg_scores):
                sum_of_avg_scores = 0.0

            sessions_with_scores = doc.get('sessions_with_scores', 0)

            avg_score = 0.0
            if sessions_with_scores > 0:
                avg_score = sum_of_avg_scores / sessions_with_scores

            total_sum_scores += sum_of_avg_scores
            total_sessions_with_scores += sessions_with_scores

            domain_stats_map[domain] = DomainStats(
                total_sessions=doc.get('total_sessions', 0),
                finished_sessions=doc.get('finished_sessions', 0),
                total_items_shown=doc.get('total_items_shown', 0),
                items_liked=doc.get('items_liked', 0),
                items_rejected=doc.get('items_rejected', 0),
                final_recs_generated=doc.get('final_recs_generated', 0),
                avg_quality_score=round(avg_score, 2),
                time_stats=TimeStats(
                    hours_interacting=round(hours_interacting, 2),
                    hours_from_final_recs=round(hours_from_recs, 2)
                )
            )

    for stats in domain_stats_map.values():
        total_stats['total_sessions'] += stats.total_sessions
        total_stats['finished_sessions'] += stats.finished_sessions
        total_stats['total_items_interacted'] += stats.total_items_shown
        total_stats['total_items_liked'] += stats.items_liked
        total_stats['total_items_rejected'] += stats.items_rejected
        total_stats['total_final_recs_generated'] += stats.final_recs_generated
        total_stats['total_hours_interacting'] += stats.time_stats.hours_interacting
        total_stats['total_hours_from_final_recs'] += stats.time_stats.hours_from_final_recs

    total_avg_quality_score = 0.0
    if total_sessions_with_scores > 0:
        total_avg_quality_score = total_sum_scores / total_sessions_with_scores

    return UserDashboardStats(
        total_sessions=total_stats['total_sessions'],
        finished_sessions=total_stats['finished_sessions'],
        total_items_interacted=total_stats['total_items_interacted'],
        total_items_liked=total_stats['total_items_liked'],
        total_items_rejected=total_stats['total_items_rejected'],
        total_final_recs_generated=total_stats['total_final_recs_generated'],
        total_avg_quality_score=round(total_avg_quality_score, 2),
        total_time_stats=TimeStats(
            hours_interacting=round(total_stats['total_hours_interacting'], 2),
            hours_from_final_recs=round(total_stats['total_hours_from_final_recs'], 2)
        ),
        domain_stats=domain_stats_map
    )

@app.get("/stats/dashboard", response_model=UserDashboardStats)
def api_get_user_dashboard_stats(user_id: str = Depends(get_current_user_uid)):
    return get_user_dashboard_stats(user_id)

# ---------- Session endpoints (nuevo flujo) ----------
async def create_session(user_id: str, domain: str) -> Tuple[str, Optional[RecItem], bool]:
    now = datetime.now(timezone.utc)
    today_utc_str = now.strftime('%Y-%m-%d')

    # 1. Buscar sesión persistente
    existing_session = await sessions_col.find_one({
        "user_id": user_id,
        "domain": domain,
        "created_date_utc": today_utc_str,
        "reset": {"$ne": True}
    }, sort=[("created_at", -1)])

    if existing_session:
        is_finished = existing_session.get("finished", False)
        session_id = existing_session["session_id"]
        seed = None

        if not is_finished:
            last_item_id = existing_session.get("last_item_id")
            if last_item_id:
                doc = await items_col.find_one({"itemId": last_item_id})
                if doc:
                    seed = await row_to_recitem(doc, distance=0.0)
            
            # --- RED DE SEGURIDAD ---
            # Si la sesión no ha terminado pero no pudimos recuperar el ítem,
            # generamos uno nuevo para no romper la interfaz.
            if not seed:
                seed = await generate_new_seed(domain)
                await sessions_col.update_one(
                    {"session_id": session_id},
                    {"$set": {"last_item_id": seed.item_id}}
                )
        
        return session_id, seed, is_finished

    # =========================================================
    # ¡AQUÍ ESTÁ LA PARTE 2 QUE SE TE HABÍA BORRADO!
    # Si no hay sesión (Usuario 99), tenemos que crearla:
    # =========================================================
    
    daily_count = await sessions_col.count_documents({
        "user_id": user_id,
        "created_date_utc": today_utc_str
    })
    
    if daily_count >= SESSION_DAILY_LIMIT: # Asegúrate de tener importado SESSION_DAILY_LIMIT
        raise HTTPException(status_code=429, detail="Límite diario alcanzado")

    session_id = str(uuid4())
    seed = await generate_new_seed(domain)

    await sessions_col.insert_one({
        "session_id": session_id,
        "user_id": user_id,
        "domain": domain,
        "created_at": now,
        "created_date_utc": today_utc_str,
        "last_item_id": seed.item_id,
        "iterations": 0,
        "limit": SESSION_ITER_LIMIT,
        "finished": False,
        "reset": False,
        "history": [(seed.item_id, 0)],
        "shown": [seed.item_id]
    })
    
    await session_feedback_col.insert_one({
        "session_id": session_id,
        "item_id": seed.item_id,
        "feedback": 0,
        "ts": now
    })
    
    return session_id, seed, False

@app.get("/user/final-grid/{domain}", response_model=FinalListResponse)
async def get_final_grid_for_domain(domain: str, user_id: str = Depends(get_outsystems_user)):
    cursor = sessions_col.find(
        {"user_id": user_id, "domain": domain, "finished": True}
    ).sort("created_at", -1).limit(1)

    # En motor para sacar solo uno de un cursor ordenado:
    results = await cursor.to_list(length=1)
    session = results[0] if results else None

    if not session or "final_grid" not in session:
        raise HTTPException(404, "No hay grid final para este usuario y dominio")

    recs = [RecItem(**item) for item in session["final_grid"]]
    return FinalListResponse(recommendations=recs)

@app.post("/session/{domain}/create", response_model=SessionCreateResponse)
async def api_create_session(domain: Domain, user_id: str = Depends(get_outsystems_user)):
    dom = domain.value
    session_id, seed, is_finished = await create_session(user_id, dom)
    return SessionCreateResponse(
        session_id=session_id, 
        seed=seed, 
        is_finished=is_finished
    )


@app.get("/session/{session_id}", response_model=SessionStateResponse)
async def api_get_session(session_id: str, user_id: str = Depends(get_outsystems_user)):
    # --- CAMBIO MOTOR: Await get_session ---
    s = await get_session(session_id)
    if not s or s["user_id"] != user_id:
        raise HTTPException(404, "Session not found or unauthorized")

    last_item = None
    last_item_id = s.get("last_item_id")
    if last_item_id:
        # --- CAMBIO MOTOR: Await find_one ---
        doc = await items_col.find_one({"itemId": last_item_id})
        if doc:
            last_item = await row_to_recitem(doc, distance=0.0)

    iterations = int(s.get("iterations", len(s.get("shown", [])) or 0))
    limit = int(s.get("limit", SESSION_ITER_LIMIT))
    finished = bool(s.get("finished", False) or (iterations >= limit))

    return SessionStateResponse(
        session_id=session_id,
        domain=s["domain"],
        last_item=last_item,
        iterations=iterations,
        limit=limit,
        finished=finished
    )

@app.post("/session/{session_id}/feedback", response_model=SeedResponse)
async def api_session_feedback(session_id: str, req: FeedbackRequest, user_id: str = Depends(get_outsystems_user)):
    # --- CAMBIO MOTOR: Await get_session ---
    s = await get_session(session_id)
    if not s or s["user_id"] != user_id:
        raise HTTPException(404, "Session not found or unauthorized")

    # --- CAMBIO MOTOR: Await find_one ---
    item_doc = await items_col.find_one({"itemId": req.item_id})
    if not item_doc:
        raise HTTPException(404, "Item no encontrado")

    history = s.get("history", [])
    limit = int(s.get("limit", SESSION_ITER_LIMIT))
    shown: List[str] = list(s.get("shown", []))
    domain = s["domain"]

    if str(item_doc.get("domain")) != domain:
        raise HTTPException(400, "El item no corresponde al dominio de la sesión")

    if not history or history[-1][0] != req.item_id or history[-1][1] != req.feedback:
        history.append((req.item_id, req.feedback))

    if not shown or shown[-1] != req.item_id:
        shown.append(req.item_id)

    iterations = len(shown)
    finished = iterations >= limit

    if finished:
        # --- CAMBIO MOTOR: Await update_one ---
        await sessions_col.update_one(
            {"session_id": session_id},
            {"$set": {
                "history": history, "shown": shown,
                "iterations": iterations, "finished": True
            }}
        )
        return SeedResponse(seed_item=None)

    # Si no ha terminado, calculamos el siguiente item
    # --- IMPORTANTE: Usamos la función refactorizada con historial explícito ---
    next_item = await compute_next_seed(
        domain=domain,
        session_history=history  # Pasamos el historial para evitar otra query
    )

    if not next_item:
        # Fallback si no hay next item
        raise HTTPException(404, "No se pudo generar recomendación")

    # Actualizamos DB
    await sessions_col.update_one(
        {"session_id": session_id},
        {"$set": {
            "history": history,
            "shown": shown,
            "last_item_id": next_item.item_id,
            "iterations": iterations
        }}
    )

    # Guardamos feedback individual
    await session_feedback_col.insert_one({
        "session_id": session_id,
        "item_id": req.item_id,
        "feedback": req.feedback,
        "ts": datetime.now(timezone.utc)
    })

    return SeedResponse(seed_item=next_item)

async def get_session_history(session_id: str) -> List[Tuple[str, int]]:
    # --- CAMBIO MOTOR: Await get_session ---
    s = await get_session(session_id)
    if not s:
        return []
    history = []
    for x in s.get("history", []):
        try:
            item_id = str(x[0])
            feedback = int(x[1])
            history.append((item_id, feedback))
        except (IndexError, ValueError, TypeError):
            continue
    return history

@app.post("/session/{session_id}/reset", response_model=SeedResponseWithSessionId)
async def api_session_reset(session_id: str, user_id: str = Depends(get_outsystems_user)):
    s = await get_session(session_id)
    if not s or s["user_id"] != user_id:
        raise HTTPException(404, "Session not found or unauthorized")

    new_session_id = str(uuid4())
    # Marcamos la sesión vieja como terminada y reseteada
    await reset_session(session_id)
    
    # Generamos la semilla para la nueva sesión
    seed = await generate_new_seed(s["domain"])

    # Insertamos la nueva sesión limpia
    await sessions_col.insert_one({
        "session_id": new_session_id,
        "user_id": user_id,
        "domain": s["domain"],
        "iterations": 0,
        "finished": False,
        "reset": False,  # <-- Añadido para mantener coherencia con la nueva lógica
        "history": [(seed.item_id, 0)],
        "shown": [seed.item_id],
        "final_grid": None,
        "last_item_id": seed.item_id,
        "created_at": datetime.now(timezone.utc)
    })

    await session_feedback_col.insert_one({
        "session_id": new_session_id,
        "item_id": seed.item_id,
        "feedback": 0,
        "ts": datetime.now(timezone.utc)
    })

    return SeedResponseWithSessionId(session_id=new_session_id, seed_item=seed)

async def reset_session(session_id: str):
    """Marca una sesión como terminada o reseteada en BD."""
    # Asumimos que "resetear" significa marcar la vieja como terminada
    # para que no interfiera, o simplemente dejarla abandonada.
    await sessions_col.update_one(
        {"session_id": session_id},
        {"$set": {"finished": True, "reset": True}}
    )


async def search_items(query: str, limit: int = 20) -> List[SearchResultItem]:
    if not query or len(query) < 2:
        return []

    # Motor: find devuelve un cursor
    cursor = items_col.find(
        {"title": {"$regex": query, "$options": "i"}},
        {"embedding": 0}
    ).limit(limit)

    # Motor: to_list para ejecutar la query
    docs = await cursor.to_list(length=limit)

    results = []
    for doc in docs:
        rec_item = row_to_recitem(doc)
        results.append(SearchResultItem(
            item_id=rec_item.item_id,
            title=rec_item.title,
            domain=doc.get("domain", "unknown"),
            image_url=rec_item.image_url
        ))
    return results


async def get_item_details(item_id: str) -> Optional[ItemDetailResponse]:
    # Motor: await find_one
    doc = await items_col.find_one({"itemId": item_id})
    if not doc:
        return None

    rec_item = row_to_recitem(doc, distance=0.0)

    genres_list = None
    genres_data = doc.get("genres")
    if isinstance(genres_data, str):
        genres_list = [g.strip() for g in genres_data.split('|') if g.strip()]
    elif isinstance(genres_data, list):
        genres_list = [str(g).strip() for g in genres_data if str(g).strip()]

    year_match = re.search(r"\((\d{4})\)", doc.get('title', ''))
    year = year_match.group(1) if year_match else doc.get("year_str")

    return ItemDetailResponse(
        item_id=rec_item.item_id,
        title=rec_item.title,
        distance=rec_item.distance,
        image_url=rec_item.image_url,
        genres=genres_list,
        year=year,
        artist=doc.get("artist"),
        google_avg_rating=_safe_int(doc.get("google_avg_rating")),
        imdb_score=_safe_float(doc.get("imdb_score")),
        listeners=_safe_int(doc.get("listeners"))
    )


@app.get("/session/{session_id}/final-grid", response_model=FinalListResponse)
async def api_get_final_grid(session_id: str, user_id: str = Depends(get_outsystems_user)):
    s = await get_session(session_id)
    if not s or s["user_id"] != user_id:
        raise HTTPException(404, "Session not found or unauthorized")

    if not bool(s.get("finished", False)) and int(s.get("iterations", 0)) < int(s.get("limit", SESSION_ITER_LIMIT)):
        raise HTTPException(400, "Session not finished yet")

    if "final_grid" in s and s["final_grid"]:
        return FinalListResponse(recommendations=[RecItem(**i) for i in s["final_grid"]])

    # build_final_grid ahora es ASYNC (ver refactor anterior)
    final_items = await build_final_grid(
        session_id=session_id,
        user_id=user_id,
        domain=s["domain"],
        history=s.get("history", []),
        target_n=TARGET_FINAL_N
    )

    return FinalListResponse(recommendations=final_items)

@app.post("/session/{session_id}/finalize", response_model=FinalListResponse)
async def api_session_finalize(session_id: str, user_id: str = Depends(get_outsystems_user)):
    # 1. Validar sesión
    s = await get_session(session_id)
    if not s or s["user_id"] != user_id:
        raise HTTPException(404, "Session not found or unauthorized")

    # 2. Marcar finished
    await sessions_col.update_one(
        {"session_id": session_id},
        {"$set": {"finished": True}}
    )

    # 3. Obtener grid (Genera recomendaciones y calcula scores internamente)
    final_response: FinalListResponse = await api_get_final_grid(session_id=session_id, user_id=user_id)

    # 4. Obtener el score actualizado
    # Necesitamos consultar de nuevo porque api_get_final_grid actualizó el documento en segundo plano
    session_doc = await sessions_col.find_one({"session_id": session_id})
    if not session_doc:
        raise HTTPException(500, "Session data missing after finalization.")

    session_avg_quality = session_doc.get("session_avg_quality_score", 0.0)

    # 5. Combinar respuesta
    response_data = final_response.model_dump()
    response_data["session_avg_quality"] = session_avg_quality

    return FinalListResponse(**response_data)

@app.get("/search", response_model=SearchResponse)
async def api_search_items(query: str, limit: int = 20):
    if len(query) < 3:
        raise HTTPException(status_code=400, detail="Query must be at least 3 characters long")

    # search_items es async ahora
    results = await search_items(query, limit)
    return SearchResponse(results=results)

# ----------------------------------------
# ENDPOINTS DE LISTAS DE USUARIO
# ----------------------------------------

@app.post("/lists", response_model=UserListBasic)
async def api_create_list(req: ListCreateRequest, user_id: str = Depends(get_current_user_uid)):
    now = datetime.now(timezone.utc)

    if not req.name or len(req.name) < 1:
        raise HTTPException(status_code=400, detail="El nombre de la lista no puede estar vacío")

    new_list = {
        "user_id": user_id,
        "name": req.name,
        "icon_name": req.icon_name,
        "color_hex": req.color_hex,
        "created_at": now,
        "items": []
    }
    try:
        # Motor: await insert_one
        result = await user_lists_col.insert_one(new_list)
        return UserListBasic(
            list_id=str(result.inserted_id),
            name=req.name,
            item_count=0,
            icon_name=req.icon_name,
            color_hex=req.color_hex
        )
    except Exception as err:
        logger.error(f"Error al crear lista: {err}")
        if "E11000" in str(err):
            raise HTTPException(status_code=400, detail="Ya existe una lista con ese nombre")
        raise HTTPException(status_code=500, detail="Error interno al crear la lista")


@app.get("/lists", response_model=List[UserListBasic])
async def api_get_my_lists(
        archived: Optional[bool] = Query(None),
        user_id: str = Depends(get_current_user_uid)
):
    query: Dict[str, Any] = {"user_id": user_id}

    if archived:
        query["is_archived"] = True
    elif archived is False:
        query["$or"] = [
            {"is_archived": False},
            {"is_archived": {"$exists": False}}
        ]
    else:
        query["$or"] = [
            {"is_archived": False},
            {"is_archived": {"$exists": False}}
        ]


    cursor = user_lists_col.find(query).sort("created_at", -1)

    results = []

    async for list_doc in cursor:
        results.append(UserListBasic(
            list_id=str(list_doc["_id"]),
            name=list_doc.get("name", "Lista sin nombre"),
            icon_name=list_doc.get("icon_name", "default"),
            color_hex=list_doc.get("color_hex", "#FFFFFF"),
            item_count=len(list_doc.get("items", [])),
            is_archived=list_doc.get("is_archived", False)
        ))
    return results


@app.post("/lists/{list_id}/items", response_model=UserListBasic)
async def api_add_item_to_list(list_id: str, req: ItemAddRequest, user_id: str = Depends(get_current_user_uid)):
    # 1. Validar item
    if not await items_col.find_one({"itemId": req.item_id}, {"_id": 1}):
        raise HTTPException(404, "Item no encontrado")

    # 2. Update
    try:
        result = await user_lists_col.update_one(
            {"_id": ObjectId(list_id), "user_id": user_id},
            {"$addToSet": {"items": req.item_id}}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="ID de lista inválido")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Lista no encontrada o no pertenece al usuario")

    # --- REUTILIZAMOS EL HELPER ---
    return await _get_list_and_map_to_basic(list_id)

@app.get("/lists/{list_id}", response_model=UserListDetail)
async def api_get_list_details(list_id: str, user_id: str = Depends(get_current_user_uid)):
    try:
        # Motor: await find_one
        list_doc = await user_lists_col.find_one({"_id": ObjectId(list_id), "user_id": user_id})
    except Exception:
        raise HTTPException(status_code=400, detail="ID de lista inválido")

    if not list_doc:
        raise HTTPException(status_code=404, detail="Lista no encontrada o no pertenece al usuario")

    item_ids = list_doc.get("items", [])
    item_details_list = []

    # --- OPTIMIZACIÓN IMPORTANTE ---
    # En lugar de hacer un bucle for con find_one (N+1 queries),
    # usamos el operador $in para traerlos todos de una vez.
    if item_ids:
        cursor = items_col.find({"itemId": {"$in": item_ids}})
        # Traemos todos los items de golpe
        items_docs = await cursor.to_list(length=None)

        # Mapeamos los resultados
        for doc in items_docs:
            rec_item = row_to_recitem(doc, distance=0.0)
            item_details_list.append(SearchResultItem(
                item_id=rec_item.item_id,
                title=rec_item.title,
                domain=doc.get("domain", "unknown"),
                image_url=rec_item.image_url
            ))

    return UserListDetail(
        list_id=str(list_doc["_id"]),
        name=list_doc.get("name"),
        item_count=len(item_details_list),
        icon_name=list_doc.get("icon_name", "default"),
        color_hex=list_doc.get("color_hex", "#FFFFFF"),
        items=item_details_list
    )


@app.put("/lists/{list_id}", response_model=UserListBasic)
async def api_update_list(list_id: str, req: ListUpdateRequest, user_id: str = Depends(get_current_user_uid)):
    if not req.name or len(req.name) < 1:
        raise HTTPException(status_code=400, detail="El nombre de la lista no puede estar vacío")

    try:
        result = await user_lists_col.update_one(
            {"_id": ObjectId(list_id), "user_id": user_id},
            {"$set": {
                "name": req.name,
                "icon_name": req.icon_name,
                "color_hex": req.color_hex
            }}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="ID de lista inválido")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Lista no encontrada o no pertenece al usuario")

    # --- REUTILIZAMOS EL HELPER ---
    return await _get_list_and_map_to_basic(list_id)
@app.delete("/lists/{list_id}", status_code=204)
async def api_delete_list(list_id: str, user_id: str = Depends(get_current_user_uid)):
    """Elimina una lista completa."""
    try:
        # Motor: await delete_one
        result = await user_lists_col.delete_one(
            {"_id": ObjectId(list_id), "user_id": user_id}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="ID de lista inválido")

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Lista no encontrada o no pertenece al usuario")

    return Response(status_code=204)


@app.delete("/lists/{list_id}/items/{item_id}", response_model=UserListBasic)
async def api_remove_item_from_list(list_id: str, item_id: str, user_id: str = Depends(get_current_user_uid)):
    """Elimina un solo item de una lista."""
    try:
        result = await user_lists_col.update_one(
            {"_id": ObjectId(list_id), "user_id": user_id},
            {"$pull": {"items": item_id}}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="ID de lista inválido")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Lista no encontrada o no pertenece al usuario")


    updated_doc = await user_lists_col.find_one({"_id": ObjectId(list_id)})

    return UserListBasic(
        list_id=str(updated_doc["_id"]),
        name=updated_doc.get("name"),
        icon_name=updated_doc.get("icon_name", "default"),
        color_hex=updated_doc.get("color_hex", "#FFFFFF"),
        item_count=len(updated_doc.get("items", []))
    )


@app.get("/item/{item_id}", response_model=ItemDetailResponse)
async def api_get_item_details(item_id: str):
    details = await get_item_details(item_id)
    if details is None:
        raise HTTPException(status_code=404, detail=f"Item with ID '{item_id}' not found")
    return details


@app.get("/user/usage", response_model=UserUsageStatus)
async def api_get_user_usage(user_id: str = Depends(get_current_user_uid)):
    today_utc_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    sessions_today = await sessions_col.count_documents({
        "user_id": user_id,
        "created_date_utc": today_utc_str
    })

    remaining = max(0, SESSION_DAILY_LIMIT - sessions_today)

    return UserUsageStatus(
        daily_limit=SESSION_DAILY_LIMIT,
        sessions_today=sessions_today,
        remaining_today=remaining
    )


@app.put("/lists/{list_id}/archive", response_model=UserListBasic)
async def api_archive_list(list_id: str, user_id: str = Depends(get_current_user_uid)):
    # Llamamos al helper con True
    return await _set_list_archive_status(list_id, user_id, archive=True)


@app.put("/lists/{list_id}/unarchive", response_model=UserListBasic)
async def api_unarchive_list(list_id: str, user_id: str = Depends(get_current_user_uid)):
    # Llamamos al helper con False
    return await _set_list_archive_status(list_id, user_id, archive=False)


@app.post("/favorites/{item_id}", status_code=201)
async def api_add_favorite(item_id: str, user_id: str = Depends(get_current_user_uid)):
    # Motor: await find_one
    if not await items_col.find_one({"itemId": item_id}, {"_id": 1}):
        raise HTTPException(404, "Item no encontrado")

    now = datetime.now(timezone.utc)
    favorite_doc = {
        "user_id": user_id,
        "item_id": item_id,
        "added_at": now
    }
    try:
        # Motor: await insert_one
        await user_favorites_col.insert_one(favorite_doc)
        return {"message": "Item añadido a favoritos"}
    except Exception as error:
        if "E11000" in str(error):
            return {"message": "Item ya estaba en favoritos"}
        # logger.error(...)
        raise HTTPException(status_code=500, detail="Error interno al añadir favorito")


@app.delete("/favorites/{item_id}", status_code=204)
async def api_remove_favorite(item_id: str, user_id: str = Depends(get_current_user_uid)):
    await user_favorites_col.delete_one({"user_id": user_id, "item_id": item_id})
    return Response(status_code=204)


@app.get("/favorites", response_model=List[SearchResultItem])
async def api_get_favorites(user_id: str = Depends(get_current_user_uid)):
    """Obtiene todos los items favoritos de un usuario (OPTIMIZADO)."""

    # 1. Obtener IDs de favoritos (Motor: to_list)
    cursor = user_favorites_col.find({"user_id": user_id}).sort("added_at", -1)
    favorites_docs = await cursor.to_list(length=None)

    if not favorites_docs:
        return []

    favorite_item_ids = [doc["item_id"] for doc in favorites_docs]

    results = []

    # 2. OPTIMIZACIÓN: Traer todos los items de una vez con $in
    # Evitamos hacer N queries dentro de un bucle
    items_cursor = items_col.find({"itemId": {"$in": favorite_item_ids}})
    items_docs = await items_cursor.to_list(length=None)

    # Mapeo rápido en memoria (esto es rapidísimo en Python)
    # Convertimos a diccionario para mantener el orden si fuera necesario,
    # aunque aquí el orden de visualización depende de cómo los proceses.
    # Si quieres mantener el orden de "agregado recientemente", itera sobre favorite_item_ids

    items_map = {doc["itemId"]: doc for doc in items_docs}

    for item_id in favorite_item_ids:
        doc = items_map.get(item_id)
        if doc:
            rec_item = row_to_recitem(doc, distance=0.0)
            results.append(SearchResultItem(
                item_id=rec_item.item_id,
                title=rec_item.title,
                domain=doc.get("domain", "unknown"),
                image_url=rec_item.image_url
            ))

    return results


@app.get("/favorites/status/{item_id}", response_model=FavoriteStatusResponse)
async def api_get_favorite_status(item_id: str, user_id: str = Depends(get_current_user_uid)):
    # Motor: await count_documents
    count = await user_favorites_col.count_documents({"user_id": user_id, "item_id": item_id})
    return FavoriteStatusResponse(item_id=item_id, is_favorite=(count > 0))

# Endpoint para randomizar
@app.post("/session/{session_id}/randomize", response_model=SeedResponse)
async def api_session_randomize(session_id: str, user_id: str = Depends(get_current_user_uid)):
    # Helper async
    s = await get_session(session_id)
    if not s or s["user_id"] != user_id:
        raise HTTPException(404, "Session not found or unauthorized")
    if bool(s.get("finished", False)):
        raise HTTPException(400, "Session already finished")

    shown_in_session = s.get("shown", [])
    history = s.get("history", [])

    exclude_ids = list(shown_in_session)
    if history:
        exclude_ids.append(history[-1][0])

    pipeline = [
        {"$match": {
            "domain": s["domain"],
            "itemId": {"$nin": exclude_ids}
        }},
        {"$sample": {"size": 1}},
        {"$project": {"embedding": 0}}
    ]

    # Motor: aggregate -> to_list
    cursor = items_col.aggregate(pipeline)
    results = await cursor.to_list(length=1)

    if not results:
        # Motor: update_one
        await sessions_col.update_one(
            {"session_id": session_id},
            {"$set": {"finished": True, "history": history, "shown": shown_in_session}}
        )
        return SeedResponse(seed_item=None)

    new_doc = results[0]
    new_seed = row_to_recitem(new_doc)

    # Motor: update_one
    await sessions_col.update_one(
        {"session_id": session_id},
        {"$set": {
            "last_item_id": new_seed.item_id,
            "history": history,
            "shown": shown_in_session
        }}
    )

    return SeedResponse(seed_item=new_seed)


@app.post("/users/profile")
async def create_or_update_profile(
        profile_data: UserProfileRequest,
        user_id: str = Depends(get_current_user_uid)
):
    try:
        # Motor: await update_one
        await db.users.update_one(
            {"_id": user_id},
            {"$set": {
                "age": profile_data.age,
                "name": profile_data.name,
                "updated_at": datetime.now()
            }},
            upsert=True
        )
        return {"status": "success", "message": "Perfil guardado"}

    except Exception as error:
        print(f"Error DB: {error}")
        raise HTTPException(status_code=500, detail="Error interno guardando perfil")

@app.get("/users/get-email/{username}" , response_model = UserLookupResponse)
async def get_email(username: str):
    user = await db.users.find_one({"name": username})
    if user:
        return {"email": user["email"]}
    else:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")


@app.post("/users/create")
async def create_user(user: UserCreate):

    # Verificar que el nombre de usuario no esté registrado
    existing_user = await db.users.find_one({"name": {"$regex": f"^{user.username}$", "$options": "i"}})
    if existing_user:
        raise HTTPException(400, "Username ya registrado")

    # Verificar que el email no esté registrado
    if await db.users.find_one({"email": user.email}):
        raise HTTPException(400, "Email ya registrado")

    new_user_doc = {
        "firebaseUid": user.firebaseUid,
        "email": user.email,
        "name": user.username,
        "age": user.age,
        "profile_picture": user.profile_picture,
        "role": "USER",
        "createdAt": datetime.now(timezone.utc)
    }
    result = await db.users.insert_one(new_user_doc)
    return {"status": "User created", "id": str(result.inserted_id)}


@app.get("/users/check-email/{email}")
async def check_email_availability(email: str):
    # Buscamos si existe el email (ignorando mayúsculas/minúsculas)
    user = await db.users.find_one({"email": {"$regex": f"^{email}$", "$options": "i"}})

    if user:
        return {"available": False}  # Ya existe
    return {"available": True}  # Está libre


@app.get("/users/check-username/{username}")
async def check_username_availability(username: str):
    # Buscamos en la BD usando regex para ignorar mayúsculas/minúsculas
    # El "^...$" asegura que sea el match exacto de inicio a fin
    existing_user = await db.users.find_one({"name": {"$regex": f"^{username}$", "$options": "i"}})

    if existing_user:
        return {"available": False}  # Ocupado
    return {"available": True}  # Libre

@app.get("/users/me/exists")
async def check_user_exists(

    current_user_uid: str = Depends(get_current_user_uid)
):
    user = await db.users.find_one({"firebaseUid": current_user_uid})
    if user:
        return {"exists": True}
    else:
        return {"exists": False}


@app.get("/users/{firebase_uid}", response_model=UserProfileResponse)
async def get_user_profile(firebase_uid: str):

    # Buscamos por el campo "firebaseUid"
    user = await db.users.find_one({"firebaseUid": firebase_uid})
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    # Mapeamos los datos de Mongo al Modelo Pydantic
    return UserProfileResponse(
        firebaseUid=user.get("firebaseUid"),
        name=user.get("name", "Usuario"),  # Fallback por seguridad
        email=user.get("email", ""),
        age=user.get("age", 0),
        profile_picture=user.get("profile_picture"),

        # Opcionales
        country=user.get("country", "International"),
        cover_image=user.get("cover_image"),
        show_age=user.get("show_age", True)
    )

# 1. El nuevo modelo que espera recibir de OutSystems
class UserSync(BaseModel):
    outsystems_id: int
    username: str

# 2. El endpoint que OutSystems llamará
@app.post("/users/sync")
async def sync_outsystems_user(user: UserSync):
    
    # Verificamos si ya existe por si acaso
    existing_user = await db.users.find_one({"outsystems_id": user.outsystems_id})
    if existing_user:
        return {"status": "User already synchronized"}

    new_user_doc = {
        "outsystems_id": user.outsystems_id,
        "username": user.username,
        "role": "USER",
        "createdAt": datetime.now(timezone.utc)
    }
    
    result = await db.users.insert_one(new_user_doc)
    return {"status": "User created in Mongo", "mongo_id": str(result.inserted_id)}
    
@app.put("/users/{firebase_uid}")
async def update_user(firebase_uid: str, update_data: UserUpdateRequest):
    # Filtramos los campos que no sean None (para no borrar datos existentes con nulls)
    update_dict = {k: v for k, v in update_data.model_dump().items() if v is not None}

    if not update_dict:
        return {"message": "No changes sent"}

    result = await db.users.update_one(
        {"firebaseUid": firebase_uid},
        {"$set": update_dict}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    return {"message": "User updated successfully"}


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/landing/carousel/horror", response_model=SearchResponse)
async def api_get_horror_carousel():
    """Devuelve una lista predefinida de películas para el carrusel de inicio."""
    
    # IMPORTANTE: Estos títulos deben coincidir EXACTAMENTE con cómo 
    # están escritos en tu base de datos (incluyendo el año si lo tienen).
    target_titles = [
        "Child's Play 2 (1990)",
        "Halloween: The Curse of Michael Myers (Halloween 6: The Curse of Michael Myers) (1995)",
        "Alien (1979)",
        "Silence of the Lambs, The (1991)",
        "Texas Chainsaw Massacre, The (2003)",
        "Grudge, The (2004)"
    ]

    # Hacemos una sola consulta rápida buscando cualquiera de esos títulos
    cursor = items_col.find(
        {
            "title": {"$in": target_titles}, 
            "domain": "movie"
        },
        {"embedding": 0} # No necesitamos el vector aquí, ahorra memoria
    ).limit(6)

    docs = await cursor.to_list(length=6)

    results = []
    for doc in docs:
        # Usamos tu función existente para asegurar que traiga la imagen de TMDB
        rec_item = await row_to_recitem(doc, distance=0.0) 
        
        results.append(SearchResultItem(
            item_id=rec_item.item_id,
            title=rec_item.title,
            domain=doc.get("domain", "movie"),
            image_url=rec_item.image_url
        ))

    # Reutilizamos tu modelo SearchResponse para no inventar estructuras nuevas
    return SearchResponse(results=results)