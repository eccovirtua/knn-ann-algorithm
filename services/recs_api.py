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
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient
from services import tmdb_api
from services.lastfm_api import get_album_art
from services.lastfm_api import PLACEHOLDER
from pydantic import BaseModel
from datetime import datetime, timezone
import numpy as np
from sklearn.cluster import KMeans


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
user_watched_col = db["user_watched"]


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
    imdb_score: str = "N/A"

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

class ColdStartRequest(BaseModel):
    selected_item_ids: List[str]

class ClusterRecommendation(BaseModel):
    cluster_title: str
    recommendations: List[SearchResultItem]

class ColdStartResponse(BaseModel):
    clusters: List[ClusterRecommendation]

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


async def compute_next_seed(domain: str, user_id: str = None, session_history: List[Tuple[str, int]] = None) -> Optional[RecItem]:
    """
    Calcula el siguiente item semilla aplicando Serendipity Re-ranking.
    """
    history = session_history
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
    if last_positive_id:
        seed_doc = await items_col.find_one({"itemId": last_positive_id}, {"embedding": 1})
        if seed_doc and "embedding" in seed_doc:
            query_vector = seed_doc["embedding"]

    pipeline = []
    if query_vector:
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
        pipeline.append({"$match": {"domain": domain}})
        pipeline.append({"$sample": {"size": 10}})

    pipeline.append({
        "$match": {
            "domain": domain,
            "itemId": {"$nin": list(shown)}
        }
    })

    pipeline.append({"$project": {"embedding": 0}})
    pipeline.append({"$limit": 10}) # Limitamos a 10 candidatos para reordenar

    cursor = items_col.aggregate(pipeline)
    results = await cursor.to_list(length=10)

    if results:
        # APLICAR SERENDIPITY RE-RANKING
        positive_ids = [iid for iid, fb in history if fb > 0]
        user_profile_genres = set()
        
        if positive_ids:
            positive_docs = await items_col.find({"itemId": {"$in": positive_ids}}).to_list(length=None)
            for d in positive_docs:
                user_profile_genres.update(_genres_to_set(d.get("genres", [])))
                
        for doc in results:
            base_score = _calculate_quality_score(doc, domain)
            if base_score <= 0.1:
                base_score = 1.0 # Empuje a items válidos sin score previo
                
            doc_genres = _genres_to_set(doc.get("genres", []))
            genre_sim = jaccard(doc_genres, user_profile_genres)
            serendipity_factor = 1.0 - (genre_sim * 0.5) # Ajustamos penalización
            
            doc["final_score"] = base_score * serendipity_factor

        results.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
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


async def build_final_grid(session_id: str, user_id: str, domain: str, history: List[Tuple[str, int]], target_n: int = 20) -> List[RecItem]:
    """Genera la lista final aplicando Serendipia y eliminando duplicados."""
    shown_ids = [iid for iid, _ in history]
    positive_ids = [iid for iid, fb in history if fb > 0]

    final_docs = []
    user_profile_genres = set()

    # 1. Recomendaciones Vectoriales (Si hay likes previos)
    if positive_ids:
        # Extraemos el perfil de géneros de todo lo que le gustó en la sesión
        positive_docs = await items_col.find({"itemId": {"$in": positive_ids}}).to_list(length=None)
        for d in positive_docs:
            user_profile_genres.update(_genres_to_set(d.get("genres", [])))
            
        last_like = positive_ids[-1]
        seed_doc = await items_col.find_one({"itemId": last_like}, {"embedding": 1})

        if seed_doc and "embedding" in seed_doc:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": seed_doc["embedding"],
                        "numCandidates": 100,
                        "limit": 30  # Expandimos límite para dejar que la Serendipia actúe
                    }
                },
                {"$match": {"domain": domain, "itemId": {"$nin": shown_ids}}},
                {"$project": {"embedding": 0}}
            ]
            cursor = items_col.aggregate(pipeline)
            final_docs.extend(await cursor.to_list(length=None))

    # 2. Relleno "Joyas Ocultas" / Random
    current_ids = [d["itemId"] for d in final_docs] + shown_ids
    needed = target_n - len(final_docs)

    if needed > 0:
        fill_pipeline = [
            {"$match": {"domain": domain, "itemId": {"$nin": current_ids}}},
            {"$sample": {"size": needed * 2}}, # Traemos el doble de muestra para que haya variedad
            {"$project": {"embedding": 0}}
        ]
        cursor = items_col.aggregate(fill_pipeline)
        final_docs.extend(await cursor.to_list(length=None))

    # 3. Re-Ranking con SERENDIPITY
    for doc in final_docs:
        base_score = _calculate_quality_score(doc, domain)
        if base_score <= 0.1:
            base_score = 1.0 # Empujoncito para joyas ocultas

        doc_genres = _genres_to_set(doc.get("genres", []))
        genre_sim = jaccard(doc_genres, user_profile_genres)
        
        serendipity_factor = 1.0 - (genre_sim * 0.5)
        final_score = base_score * serendipity_factor
        
        doc["final_score"] = final_score
        doc["serendipity_factor"] = round(serendipity_factor, 2)

    # 4. Ordenar, deduplicar y recortar
    final_docs.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    
    seen = set()
    unique_docs = []
    for d in final_docs:
        if d["itemId"] not in seen:
            seen.add(d["itemId"])
            unique_docs.append(d)
            if len(unique_docs) == target_n:
                break
                
    final_docs = unique_docs

    final_rec_items = []
    final_items_to_save = []

    for doc in final_docs:
        rec_item = await row_to_recitem(doc)
        item_data = rec_item.model_dump()
        
        item_data["quality_score"] = round(doc.get("final_score", 0.0), 2)
        item_data["serendipity_factor"] = doc.get("serendipity_factor", 1.0)

        final_rec_items.append(rec_item)
        final_items_to_save.append(item_data)

    avg_quality = 0.0
    if final_items_to_save:
        avg_quality = sum(x["quality_score"] for x in final_items_to_save) / len(final_items_to_save)

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

    # 1. Optimización de Query: 
    # Usamos proyección para traer solo lo que necesitamos desde la base de datos
    # Si configuraste un "Text Index" en MongoDB, podrías usar {"$text": {"$search": query}}
    filter_query = {"title": {"$regex": query, "$options": "i"}}
    projection = {
        "itemId": 1, 
        "title": 1, 
        "image_url": 1, 
        "domain": 1, 
        "_id": 0 # Excluimos el embedding automáticamente al no pedirlo
    }

    cursor = items_col.find(filter_query, projection).limit(limit)
    docs = await cursor.to_list(length=limit)

    # 2. Mapeo Limpio:
    # Evitamos llamar a funciones extra si podemos construir el objeto directamente
    return [
        SearchResultItem(
            item_id=str(doc.get("itemId", "unknown")),
            title=doc.get("title", "Untitled"),
            domain=doc.get("domain", "movie"),
            image_url=doc.get("image_url", "")
        ) for doc in docs
    ]

async def get_item_details(item_id: str) -> Optional[ItemDetailResponse]:
    # Motor: await find_one
    doc = await items_col.find_one({"itemId": item_id})
    if not doc:
        return None

    rec_item = await row_to_recitem(doc, distance=0.0)

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
    # Validamos que no sea solo espacios en blanco
    clean_query = query.strip()
    if len(clean_query) < 3:
        raise HTTPException(
            status_code=400, 
            detail="La búsqueda debe tener al menos 3 caracteres"
        )

    try:
        results = await search_items(clean_query, limit)
        return SearchResponse(results=results)
    except Exception as e:
        # Log del error y respuesta genérica para no exponer el backend
        print(f"Error en búsqueda: {e}")
        raise HTTPException(status_code=500, detail="Error interno en el motor de búsqueda")
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
async def api_add_favorite(item_id: str, user_id: str = Depends(get_outsystems_user)):
    """Añade una película a favoritos. Verifica que exista primero."""
    
    # 1. Validar que la película realmente exista en nuestra colección principal
    if not await items_col.find_one({"itemId": item_id}, {"_id": 1}):
        raise HTTPException(404, "Item no encontrado en la base de datos")

    # 2. Guardar o actualizar usando upsert (evita el error E11000 de duplicidad)
    await user_favorites_col.update_one(
        {"user_id": user_id, "item_id": item_id},
        {"$set": {
            "user_id": user_id,
            "item_id": item_id,
            "domain": "movie",
            "timestamp": datetime.utcnow()
        }},
        upsert=True
    )
    
    return {"status": "success", "message": "Item añadido a favoritos"}


@app.delete("/favorites/{item_id}")
async def api_remove_favorite(item_id: str, user_id: str = Depends(get_outsystems_user)):
    """Elimina una película de favoritos."""
    
    await user_favorites_col.delete_one({"user_id": user_id, "item_id": item_id})
    
    # Retornar un JSON es más fácil de leer para OutSystems que un Response 204 vacío
    return {"status": "success", "message": "Item eliminado de favoritos"}

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
        "Sueño de fuga",
        "El Padrino",
        "Batman: El caballero de la noche",
        "El Padrino II",
        "12 Hombres en Pugna",
        "El señor de los anillos: El retorno del rey",
        "La Lista de Schindler",
        "Tiempos Violentos",
        "El Bueno, El Malo y El Feo",
        "El señor de los anillos: Las dos torres",
        "Forrest Gump",
        "El Club de la Pelea"



    ]

    # Hacemos una sola consulta rápida buscando cualquiera de esos títulos
    cursor = items_col.find(
        {
            "title": {"$in": target_titles}, 
            "domain": "movie"
        },
        {"embedding": 0} # Esto trae TODO lo demás, incluyendo imdb_score
    ).limit(12)
    docs = await cursor.to_list(length=12)

    results = []
    for doc in docs:
        # Usamos tu función existente para asegurar que traiga la imagen de TMDB
        rec_item = await row_to_recitem(doc, distance=0.0) 
        
        results.append(SearchResultItem(
            item_id=rec_item.item_id,
            title=rec_item.title,
            domain=doc.get("domain", "movie"),
            image_url=rec_item.image_url,
            imdb_score=str(doc.get("imdb_score", "N/A"))
        ))

    # Reutilizamos tu modelo SearchResponse para no inventar estructuras nuevas
    return SearchResponse(results=results)

@app.get("/onboarding/movies", response_model=SearchResponse)
async def api_get_onboarding_movies(
    filter_by: str = Query("mejores", description="Opciones: mejores, nuevas, genero"),
    genre: Optional[str] = Query(None, description="Nombre del género si filter_by es 'genero'")
):
    """Devuelve 30 películas para la pantalla de Onboarding filtradas dinámicamente"""
    
    # 1. Filtro base: que sean películas y que ya tengan la info enriquecida (director existe)
    match_stage = {"domain": "movie", "director": {"$exists": True}}
    
    # Si el usuario eligió "genero" en la UI y envió un género específico
    if filter_by == "genero" and genre:
        # Busca el género exacto dentro de la lista de géneros en Mongo (ignorando mayúsculas/minúsculas)
        match_stage["genres"] = {"$regex": f"^{genre}$", "$options": "i"}
        
    pipeline = [{"$match": match_stage}]
    
    # 2. Lógica de Ordenamiento (Sorting) según el botón que presionó el usuario
    if filter_by == "mejores":
        # Ordenamos por imdb_score de mayor a menor (-1)
        pipeline.append({"$sort": {"imdb_score": -1}})
        
    elif filter_by == "nuevas":
        pipeline.append({"$sort": {"year_str": -1}}) # Ordena por el año más reciente
        
    elif filter_by == "genero":
        # Para el filtro de género, también mostramos las mejores calificadas de ese género
        pipeline.append({"$sort": {"imdb_score": -1}})
        
    # 3. Limitamos a 30 y quitamos el vector para que sea súper rápido
    pipeline.append({"$limit": 30})
    pipeline.append({"$project": {"embedding": 0}})
    
    cursor = items_col.aggregate(pipeline)
    docs = await cursor.to_list(length=30)
    
    results = []
    for doc in docs:
        rec_item = await row_to_recitem(doc, distance=0.0)
        results.append(SearchResultItem(
            item_id=rec_item.item_id,
            title=rec_item.title,
            domain="movie",
            image_url=rec_item.image_url
        ))
        
    return SearchResponse(results=results)

@app.post("/onboarding/cold-start", response_model=ColdStartResponse)
async def api_generate_cold_start_recs(
    req: ColdStartRequest,
    user_id: str = Depends(get_outsystems_user)
    ):
    """
    Recibe las 5 películas elegidas en el onboarding, las agrupa matemáticamente 
    en 3 focos de interés, y devuelve recomendaciones dinámicas para cada foco.
    Además, guarda estas selecciones iniciales como los primeros Favoritos del usuario.
    """
    if not req.selected_item_ids:
        raise HTTPException(status_code=400, detail="Debe proporcionar items seleccionados.")

    # =======================================================================
    # NUEVA FASE 1: GUARDAR SELECCIONES COMO FAVORITOS
    # Usamos upsert=True para crear el registro si no existe, o actualizar 
    # la fecha si el usuario por alguna razón repite el Onboarding.
    # =======================================================================
    for item_id in req.selected_item_ids:
        await user_favorites_col.update_one(
            {"user_id": user_id, "item_id": item_id},
            {"$set": {
                "user_id": user_id, 
                "item_id": item_id, 
                "domain": "movie",
                "timestamp": datetime.utcnow()
            }},
            upsert=True
        )
    # =======================================================================

    # 1. Traer los documentos de las películas seleccionadas (con sus vectores)
    docs = await items_col.find({
        "itemId": {"$in": req.selected_item_ids}
    }).to_list(length=len(req.selected_item_ids))

    if not docs:
        raise HTTPException(status_code=404, detail="No se encontraron las películas.")

    # Extraer vectores y títulos (priorizamos el título en español si el script ya lo puso)
    vectors = []
    titles = []
    for d in docs:
        if "embedding" in d:
            vectors.append(d["embedding"])
            titles.append(d.get("title", d.get("title", "Desconocida")))

    if len(vectors) == 0:
        raise HTTPException(status_code=400, detail="Las películas seleccionadas no tienen vectores.")

    # 2. Clustering (Agrupamiento Matemático K-Means)
    # Si eligió 5 películas, hacemos 3 grupos. Si eligió menos, hacemos menos grupos.
    num_clusters = min(3, len(vectors))
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)

    response_clusters = []

    # 3. Procesar cada clúster
    for i in range(num_clusters):
        # Películas que cayeron en este grupo
        cluster_indices = np.where(labels == i)[0]
        cluster_titles = [titles[idx] for idx in cluster_indices]
        
        # El Vector Promedio (Centroide) de este grupo
        centroid = kmeans.cluster_centers_[i].tolist()

        # 4. Generar el título dinámico de la sección
        if len(cluster_titles) == 1:
            row_title = f"Porque te gustó {cluster_titles[0]}"
        elif len(cluster_titles) == 2:
            row_title = f"Si te interesan {cluster_titles[0]} y {cluster_titles[1]}"
        else:
            row_title = f"Inspirado en {cluster_titles[0]}, {cluster_titles[1]} y más"

        # 5. Búsqueda Vectorial Pura en MongoDB usando el Centroide
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index", # Usa el nombre de tu índice vectorial en Mongo
                    "path": "embedding",
                    "queryVector": centroid,
                    "numCandidates": 150,
                    "limit": 45
                }
            },
            {
                "$match": {
                    "domain": "movie",
                    "itemId": {"$nin": req.selected_item_ids} # Excluir las que ya seleccionó
                }
            },
            {"$project": {"embedding": 0}},
            {"$limit": 45} # Devolvemos 45 películas por fila
        ]

        cursor = items_col.aggregate(pipeline)
        cluster_docs = await cursor.to_list(length=45)

        # 6. Mapear resultados
        recommendations = []
        for c_doc in cluster_docs:
            rec_item = await row_to_recitem(c_doc, distance=0.0)
            recommendations.append(SearchResultItem(
                item_id=rec_item.item_id,
                title=c_doc.get("title", rec_item.title),
                domain="movie",
                image_url=rec_item.image_url,
                imdb_score=str(c_doc.get("imdb_score", "N/A"))
            ))

        if recommendations:
            response_clusters.append(ClusterRecommendation(
                cluster_title=row_title,
                recommendations=recommendations
            ))

    return ColdStartResponse(clusters=response_clusters)

# ==========================================
# ENDPOINTS PARA "VISTO" (WATCHED)
# ==========================================

@app.post("/watched/{item_id}")
async def mark_as_watched(item_id: str, user_id: str = Depends(get_outsystems_user)):
    """Marca una película como vista por el usuario."""
    await user_watched_col.update_one(
        {"user_id": user_id, "item_id": item_id},
        {"$set": {"user_id": user_id, "item_id": item_id, "domain": "movie", "timestamp": datetime.utcnow()}},
        upsert=True
    )
    return {"status": "success", "message": "Marcada como vista"}

@app.delete("/watched/{item_id}")
async def unmark_as_watched(item_id: str, user_id: str = Depends(get_outsystems_user)):
    """Desmarca una película como vista."""
    await user_watched_col.delete_one({"user_id": user_id, "item_id": item_id})
    return {"status": "success", "message": "Desmarcada como vista"}


# ==========================================
# ENDPOINT DE ESTADO (Para la Pantalla de Detalle)
# ==========================================

@app.get("/item-status/{item_id}")
async def get_item_status(item_id: str, user_id: str = Depends(get_outsystems_user)):
    """
    Devuelve True o False indicando si el usuario tiene esta película 
    en sus favoritos y/o en sus vistas. Ideal para inicializar los botones.
    """
    fav = await user_favorites_col.find_one({"user_id": user_id, "item_id": item_id})
    watched = await user_watched_col.find_one({"user_id": user_id, "item_id": item_id})
    
    return {
        "is_favorite": fav is not None,
        "is_watched": watched is not None
    }

@app.get("/recommendations/dynamic", response_model=ColdStartResponse)
async def api_dynamic_recommendations(user_id: str = Depends(get_outsystems_user)):
    """
    Lee TODOS los favoritos del usuario, genera clústeres basados en su 
    historial completo y excluye de las recomendaciones las películas 
    que ya vio o que ya tiene en favoritos.
    """
    
    # 1. Obtener el historial completo de Favoritos
    fav_cursor = user_favorites_col.find({"user_id": user_id})
    fav_item_ids = [doc["item_id"] async for doc in fav_cursor]

    if not fav_item_ids:
        # Si por alguna razón no tiene favoritos, podríamos devolver una lista vacía 
        # o lanzar un error para que OutSystems lo mande al Onboarding
        raise HTTPException(status_code=404, detail="El usuario no tiene favoritos para generar recomendaciones.")

    # 2. Obtener el historial de Vistas (Watched)
    watched_cursor = user_watched_col.find({"user_id": user_id})
    watched_item_ids = [doc["item_id"] async for doc in watched_cursor]

    # Lista maestra de exclusión (No recomendar lo que ya le gusta o ya vio)
    items_to_exclude = list(set(fav_item_ids + watched_item_ids))

    # 3. Traer los vectores solo de sus favoritos
    docs = await items_col.find({
        "itemId": {"$in": fav_item_ids}
    }).to_list(length=None)

    vectors = []
    titles = []
    for d in docs:
        if "embedding" in d:
            vectors.append(d["embedding"])
            titles.append(d.get("title", "Desconocida"))

    if len(vectors) == 0:
        raise HTTPException(status_code=400, detail="Los favoritos no tienen vectores.")

    # 4. Clustering (K-Means)
    num_clusters = min(3, len(vectors))
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)

    response_clusters = []

    # 5. Procesar cada clúster
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_titles = [titles[idx] for idx in cluster_indices]
        centroid = kmeans.cluster_centers_[i].tolist()

        # Generar título basado en sus gustos
        if len(cluster_titles) == 1:
            row_title = f"Porque te encanta {cluster_titles[0]}"
        elif len(cluster_titles) == 2:
            row_title = f"Si sigues pensando en {cluster_titles[0]} y {cluster_titles[1]}"
        else:
            row_title = f"Inspirado en tu gusto por {cluster_titles[0]}, {cluster_titles[1]} y más"

        # Búsqueda Vectorial con la lista maestra de exclusión
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index", 
                    "path": "embedding",
                    "queryVector": centroid,
                    "numCandidates": 150,
                    "limit": 45
                }
            },
            {
                "$match": {
                    "domain": "movie",
                    "itemId": {"$nin": items_to_exclude} # <--- LA MAGIA ESTÁ AQUÍ
                }
            },
            {"$project": {"embedding": 0}},
            {"$limit": 45} 
        ]

        cursor = items_col.aggregate(pipeline)
        cluster_docs = await cursor.to_list(length=45)

        recommendations = []
        for c_doc in cluster_docs:
            rec_item = await row_to_recitem(c_doc, distance=0.0)
            recommendations.append(SearchResultItem(
                item_id=rec_item.item_id,
                title=c_doc.get("title", rec_item.title),
                domain="movie",
                image_url=rec_item.image_url,
                imdb_score=str(c_doc.get("imdb_score", "N/A"))
            ))

        if recommendations:
            response_clusters.append(ClusterRecommendation(
                cluster_title=row_title,
                recommendations=recommendations
            ))

    return ColdStartResponse(clusters=response_clusters)

@app.get("/user/has-onboarded")
async def check_user_onboarded(user_id: str = Depends(get_outsystems_user)):
    """Verifica si el usuario ya tiene al menos 1 favorito guardado."""
    # Usamos limit=1 para que Mongo deje de buscar apenas encuentre el primero (máximo rendimiento)
    count = await user_favorites_col.count_documents({"user_id": user_id}, limit=1)
    
    return {"has_onboarded": count > 0}

@app.get("/favorites", response_model=List[SearchResultItem])
async def get_user_favorites_list(user_id: str = Depends(get_outsystems_user)):
    """Devuelve la lista completa de películas favoritas del usuario, de la más nueva a la más antigua."""
    
    # 1. Obtener IDs ordenados por fecha de adición (más reciente primero)
    fav_docs = await user_favorites_col.find({"user_id": user_id}).sort("timestamp", -1).to_list(length=None)
    fav_ids = [doc["item_id"] for doc in fav_docs]
    
    if not fav_ids:
        return []

    # 2. Traer los detalles de la base de datos principal
    items_cursor = items_col.find({"itemId": {"$in": fav_ids}})
    
    # Convertimos a diccionario para mantener el orden original de fav_ids
    items_dict = {doc["itemId"]: doc async for doc in items_cursor}
    
    # 3. Mapear al modelo de salida
    results = []
    for item_id in fav_ids:
        if item_id in items_dict:
            d = items_dict[item_id]
            rec_item = await row_to_recitem(d, distance=0.0)
            results.append(SearchResultItem(
                item_id=rec_item.item_id,
                title=d.get("title", rec_item.title),
                domain="movie",
                image_url=rec_item.image_url,
                imdb_score=str(d.get("imdb_score", "N/A"))
            ))
            
    return results


@app.get("/watched", response_model=List[SearchResultItem])
async def get_user_watched_list(user_id: str = Depends(get_outsystems_user)):
    """Devuelve la lista completa de películas vistas por el usuario."""
    
    watched_docs = await user_watched_col.find({"user_id": user_id}).sort("timestamp", -1).to_list(length=None)
    watched_ids = [doc["item_id"] for doc in watched_docs]
    
    if not watched_ids:
        return []

    items_cursor = items_col.find({"itemId": {"$in": watched_ids}})
    items_dict = {doc["itemId"]: doc async for doc in items_cursor}
    
    results = []
    for item_id in watched_ids:
        if item_id in items_dict:
            d = items_dict[item_id]
            rec_item = await row_to_recitem(d, distance=0.0)
            results.append(SearchResultItem(
                item_id=rec_item.item_id,
                title=d.get("title", rec_item.title),
                domain="movie",
                image_url=rec_item.image_url,
                imdb_score=str(d.get("imdb_score", "N/A"))
            ))
            
    return results

@app.get("/search/advanced", response_model=List[SearchResultItem])
async def advanced_search(
    q: Optional[str] = Query(None, description="Búsqueda por título"),
    genre: Optional[str] = Query(None, description="Filtro de género exacto"),
    keyword: Optional[str] = Query(None, description="Palabra clave"),
    director: Optional[str] = Query(None, description="Nombre del director"),
    year: Optional[str] = Query(None, description="Año de lanzamiento"),
    min_score: Optional[float] = Query(None, description="Puntuación mínima de 0 a 10"),
    sort_by: Optional[str] = Query("popularity", description="popularity o score")
):
    """
    Buscador avanzado. Combina cualquier cantidad de filtros. 
    No requiere autenticación.
    """
    
    # 1. El diccionario base. Siempre buscamos películas.
    query = {"domain": "movie"}

    # 2. Vamos agregando bloques a la consulta solo si el usuario los llenó
    if q:
        # Búsqueda insensible a mayúsculas en el título normalizado
        query["title_norm"] = {"$regex": q.lower(), "$options": "i"}
        
    if genre:
        # Busca el género exacto dentro del array genres_es
        query["genres_es"] = genre 
        
    if keyword:
        # Busca si la palabra clave está dentro del array keywords_es
        query["keywords_es"] = {"$regex": keyword.lower(), "$options": "i"}
        
    if director:
        # Búsqueda parcial del director
        query["director"] = {"$regex": director, "$options": "i"}
        
    if year:
        # Coincidencia exacta del año (es string en tu BD)
        query["year_str"] = year
        
    if min_score:
        # Mayor o igual ($gte) al score seleccionado
        query["imdb_score"] = {"$gte": min_score}

    # 3. Determinar el orden (Sorting)
    # Si eligió 'score', ordenamos por calificación.
    # Si eligió 'popularity', usamos imdb_votes (que es el mejor indicador de qué tan conocida es)
    if sort_by == "score":
        sort_criteria = [("imdb_score", -1)] # -1 significa descendente (de mayor a menor)
    else:
        sort_criteria = [("imdb_votes", -1)]

    # 4. Ejecutar la búsqueda en MongoDB
    # Limitamos a 50 resultados para no colapsar la memoria de OutSystems
    cursor = items_col.find(query).sort(sort_criteria).limit(50)
    
    # 5. Mapear resultados a nuestra estructura estándar
    results = []
    async for d in cursor:
        rec_item = await row_to_recitem(d, distance=0.0)
        results.append(SearchResultItem(
            item_id=rec_item.item_id,
            title=d.get("title_es", rec_item.title), # Priorizamos título en español
            domain="movie",
            image_url=rec_item.image_url,
            imdb_score=str(d.get("imdb_score", "N/A"))
        ))

    return results