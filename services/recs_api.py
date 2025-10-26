# services/recs_api.py
import os
import re
from fastapi import Response, Query  # <-- Asegúrate que 'Response' esté
from starlette.status import HTTP_204_NO_CONTENT
from bson import ObjectId # <-- ¡MUY IMPORTANTE para MongoDB!
import sys
import logging
from enum import Enum
from uuid import uuid4
from pathlib import Path
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
import numpy as np
from annoy import AnnoyIndex
from pymongo import MongoClient
import jwt
from jwt.exceptions import InvalidTokenError
import asyncio
from services import tmdb_api
from services.tmdb_api import fetch_movie_poster
from services.lastfm_api import get_album_art
from services.lastfm_api import PLACEHOLDER
# ---------- logging ----------
logger = logging.getLogger("recs_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
logger.addHandler(handler)

# ---------- config & env ----------
load_dotenv()
JWT_SECRET = os.getenv("JWT_SECRET", "N2wwJveBGKL6f8iWIL7nx+Cl0rMoJUWpyCfsbu+7mHQ=")
MONGO_URI = os.getenv("MONGODB_URI")
if not MONGO_URI:
    raise RuntimeError("MONGODB_URI is not defined")

SESSION_DAILY_LIMIT = 3 # Límite diario
# ---------- app ----------
app = FastAPI(title="Recommendation Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restringir en producción
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
# monta el router con prefijo /external
app.include_router(tmdb_api.router, prefix="/external", tags=["External APIs"])
security = HTTPBearer()

# ---------- Mongo (colecciones) ----------
mongo = MongoClient(MONGO_URI)
db = mongo.get_database()
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

# ---------- assets vectorizados (items + embeddings + annoy) ----------
BASE_DIR = Path(__file__).resolve().parents[1]
VECT_DIR = BASE_DIR / "data" / "vectorized"

items_df = pd.read_parquet(VECT_DIR / "items.parquet").reset_index(drop=True)
movieid_to_index = {item: i for i, item in enumerate(items_df["itemId"].tolist())}

embeds = np.load(VECT_DIR / "items_embeds.npz")["embeddings"]
dim = embeds.shape[1]
ann_index = AnnoyIndex(dim, metric="angular")
ann_index.load(str(VECT_DIR / "items_index.ann"))

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
    seed: RecItem
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
def get_user_id_from_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = payload.get("userId") or payload.get("sub")
        if not user_id:
            raise InvalidTokenError("Missing sub/userId")
        return user_id
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")

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
def get_history(user_id: str, domain: str) -> List[Tuple[str, int]]:
    docs = feedback_col.find({"user_id": user_id, "domain": domain}).sort("ts", 1)
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

def row_to_recitem(row: pd.Series, distance: float = 0.0) -> RecItem:
    item_id = row.get("item_id") or row.get("itemId")
    if item_id is None:
        print("⚠️ row_to_recitem recibió un row sin item_id válido:", row.to_dict())
    # Omitimos prints de DEBUG si ya verificamos que funcionan
    domain = row.get("domain")
    image_url = row.get("image_url") # Leemos la URL del dataset
    if domain == "music" and image_url == PLACEHOLDER:
        # Ya verificaste que esta línea se ejecuta y pone la URL a None
        image_url = None
    # --- Movies (TMDB) ---
    if domain == "movie" and not image_url:
        title = row.get("title", "")
        clean_title = title.split("(")[0].strip()
        try:
            image_url = asyncio.run(fetch_movie_poster(clean_title))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            image_url = loop.run_until_complete(fetch_movie_poster(clean_title))
    elif domain == "music" and not image_url:
        artist = row.get("artist", "")
        track = row.get("title", "")
        item_id = row.get("item_id") or row.get("itemId") or ""
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
                # 📢 La función get_album_art debe retornar la URL real o None.
                image_url = get_album_art(artist, track)
            except Exception as err:
                print(f"⚠️ Error obteniendo imagen de Last.fm: {err}")
                image_url = None
    # --- Fallback general ---
    if not image_url:
        image_url = "https://placehold.co/300x450?text=No+Image"

    return RecItem(
        item_id=row.get("item_id") or row.get("itemId") or "",
        title=row.get("title", ""),
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
def generate_new_seed(domain: str) -> RecItem:
    candidates = items_df[items_df["domain"] == domain]
    if candidates.empty:
        raise HTTPException(404, "no hay items para el dominio")
    row = candidates.sample(1).iloc[0]
    return row_to_recitem(row, distance=0.0)

def compute_next_seed(user_id: str, domain: str) -> Optional[RecItem]:
    history = get_history(user_id, domain)
    logger.info("compute_next_seed user=%s domain=%s history_len=%d", user_id, domain, len(history))

    shown = {item for item, _ in history}
    positives = [item for item, fb in history if fb > 0]

    if positives:
        base = positives[-1]
        idx = movieid_to_index.get(base)
        if idx is not None:
            K = 50
            neigh_idxs, _ = ann_index.get_nns_by_item(idx, K, include_distances=True)
            for neigh_idx in neigh_idxs[1:]:
                row = items_df.iloc[neigh_idx]
                candidate_id = row["itemId"]
                if row["domain"] == domain and candidate_id not in shown:
                    return row_to_recitem(row, distance=0.0)
    # fallback: random outside shown
    candidates = items_df[(items_df["domain"] == domain) & (~items_df["itemId"].isin(shown))]
    if not candidates.empty:
        row = candidates.sample(1).iloc[0]
        return row_to_recitem(row, distance=0.0)
    return None

def compute_next_seed_from_history(session_history: List[Tuple[str, int]], domain: str) -> Optional[RecItem]:
    shown = {item for item, _ in session_history}
    if not session_history:
        return None

    last_item, last_feedback = session_history[-1]
    idx = movieid_to_index.get(last_item)
    if idx is None:
        return None

    if last_feedback > 0:  # 👍 Like
        neigh_idxs, dists = ann_index.get_nns_by_item(idx, 30, include_distances=True)
        candidates = [(n_idx, d) for n_idx, d in zip(neigh_idxs[1:], dists[1:]) if items_df.iloc[n_idx]["domain"] == domain]
        candidates = [(n_idx, d) for n_idx, d in candidates if items_df.iloc[n_idx]["itemId"] not in shown]

        if not candidates:
            # Fallback: Busca un ítem aleatorio del dominio que no se haya mostrado
            fallback_candidates = items_df[(items_df["domain"] == domain) & (~items_df["itemId"].isin(shown))]
            if not fallback_candidates.empty:
                row = fallback_candidates.sample(1).iloc[0]
                return row_to_recitem(row, distance=0.0)
            # Si no hay absolutamente nada más que mostrar, entonces sí termina.
            return None
        cut = max(1, int(len(candidates) * 0.7))
        close = candidates[:cut]
        far = candidates[cut:]
        pool = list(close)
        if far:
            pool += random.sample(far, min(3, len(far)))
        n_idx, _ = random.choice(pool)
        row = items_df.iloc[n_idx]
        return row_to_recitem(row, distance=0.0)

    elif last_feedback < 0:  # 👎 Dislike
        vec = embeds[idx]
        all_ids = np.arange(len(embeds))
        dists = np.linalg.norm(embeds - vec, axis=1)
        farthest = all_ids[np.argsort(-dists)]
        for n_idx in farthest:
            row = items_df.iloc[n_idx]
            if row["domain"] == domain and row["itemId"] not in shown:
                return row_to_recitem(row, distance=0.0)
        return None

    else:  # feedback == 0 (neutro, semilla inicial)
        neigh_idxs, _ = ann_index.get_nns_by_item(idx, 20, include_distances=True)
        for n_idx in neigh_idxs[1:]:
            row = items_df.iloc[n_idx]
            if row["domain"] == domain and row["itemId"] not in shown:
                return row_to_recitem(row, distance=0.0)
        return None

def _get_quality_score(row: pd.Series) -> float:
    domain = row.get("domain")
    if domain == "movie":
        return float(_col(row, "imdb_score", 0.0) or 0.0)
    elif domain == "book":
        return float(_col(row, "google_rating", 0.0) or 0.0)
    elif domain == "music":
        return float(_col(row, "playcount", 0.0) or 0.0)
    return 0.0

def generate_diverse_recommendations(seen_items: List[str], top_per_domain: int = 5):
    df = items_df.copy()
    df = df[~df["itemId"].isin(seen_items)]
    if df.empty:
        return []
    df["quality_score"] = df.apply(_get_quality_score, axis=1)
    base_score = df["base_score"] if "base_score" in df.columns else 1.0
    df["boosted_score"] = base_score * (1 + df["quality_score"] / 10.0)
    recommendations = []
    for domain, group in df.groupby("domain"):
        top_items = group.nlargest(top_per_domain, "boosted_score")
        for _, row in top_items.iterrows():
            recommendations.append(
                RecItem(item_id=row["itemId"], title=row["title"], distance=0.0, image_url=row.get("image_url"))
            )
    return recommendations

def _collect_candidates(domain: str, shown: set, positives: List[str],
                        k_neighbors: int = K_VECINOS, exploration_sample: int = EXPLORATION_SAMPLE) -> dict:
    candidates = {}
    # vecinos de positivos
    for base in positives[-CONSIDER_LAST_POSITIVES:]:
        idx = movieid_to_index.get(base)
        if idx is None:
            continue
        neigh_idxs, dists = ann_index.get_nns_by_item(idx, k_neighbors, include_distances=True)
        for n_idx, dist in zip(neigh_idxs[1:], dists[1:]):
            row = items_df.iloc[n_idx]
            if row["domain"] != domain:
                continue
            cid = row["itemId"]
            if cid in shown:
                continue
            prev = candidates.get(cid)
            if prev is None or dist < prev:
                candidates[cid] = float(dist)
    pool = items_df[(items_df["domain"] == domain) & (~items_df["itemId"].isin(shown))]
    if not pool.empty:
        sample = pool.sample(min(exploration_sample, len(pool)))
        for _, row in sample.iterrows():
            cid = row["itemId"]
            # si no viene de vecinos damos una distancia alta para favorecer exploración
            if cid not in candidates:
                candidates[cid] = float(999.0)
    return candidates

def _score_and_rank_candidates(candidates: dict,
                               alpha_sim: float = ALPHA_SIM,
                               beta_pop: float = BETA_POP,
                               gamma_imdb: float = GAMMA_IMDB) -> List[dict]:
    scored = []
    for cid, dist in candidates.items():
        try:
            row = items_df.loc[movieid_to_index[cid]]
        except (ValueError, TypeError):
            continue
        sim_score = math.exp(-dist) if dist < 900 else 0.01
        pop = _get_popularity(row)
        imdb = float(_col(row, "imdb_score", 0.0)) / 10.0
        raw = alpha_sim * sim_score + beta_pop * pop + gamma_imdb * imdb
        scored.append({
            "item_id": cid,
            "score": raw,
            "dist": dist,
            "genres": _genres_to_set(_col(row, "genres", None)),
            "row": row
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

def build_final_grid(session_id: str,
                     user_id: str,
                     domain: str,
                     history: List[Tuple[str, int]],
                     target_n: int = TARGET_FINAL_N,
                     diversity_threshold: float = DIVERSITY_JACCARD_THRESHOLD) -> List[RecItem]:
    shown = {iid for iid, _ in history}
    positives = [iid for iid, fb in history if fb > 0]
    # --- 1) pool + scoring desde likes ---
    candidates = _collect_candidates(domain, shown, positives)
    scored = _score_and_rank_candidates(candidates)
    # greedy con diversidad (tomamos hasta 12 de aquí)
    picked_from_scored: List[str] = []
    selected_genres = []
    for c in scored:
        if len(picked_from_scored) >= 12:
            break
        cand_gen = c["genres"]
        max_j = 0.0
        for sg in selected_genres:
            max_j = max(max_j, jaccard(cand_gen, sg))
        adjusted_score = c["score"] - DELTA_NOVELTY * max_j
        if max_j < diversity_threshold or len(picked_from_scored) < 2:
            picked_from_scored.append(c["item_id"])
            selected_genres.append(cand_gen)
        if adjusted_score > 0 and (max_j < diversity_threshold or len(picked_from_scored) < 2):
            picked_from_scored.append(c["item_id"])
            selected_genres.append(cand_gen)

    # --- 2) hidden gems (alto rating + baja popularidad) ---
    dom_df = items_df[(items_df["domain"] == domain) & (~items_df["itemId"].isin(shown))]
    if dom_df.empty:
        dom_df = items_df[items_df["domain"] == domain]
    # heurísticas robustas
    dom_df = dom_df.copy()
    dom_df["__rating"] = dom_df.apply(_rating, axis=1)
    dom_df["__pop"] = dom_df.apply(_get_popularity, axis=1)
    dom_df["__rc"] = dom_df.apply(_rating_count, axis=1)
    # thresholds dinámicos
    pop_q20 = float(dom_df["__pop"].quantile(0.20)) if len(dom_df) > 0 else 0.0
    rating_q75 = float(dom_df["__rating"].quantile(0.75)) if len(dom_df) > 0 else 4.0
    hidden_mask = (dom_df["__rating"] >= rating_q75) & (dom_df["__pop"] <= pop_q20)
    hidden_df = dom_df[hidden_mask]
    hidden_ids = hidden_df["itemId"].tolist()
    random.shuffle(hidden_ids)
    hidden_ids = hidden_ids[:4]  # 4 hidden gems
    rc_q25 = int(dom_df["__rc"].quantile(0.25)) if len(dom_df) > 0 else 10
    und_mask = (dom_df["__rc"] <= max(5, rc_q25)) & (dom_df["__rating"] >= (rating_q75 * 0.75))
    und_df = dom_df[und_mask]
    underdog_ids = und_df["itemId"].tolist()
    random.shuffle(underdog_ids)
    underdog_ids = underdog_ids[:4]
    # --- 4) combinar y rellenar ---
    combined_ids = []
    def _safe_extend(ids):
        for iid in ids:
            if iid not in combined_ids and iid not in shown:
                combined_ids.append(iid)
    _safe_extend(picked_from_scored)
    _safe_extend(hidden_ids)
    _safe_extend(underdog_ids)
    # si faltan, rellenar con resto del dominio evitando repetidos y manteniendo diversidad suave
    if len(combined_ids) < target_n:
        need = target_n - len(combined_ids)
        # tomar por score remanente primero, luego random del dominio
        remaining_pool = [c["item_id"] for c in scored if c["item_id"] not in combined_ids]
        if len(remaining_pool) < need:
            extra_dom = dom_df[~dom_df["itemId"].isin(set(combined_ids) | shown)]["itemId"].tolist()
            random.shuffle(extra_dom)
            remaining_pool.extend(extra_dom)
        remaining_pool = remaining_pool[:need]
        _safe_extend(remaining_pool)
    # recortar y randomizar orden final
    final_ids = combined_ids[:target_n]
    random.shuffle(final_ids)

    final_items_to_save = []
    final_rec_items = []

    # Prepara el DataFrame para buscar solo los ítems necesarios
    final_ids_set = set(final_ids)

    # 1. Obtener todas las filas de ítems de una sola vez (más eficiente)
    # Asumimos que la columna de IDs es "itemId"
    final_items_df = items_df[items_df["itemId"].isin(final_ids_set)]

    for iid in final_ids:
        try:
            # Aseguramos que 'row' es la Serie de Pandas correspondiente al ítem
            row = final_items_df[final_items_df["itemId"] == iid].iloc[0]

        except IndexError:
            # Esto maneja el caso de que el ID no se encuentre en el DataFrame
            logger.warning(f"Item ID {iid} not found in items_df during final grid build.")
            continue  # Saltar este ítem si no se encuentra

        # Utilizamos la función row_to_recitem y el model_dump
        rec_item = row_to_recitem(row, distance=0.0)
        final_rec_items.append(rec_item)

        domain = row["domain"]  # Acceso seguro a la Serie de Pandas
        score = 0.0

        try:
            if domain == "movie":
                raw_score = row.get("imdb_score")

                imdb_score = float(raw_score) if raw_score is not None and isinstance(raw_score, (int, float)) else 0.0
                score = imdb_score / 2.0 if imdb_score else 0.0  # Escalando de 10 a 5

            elif domain == "book":
                # ✅ CORRECCIÓN: Acceso directo a la Serie de Pandas
                raw_score = row.get("google_avg_rating")

                # Conversión segura
                score = float(raw_score) if raw_score is not None and isinstance(raw_score, (int, float)) else 0.0

            elif domain == "music":
                # ✅ CORRECCIÓN: Acceso directo a la Serie de Pandas
                raw_listeners = row.get("listeners")
                listeners = float(raw_listeners) if raw_listeners is not None and isinstance(raw_listeners,
                                                                                             (int, float)) else 1.0

                if listeners <= 0: listeners = 1.0
                score = math.log10(listeners)
                score = min(5.0, score / 1.6)

            if not math.isfinite(score):
                score = 0.0

        except Exception as score_e:
            logger.error(f"Error calculating score for {iid} in {domain}: {score_e}")
            score = 0.0

        item_data_for_mongo = rec_item.model_dump()
        item_data_for_mongo["quality_score"] = round(score, 2)
        duration_hours = TIME_ESTIMATES.get(domain, 0.0)
        item_data_for_mongo["duration_hours"] = round(duration_hours, 2)
        final_items_to_save.append(item_data_for_mongo)
        # ----------------------------------------------------
        # ✅ 1. CALCULAR Y ALMACENAR EL PROMEDIO DE LA SESIÓN FINAL
        # ----------------------------------------------------
        total_quality = sum(item["quality_score"] for item in final_items_to_save)
        count = len(final_items_to_save)

        # Este es el promedio de calidad de los 20 ítems finales (el valor que quieres)
        session_avg_quality = round(total_quality / count, 4) if count > 0 else 0.0

        # persistir en la sesión (guardamos la lista con scores Y EL PROMEDIO)
        sessions_col.update_one(
            {"session_id": session_id},
            {"$set": {
                "final_grid": final_items_to_save,
                "session_avg_quality_score": session_avg_quality  # 👈 NUEVO CAMPO CRUCIAL
            }}
        )
    logger.info("build_final_grid user=%s session=%s domain=%s -> %d items (likes=%d, hidden=%d, underdogs=%d)",
                user_id, session_id, domain, len(final_rec_items), len(positives), len(hidden_ids), len(underdog_ids))
    # Devolvemos la lista original de RecItems (sin el score) a la app
    return final_rec_items
# Constantes para estimación de tiempo en HORAS
TIME_ESTIMATES = {
    "movie": 1.75,  # 1 h 45 m en promedio por película
    "book": 6.0,  # 6h en promedio por libro
    "music": 0.058,  # 3.5 minutos en promedio por canción
    "interaction_seconds": 30 / 3600  # 30 segundos por interacción, convertido a horas
}

def get_user_dashboard_stats(user_id: str) -> UserDashboardStats:
    # 1. Pipeline de Agregación de MongoDB
    pipeline = [
        {
            '$match': {'user_id': user_id}
        },
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
                                                'vars': {'feedback_obj': {
                                                    '$arrayElemAt': [{'$objectToArray': '$$feedback_target'}, 0]}},
                                                'in': '$$feedback_obj.v'
                                            }}},
                                            {'$cond': [
                                                {'$in': [{'$type': '$$feedback_target'},
                                                         ['array', 'null', 'undefined']]},
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
                        {'$and': [
                            '$finished',
                            {'$gt': [{'$size': {'$ifNull': ['$final_grid', []]}}, 0]}
                        ]},
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
                        '$cond': [
                            '$finished',
                            {'$size': {'$ifNull': ['$final_grid', []]}},
                            0
                        ]
                    }
                },
                'sum_of_avg_scores': {'$sum': '$avg_grid_score'},
                'sessions_with_scores': {'$sum': {'$cond': ['$avg_grid_score', 1, 0]}}
            }
        }
    ]
    results = list(sessions_col.aggregate(pipeline))
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

    # 3. Calcular los totales globales
    for stats in domain_stats_map.values():
        total_stats['total_sessions'] += stats.total_sessions
        total_stats['finished_sessions'] += stats.finished_sessions
        total_stats['total_items_interacted'] += stats.total_items_shown
        total_stats['total_items_liked'] += stats.items_liked
        total_stats['total_items_rejected'] += stats.items_rejected
        total_stats['total_final_recs_generated'] += stats.final_recs_generated
        total_stats['total_hours_interacting'] += stats.time_stats.hours_interacting
        total_stats['total_hours_from_final_recs'] += stats.time_stats.hours_from_final_recs

    # ✅ CORRECCIÓN: Evitar división por cero para el promedio global
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
def api_get_user_dashboard_stats(user_id: str = Depends(get_user_id_from_jwt)):
    return get_user_dashboard_stats(user_id)

# ---------- Session endpoints (nuevo flujo) ----------
def create_session(user_id: str, domain: str) -> Tuple[str, RecItem]:

    # 1. Obtener fecha UTC actual
    now = datetime.now(timezone.utc)
    today_utc_str = now.strftime('%Y-%m-%d')  # Formato YYYY-MM-DD

    daily_count = sessions_col.count_documents({
        "user_id": user_id,
        "created_date_utc": today_utc_str
    })

    if daily_count >= SESSION_DAILY_LIMIT:
        logger.warning(f"Límite diario alcanzado para usuario {user_id}. Count={daily_count}")
        raise HTTPException(
            status_code=429,  # Too Many Requests
            detail=f"Has alcanzado el límite de {SESSION_DAILY_LIMIT} sesiones por día."
        )


    session_id = str(uuid4())
    seed = generate_new_seed(domain)
    now = datetime.now(timezone.utc)
    sessions_col.insert_one({
        "session_id": session_id,
        "user_id": user_id,
        "domain": domain,
        "created_at": now,
        "created_date_utc": today_utc_str,  # Solo la fecha YYYY-MM-DD para contar
        "last_item_id": seed.item_id,
        "iterations": 0,
        "limit": SESSION_ITER_LIMIT,
        "finished": False,
        "history": [(seed.item_id, 0)],  # guardamos la seed inicial como neutral
        "shown": [seed.item_id]
    })
    session_feedback_col.insert_one({"session_id": session_id, "item_id": seed.item_id, "feedback": 0, "ts": now})
    return session_id, seed
@app.get("/user/final-grid/{domain}", response_model=FinalListResponse)
def get_final_grid_for_domain(domain: str, user_id: str = Depends(get_user_id_from_jwt)):
    # Buscar la última sesión finalizada del usuario en ese dominio
    session = db.sessions.find_one(
        {"user_id": user_id, "domain": domain, "finished": True},
        sort=[("created_at", -1)]
    )
    if not session or "final_grid" not in session:
        raise HTTPException(404, "No hay grid final para este usuario y dominio")
    recs = [RecItem(**item) for item in session["final_grid"]]
    return FinalListResponse(recommendations=recs)

def get_session(session_id: str):
    return sessions_col.find_one({"session_id": session_id})

def get_session_history(session_id: str) -> List[Tuple[str, int]]:
    s = get_session(session_id)
    if not s:
        return []
    history = []
    for x in s.get("history", []):
        try:
            item_id = str(x[0])
            feedback = int(x[1])
            history.append((item_id, feedback))
        except (IndexError, ValueError, TypeError):
            continue  # saltar elementos corruptos
    return history
def reset_session(session_id: str):
    sessions_col.update_one({"session_id": session_id}, {"$set": {
        "iterations": 0,
        "finished": False,
        "history": [],
        "shown": [],
        "final_grid": None
    }})
    session_feedback_col.delete_many({"session_id": session_id})

@app.post("/session/{domain}/create", response_model=SessionCreateResponse)
def api_create_session(domain: Domain, user_id: str = Depends(get_user_id_from_jwt)):
    dom = domain.value
    session_id, seed = create_session(user_id, dom)
    return SessionCreateResponse(session_id=session_id, seed=seed)

@app.get("/session/{session_id}", response_model=SessionStateResponse)
def api_get_session(session_id: str, user_id: str = Depends(get_user_id_from_jwt)):
    s = get_session(session_id)
    if not s or s["user_id"] != user_id:
        raise HTTPException(404, "Session not found or unauthorized")
    last_item = None
    last_item_id = s.get("last_item_id")
    if last_item_id and last_item_id in movieid_to_index:
        row = items_df.loc[movieid_to_index[last_item_id]]
        last_item = row_to_recitem(row, distance=0.0)
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
def api_session_feedback(session_id: str, req: FeedbackRequest, user_id: str = Depends(get_user_id_from_jwt)):
    s = get_session(session_id)
    if not s or s["user_id"] != user_id:
        raise HTTPException(404, "Session not found or unauthorized")
    domain = s["domain"]
    history = s.get("history", [])
    shown: List[str] = list(s.get("shown", []))
    limit = int(s.get("limit", SESSION_ITER_LIMIT))

    # validar que el item pertenece al dominio y existe
    if req.item_id not in movieid_to_index:
        raise HTTPException(404, "Item no encontrado")
    row = items_df.loc[movieid_to_index[req.item_id]]
    if str(row["domain"]) != domain:
        raise HTTPException(400, "El item no corresponde al dominio de la sesión")
    if not history or history[-1][0] != req.item_id or history[-1][1] != req.feedback:
        history.append((req.item_id, req.feedback))
    if not shown or shown[-1] != req.item_id:
        shown.append(req.item_id)
    iterations = len(shown)  # 1 ítem mostrado == 1 iteración
    finished = iterations >= limit
    if finished:
        sessions_col.update_one(
            {"session_id": session_id},
            {"$set": {
                "history": history, "shown": shown,
                "iterations": iterations, "finished": True
            }}
        )
        return SeedResponse(seed_item=None)
    # si no terminó, calcular siguiente seed
    new_seed = compute_next_seed_from_history(history, domain)
    if not new_seed:
        # si no hay candidatos, marcamos como terminada y el front pedirá el grid final
        sessions_col.update_one(
            {"session_id": session_id},
            {"$set": {"history": history, "shown": shown, "iterations": iterations, "finished": True}}
        )
        return SeedResponse(seed_item=None)
    sessions_col.update_one(
        {"session_id": session_id},
        {"$set": {
            "history": history, "shown": shown, "iterations": iterations,
            "last_item_id": new_seed.item_id, "finished": False
        }}
    )
    session_feedback_col.insert_one({"session_id": session_id, "item_id": req.item_id, "feedback": req.feedback, "ts": datetime.now(timezone.utc)})
    return SeedResponse(seed_item=new_seed)

@app.post("/session/{session_id}/reset", response_model=SeedResponseWithSessionId)
def api_session_reset(session_id: str, user_id: str = Depends(get_user_id_from_jwt)):
    s = get_session(session_id)
    if not s or s["user_id"] != user_id:
        raise HTTPException(404, "Session not found or unauthorized")
    # Generar un nuevo session_id
    new_session_id = str(uuid4())
    # Reiniciar la sesión antigua en DB
    reset_session(session_id)
    # Generar seed para la nueva sesión
    seed = generate_new_seed(s["domain"])
    # Insertar nueva sesión en la DB con el nuevo session_id
    sessions_col.insert_one({
        "session_id": new_session_id,
        "user_id": user_id,
        "domain": s["domain"],
        "iterations": 0,
        "finished": False,
        "history": [(seed.item_id, 0)],
        "shown": [seed.item_id],
        "final_grid": None,
        "last_item_id": seed.item_id,
        "created_at": datetime.now(timezone.utc)
    })
    session_feedback_col.insert_one({
        "session_id": new_session_id,
        "item_id": seed.item_id,
        "feedback": 0,
        "ts": datetime.now(timezone.utc)
    })
    return SeedResponseWithSessionId(session_id=new_session_id, seed_item=seed)

@app.get("/session/{session_id}/final-grid", response_model=FinalListResponse)
def api_get_final_grid(session_id: str, user_id: str = Depends(get_user_id_from_jwt)):
    s = get_session(session_id)
    if not s or s["user_id"] != user_id:
        raise HTTPException(404, "Session not found or unauthorized")
    if not bool(s.get("finished", False)) and int(s.get("iterations", 0)) < int(s.get("limit", SESSION_ITER_LIMIT)):
        raise HTTPException(400, "Session not finished yet")
    # si ya existe final_grid → devolverlo
    if "final_grid" in s and s["final_grid"]:
        return FinalListResponse(recommendations=[RecItem(**i) for i in s["final_grid"]])
    final_items = build_final_grid(
        session_id=session_id,
        user_id=user_id,
        domain=s["domain"],
        history = s.get("history", []),
        target_n=TARGET_FINAL_N,
        diversity_threshold=DIVERSITY_JACCARD_THRESHOLD
    )
    return FinalListResponse(recommendations=final_items)


@app.post("/session/{session_id}/finalize", response_model=FinalListResponse)
def api_session_finalize(session_id: str, user_id: str = Depends(get_user_id_from_jwt)):
    s = get_session(session_id)
    if not s or s["user_id"] != user_id:
        raise HTTPException(404, "Session not found or unauthorized")


    sessions_col.update_one(
        {"session_id": session_id},
        {"$set": {"finished": True}}
    )
    final_response: FinalListResponse = api_get_final_grid(session_id, user_id)

    session_doc = sessions_col.find_one({"session_id": session_id})
    if not session_doc:
        raise HTTPException(500, "Session data missing after finalization.")

    session_avg_quality = session_doc.get("session_avg_quality_score", 0.0)

    response_data = final_response.model_dump()
    response_data["session_avg_quality"] = session_avg_quality

    # Opción 2: Usar .model_copy (Pydantic V2) o .copy(update={...}) (Pydantic V1)
    # Usaremos la Opción 1 con la reconstrucción por seguridad:
    return FinalListResponse(**response_data)

def search_items(query: str, limit: int = 20) -> List[SearchResultItem]:
    """
    Searches items_df for titles containing the query (case-insensitive).
    Returns a list of matching items formatted as SearchResultItem.
    """
    if not query or len(query) < 2: # Basic validation
        return []

    # Case-insensitive search on the 'title' column
    # We use .str.contains() for partial matches
    matches = items_df[items_df['title'].str.contains(query, case=False, na=False)]

    # Limit the number of results
    matches = matches.head(limit)

    results = []
    for _, row in matches.iterrows():
        # Use row_to_recitem to get consistent image URL handling (incl. placeholders)
        rec_item = row_to_recitem(row, distance=0.0) # distance is irrelevant here
        results.append(SearchResultItem(
            item_id=row['itemId'],
            title=row['title'],
            domain=row['domain'],
            image_url=rec_item.image_url # Get the potentially cleaned/fetched image URL
        ))
    return results

def get_item_details(item_id: str) -> Optional[ItemDetailResponse]:
    """
    Finds an item by its ID in items_df and returns detailed information.
    """
    # Find the row corresponding to the item_id
    item_row = items_df[items_df['itemId'] == item_id]

    if item_row.empty:
        return None # Item not found

    row = item_row.iloc[0] # Get the first (and only) row as a Series

    # Use row_to_recitem to get basic info + cleaned image URL
    rec_item = row_to_recitem(row, distance=0.0)

    # Extract additional details based on domain (adapt field names to YOUR dataset)
    genres_list = None


    genres_data = row.get("genres") # Assuming 'genres' column exists
    if isinstance(genres_data, str):
        genres_list = [g.strip() for g in genres_data.split('|') if g.strip()]
    elif isinstance(genres_data, (list, np.ndarray)):
         genres_list = [str(g).strip() for g in genres_data if str(g).strip()]

    year_match = re.search(r"\((\d{4})\)", row['title'])
    year = year_match.group(1) if year_match else row.get("year_str") # Reuse year extraction if available

    # Build the detailed response
    details = ItemDetailResponse(
        item_id=rec_item.item_id,
        title=rec_item.title,
        distance=rec_item.distance,
        image_url=rec_item.image_url,
        genres=genres_list,
        year=year,
        # Add domain-specific fields if they exist in your items_df
        artist=row.get("artist"), # Will be None if column doesn't exist or is empty
        # Usa _safe_int porque lo cambiaste a Optional[int]
        google_avg_rating=_safe_int(row.get("google_avg_rating")),

        # Usa _safe_float para campos float
        imdb_score=_safe_float(row.get("imdb_score")),

        # Usa _safe_int para campos int
        listeners=_safe_int(row.get("listeners"))
    )

    return details


@app.get("/search", response_model=SearchResponse)
def api_search_items(query: str, limit: int = 20, user_id: str = Depends(get_user_id_from_jwt)):
    """
    Endpoint to search for items by title query. Requires authentication.
    """
    if len(query) < 3:
        raise HTTPException(status_code=400, detail="Query must be at least 3 characters long")

    results = search_items(query, limit)
    return SearchResponse(results=results)


# ----------------------------------------
# ENDPOINTS DE LISTAS DE USUARIO
# ----------------------------------------

@app.post("/lists", response_model=UserListBasic)
def api_create_list(req: ListCreateRequest, user_id: str = Depends(get_user_id_from_jwt)):
    """Crea una nueva lista para el usuario."""
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
        result = user_lists_col.insert_one(new_list)
        return UserListBasic(
            list_id=str(result.inserted_id),
            name=req.name,
            item_count=0,

            icon_name=req.icon_name,
            color_hex=req.color_hex
        )
    except Exception as err:
        logger.error(f"Error al crear lista: {err}")
        # Error 11000 es duplicado en Mongo
        if "E11000" in str(err):
            raise HTTPException(status_code=400, detail="Ya existe una lista con ese nombre")
        raise HTTPException(status_code=500, detail="Error interno al crear la lista")


@app.get("/lists", response_model=List[UserListBasic])
def api_get_my_lists(
    archived: Optional[bool] = Query(None, description="Filtrar por estado archivado (true/false)"),
    user_id: str = Depends(get_user_id_from_jwt)
):
    """Obtiene listas de un usuario, opcionalmente filtradas por estado archivado."""
    # 🎯 Explicitly type the query dictionary to accept Any value type
    query: Dict[str, Any] = {"user_id": user_id}

    if archived is True:
        # If explicitly asking for archived, only get those where is_archived is true
        query["is_archived"] = True
    elif archived is False:
        # If explicitly asking for non-archived, get those where is_archived is false OR the field doesn't exist
        query["$or"] = [
            {"is_archived": False},
            {"is_archived": {"$exists": False}}
        ]
    else:  # archived is None (default case when calling from UserProfileViewModel)
        # Default: Get non-archived (is_archived is false OR field doesn't exist)
        query["$or"] = [
            {"is_archived": False},
            {"is_archived": {"$exists": False}}
        ]

    lists_cursor = user_lists_col.find(query).sort("created_at", -1)
    results = []
    for list_doc in lists_cursor:
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
def api_add_item_to_list(list_id: str, req: ItemAddRequest, user_id: str = Depends(get_user_id_from_jwt)):
    """Añade un item_id a una lista. Es 'idempotente' (no añade duplicados)."""

    # 1. Validar que el item existe
    if req.item_id not in movieid_to_index:
        raise HTTPException(status_code=404, detail="Item no encontrado en el catálogo")

    # 2. Añadir a la lista (usando $addToSet para evitar duplicados)
    try:
        result = user_lists_col.update_one(
            {"_id": ObjectId(list_id), "user_id": user_id},
            {"$addToSet": {"items": req.item_id}}
        )
    except Exception as err:
        raise HTTPException(status_code=400, detail="ID de lista inválido")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Lista no encontrada o no pertenece al usuario")

    # 3. Devolver el estado actualizado de la lista
    updated_doc = user_lists_col.find_one({"_id": ObjectId(list_id)})
    return UserListBasic(
        list_id=str(updated_doc["_id"]),
        name=updated_doc.get("name"),
        icon_name=updated_doc.get("icon_name", "default"),
        color_hex=updated_doc.get("color_hex", "#FFFFFF"),
        item_count=len(updated_doc.get("items", []))
    )


# --- ENDPOINTS ADICIONALES ---

@app.get("/lists/{list_id}", response_model=UserListDetail)
def api_get_list_details(list_id: str, user_id: str = Depends(get_user_id_from_jwt)):
    """Obtiene una lista específica, incluyendo todos sus items."""
    try:
        list_doc = user_lists_col.find_one({"_id": ObjectId(list_id), "user_id": user_id})
    except Exception:
        raise HTTPException(status_code=400, detail="ID de lista inválido")

    if not list_doc:
        raise HTTPException(status_code=404, detail="Lista no encontrada o no pertenece al usuario")

    item_ids = list_doc.get("items", [])
    item_details_list = []

    # Buscamos los detalles de cada item_id
    for item_id in item_ids:
        if item_id in movieid_to_index:
            row = items_df.loc[movieid_to_index[item_id]]
            rec_item = row_to_recitem(row, distance=0.0)  # Reutilizamos la función que ya obtiene imágenes
            item_details_list.append(SearchResultItem(
                item_id=rec_item.item_id,
                title=rec_item.title,
                domain=row.get("domain", "unknown"),
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
def api_update_list(list_id: str, req: ListUpdateRequest, user_id: str = Depends(get_user_id_from_jwt)):
    """Actualiza nombre, icono y color de una lista."""  # <-- Descripción actualizada
    if not req.name or len(req.name) < 1:
        raise HTTPException(status_code=400, detail="El nombre de la lista no puede estar vacío")

    # 🎯 CREA EL DICCIONARIO CON LOS 3 CAMPOS
    update_data = {
        "name": req.name,
        "icon_name": req.icon_name,
        "color_hex": req.color_hex
    }

    try:
        result = user_lists_col.update_one(
            {"_id": ObjectId(list_id), "user_id": user_id},
            {"$set": update_data}  # <-- USA EL DICCIONARIO COMPLETO
        )
    except Exception:
        raise HTTPException(status_code=400, detail="ID de lista inválido")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Lista no encontrada o no pertenece al usuario")

    updated_doc = user_lists_col.find_one({"_id": ObjectId(list_id)})
    # 🎯 DEVUELVE EL UserListBasic COMPLETO
    return UserListBasic(
        list_id=str(updated_doc["_id"]),
        name=updated_doc.get("name"),
        icon_name=updated_doc.get("icon_name", "default"),  # Añade valores por defecto
        color_hex=updated_doc.get("color_hex", "#FFFFFF"),  # Añade valores por defecto
        item_count=len(updated_doc.get("items", []))
    )


@app.delete("/lists/{list_id}", status_code=HTTP_204_NO_CONTENT)
def api_delete_list(list_id: str, user_id: str = Depends(get_user_id_from_jwt)):
    """Elimina una lista completa."""
    try:
        result = user_lists_col.delete_one(
            {"_id": ObjectId(list_id), "user_id": user_id}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="ID de lista inválido")

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Lista no encontrada o no pertenece al usuario")

    # Si tiene éxito, devuelve un 204 No Content
    return Response(status_code=HTTP_204_NO_CONTENT)


@app.delete("/lists/{list_id}/items/{item_id}", response_model=UserListBasic)
def api_remove_item_from_list(list_id: str, item_id: str, user_id: str = Depends(get_user_id_from_jwt)):
    """Elimina un solo item de una lista."""
    try:
        result = user_lists_col.update_one(
            {"_id": ObjectId(list_id), "user_id": user_id},
            {"$pull": {"items": item_id}}  # $pull elimina el item del array
        )
    except Exception:
        raise HTTPException(status_code=400, detail="ID de lista inválido")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Lista no encontrada o no pertenece al usuario")

    if result.modified_count == 0:
        # Esto no es un error, solo significa que el item no estaba en la lista
        pass

    updated_doc = user_lists_col.find_one({"_id": ObjectId(list_id)})
    return UserListBasic(
        list_id=str(updated_doc["_id"]),
        name=updated_doc.get("name"),
        icon_name=updated_doc.get("icon_name", "default"),
        color_hex=updated_doc.get("color_hex", "#FFFFFF"),
        item_count=len(updated_doc.get("items", []))
    )
@app.get("/item/{item_id}", response_model=ItemDetailResponse)
def api_get_item_details(item_id: str, user_id: str = Depends(get_user_id_from_jwt)):
    """
    Endpoint to get detailed information for a specific item_id. Requires authentication.
    """
    details = get_item_details(item_id)
    if details is None:
        raise HTTPException(status_code=404, detail=f"Item with ID '{item_id}' not found")
    return details


@app.get("/user/usage", response_model=UserUsageStatus)
def api_get_user_usage(user_id: str = Depends(get_user_id_from_jwt)):
    """Devuelve el estado de uso de sesiones diarias del usuario."""
    today_utc_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    sessions_today = sessions_col.count_documents({
        "user_id": user_id,
        "created_date_utc": today_utc_str
    })

    remaining = max(0, SESSION_DAILY_LIMIT - sessions_today)

    return UserUsageStatus(
        daily_limit=SESSION_DAILY_LIMIT,
        sessions_today=sessions_today,
        remaining_today=remaining
    )

# --- Nuevos Endpoints (añadir cerca de los otros endpoints de listas) ---

@app.put("/lists/{list_id}/archive", response_model=UserListBasic)
def api_archive_list(list_id: str, user_id: str = Depends(get_user_id_from_jwt)):
    """Marca una lista como archivada."""
    try:
        result = user_lists_col.update_one(
            {"_id": ObjectId(list_id), "user_id": user_id},
            {"$set": {"is_archived": True}}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="ID de lista inválido")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Lista no encontrada o no pertenece al usuario")

    updated_doc = user_lists_col.find_one({"_id": ObjectId(list_id)})
    # Devuelve el estado actualizado completo
    return UserListBasic(
        list_id=str(updated_doc["_id"]),
        name=updated_doc.get("name"),
        icon_name=updated_doc.get("icon_name", "default"),
        color_hex=updated_doc.get("color_hex", "#FFFFFF"),
        item_count=len(updated_doc.get("items", [])),
        is_archived=updated_doc.get("is_archived", False) # Incluir estado archivado
    )

@app.put("/lists/{list_id}/unarchive", response_model=UserListBasic)
def api_unarchive_list(list_id: str, user_id: str = Depends(get_user_id_from_jwt)):
    """Desmarca una lista como archivada."""
    try:
        result = user_lists_col.update_one(
            {"_id": ObjectId(list_id), "user_id": user_id},
            # Puedes usar $set o $unset, $set es más simple si siempre existe
            {"$set": {"is_archived": False}}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="ID de lista inválido")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Lista no encontrada o no pertenece al usuario")

    updated_doc = user_lists_col.find_one({"_id": ObjectId(list_id)})
    # Devuelve el estado actualizado completo
    return UserListBasic(
        list_id=str(updated_doc["_id"]),
        name=updated_doc.get("name"),
        icon_name=updated_doc.get("icon_name", "default"),
        color_hex=updated_doc.get("color_hex", "#FFFFFF"),
        item_count=len(updated_doc.get("items", [])),
        is_archived=updated_doc.get("is_archived", False) # Incluir estado archivado
    )
# --- Nuevos Endpoints (añadir cerca de los otros endpoints de listas/final) ---

@app.post("/favorites/{item_id}", status_code=201) # 201 Created
def api_add_favorite(item_id: str, user_id: str = Depends(get_user_id_from_jwt)):
    """Añade un item a los favoritos del usuario."""
    if item_id not in movieid_to_index:
        raise HTTPException(status_code=404, detail="Item no encontrado en el catálogo")

    now = datetime.now(timezone.utc)
    favorite_doc = {
        "user_id": user_id,
        "item_id": item_id,
        "added_at": now
    }
    try:
        user_favorites_col.insert_one(favorite_doc)
        return {"message": "Item añadido a favoritos"}
    except Exception as e: # Captura error de duplicado (índice único)
        if "E11000" in str(e):
             # No es un error si ya existe, es idempotente
            return {"message": "Item ya estaba en favoritos"}
        logger.error(f"Error añadiendo favorito: {e}")
        raise HTTPException(status_code=500, detail="Error interno al añadir favorito")

@app.delete("/favorites/{item_id}", status_code=HTTP_204_NO_CONTENT)
def api_remove_favorite(item_id: str, user_id: str = Depends(get_user_id_from_jwt)):
    """Elimina un item de los favoritos del usuario."""
    result = user_favorites_col.delete_one({"user_id": user_id, "item_id": item_id})
    if result.deleted_count == 0:
        # No encontrado, pero la operación es idempotente (el estado deseado es "no favorito")
        pass
    return Response(status_code=HTTP_204_NO_CONTENT)

@app.get("/favorites", response_model=List[SearchResultItem])
def api_get_favorites(user_id: str = Depends(get_user_id_from_jwt)):
    """Obtiene todos los items favoritos de un usuario."""
    favorites_cursor = user_favorites_col.find({"user_id": user_id}).sort("added_at", -1)
    favorite_item_ids = [doc["item_id"] for doc in favorites_cursor]

    results = []
    for item_id in favorite_item_ids:
        if item_id in movieid_to_index:
            row = items_df.loc[movieid_to_index[item_id]]
            rec_item = row_to_recitem(row, distance=0.0)
            results.append(SearchResultItem(
                item_id=rec_item.item_id,
                title=rec_item.title,
                domain=row.get("domain", "unknown"),
                image_url=rec_item.image_url
            ))
    return results

@app.get("/favorites/status/{item_id}", response_model=FavoriteStatusResponse)
def api_get_favorite_status(item_id: str, user_id: str = Depends(get_user_id_from_jwt)):
    """Verifica si un item específico está en los favoritos del usuario."""
    count = user_favorites_col.count_documents({"user_id": user_id, "item_id": item_id})
    return FavoriteStatusResponse(item_id=item_id, is_favorite=(count > 0))

@app.get("/health")
def health():
    return {"status": "ok"}