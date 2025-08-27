# services/recs_api.py
import os
import sys
import logging
from enum import Enum
from uuid import uuid4
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv
import math
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from pymongo import MongoClient
import jwt
from jwt.exceptions import InvalidTokenError

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

# ---------- app ----------
app = FastAPI(title="Recommendation Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restringir en producción
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ---------- Mongo (colecciones) ----------
mongo = MongoClient(MONGO_URI)
db = mongo.get_database()
feedback_col = db.get_collection("feedback")
sessions_col = db.get_collection("sessions")
session_feedback_col = db.get_collection("session_feedback")

# crear índices defensivos una sola vez
try:
    feedback_col.create_index([("user_id",1),("domain",1),("item_id",1)], unique=True)
except Exception as e:
    logger.warning("No se pudo crear indice unico feedback_col: %s", e)

# ---------- assets vectorizados (items + embeddings + annoy) ----------
BASE_DIR = Path(__file__).resolve().parents[1]
VECT_DIR = BASE_DIR / "data" / "vectorized"

items_df = pd.read_parquet(VECT_DIR / "items.parquet").reset_index(drop=True)
movieid_to_index = { item: i for i, item in enumerate(items_df["itemId"].tolist()) }

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
    seed_item: RecItem

# Session models
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

class SessionFeedbackRequest(BaseModel):
    item_id: str
    feedback: int

class FinalListResponse(BaseModel):
    recommendations: list[RecItem]

# ---------- Enum de dominios ----------
class Domain(str, Enum):
    movie = "movie"
    book = "book"
    music = "music"

# ---------- JWT helper ----------
def get_user_id_from_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload.get("userId") or payload.get("sub")
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")

# ---------- Mongo helpers (global feedback) ----------
def save_feedback(user_id: str, domain: str, item_id: str, feedback: int):
    """Upsert simple: guarda último feedback y timestamp (persistente)."""
    now = datetime.utcnow()
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

# pesos para scoring
ALPHA_SIM = 0.60    # similitud (desde Annoy / distancia)
BETA_POP = 0.30     # popularidad / imdb
GAMMA_IMDB = 0.10
DELTA_NOVELTY = 0.40

def generate_final_recommendations(session_history: List[Tuple[str,int]], domain: str, target_n: int = TARGET_FINAL_N) -> List[RecItem]:
    """
    Genera hasta `target_n` recomendaciones finales usando:
      - vecinos de los últimos positivos
      - un pool de exploración aleatoria
      - scoring combinado (sim, popularity, imdb)
      - selección greedy con restricción de diversidad por Jaccard de géneros
    """
    shown = {item for item, _ in session_history}
    positives = [item for item, fb in session_history if fb > 0]
    # Map item_id -> best (lowest) distance encountered
    candidates_scores = {}  # item_id -> best_distance

    # 1) vecinos de últimos positivos
    last_positives = positives[-CONSIDER_LAST_POSITIVES:] if positives else []
    for base in last_positives:
        idx = movieid_to_index.get(base)
        if idx is None:
            continue
        neigh_idxs, dists = ann_index.get_nns_by_item(idx, K_VECINOS, include_distances=True)
        for n_idx, dist in zip(neigh_idxs[1:], dists[1:]):
            row = items_df.iloc[n_idx]
            if row["domain"] != domain:
                continue
            cid = row["itemId"]
            if cid in shown:
                continue
            # almacenar el mejor (menor) dist
            prev = candidates_scores.get(cid)
            if prev is None or dist < prev:
                candidates_scores[cid] = float(dist)

    # 2) pool de exploración aleatoria
    pool = items_df[items_df["domain"] == domain]
    pool = pool[~pool["itemId"].isin(shown)]
    if not pool.empty:
        sample = pool.sample(min(EXPLORATION_SAMPLE, len(pool)))
        for _, row in sample.iterrows():
            cid = row["itemId"]
            if cid not in candidates_scores:
                candidates_scores[cid] = float(999.0)  # distancia grande para exploration

    if not candidates_scores:
        return []

    # 3) calcular scores finales
    candidates = []
    for cid, dist in candidates_scores.items():
        row = items_df.loc[movieid_to_index[cid]]
        sim_score = math.exp(-dist) if dist < 900 else 0.01  # si exploration usamos baja similitud
        pop = _get_popularity(row)
        imdb = float(row.get("imdb_score", 0.0)) / 10.0 if "imdb_score" in row else 0.0
        raw_score = ALPHA_SIM*sim_score + BETA_POP*pop + GAMMA_IMDB*imdb
        candidates.append({
            "item_id": cid,
            "score": raw_score,
            "dist": dist,
            "genres": _genres_to_set(row.get("genres")),
            "row": row
        })

    # ordenar por score desc
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # 4) selección greedy con diversidad
    selected = []
    selected_genres = []
    for c in candidates:
        if len(selected) >= target_n:
            break
        # calcular penalty por similitud de géneros respecto a selected
        cand_gen = c["genres"]
        max_j = 0.0
        for sg in selected_genres:
            max_j = max(max_j, jaccard(cand_gen, sg))
        novelty_penalty = max_j
        adjusted_score = c["score"] - DELTA_NOVELTY * novelty_penalty
        if adjusted_score <= -1e9:
            continue
        # aplicar umbral de diversidad
        if max_j < DIVERSITY_JACCARD_THRESHOLD or len(selected) < 2:
            # añadir
            row = c["row"]
            selected.append(RecItem(item_id=row["itemId"], title=row["title"], distance=0.0, image_url=row.get("image_url", None)))
            selected_genres.append(cand_gen)

    # 5) si no llegamos a target_n, rellenar con aleatorios fuera de shown+selected
    if len(selected) < target_n:
        remaining = items_df[items_df["domain"] == domain]
        exclude = shown.union({r.item_id for r in selected})
        remaining = remaining[~remaining["itemId"].isin(exclude)]
        if not remaining.empty:
            need = target_n - len(selected)
            sample = remaining.sample(min(need, len(remaining)))
            for _, row in sample.iterrows():
                selected.append(RecItem(item_id=row["itemId"], title=row["title"], distance=0.0, image_url=row.get("image_url", None)))

    # asegurar unicidad y cortar a target_n
    unique = []
    seen = set()
    for r in selected:
        if r.item_id in seen: continue
        seen.add(r.item_id)
        unique.append(r)
        if len(unique) >= target_n: break

    logger.debug("generate_final_recommendations: last_positives=%s shown_count=%d pool_size=%d", last_positives,
                 len(shown), len(pool))

    return unique

def generate_new_seed(domain: str) -> RecItem:
    candidates = items_df[items_df["domain"] == domain]
    if candidates.empty:
        raise HTTPException(404, "no hay items para el dominio")
    row = candidates.sample(1).iloc[0]
    return RecItem(item_id=row["itemId"], title=row["title"], distance=0.0, image_url=row.get("image_url"))
def _genres_to_set(genres):
    """Normaliza la columna genres (lista o string 'a|b')."""
    if genres is None:
        return set()
    if isinstance(genres, list):
        return set(genres)
    if isinstance(genres, str):
        return set([g.strip().lower() for g in genres.split("|") if g.strip()])
    return set()

def jaccard(a:set, b:set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni > 0 else 0.0

def _get_popularity(item_row):
    # fallback seguro si no existe la columna
    p = item_row.get("popularity") if "popularity" in item_row else None
    if p is None:
        # si tienes imdb_score y quieres usarlo:
        s = item_row.get("imdb_score") if "imdb_score" in item_row else None
        return float(s) / 10.0 if s is not None else 0.0
    return float(p)

# ---------- Fase 2: Filtrado + Boosting + Diversidad ----------
def _get_quality_score(row):
    """Devuelve un score de calidad dependiendo del dominio."""
    domain = row.get("domain")
    if domain == "movie":
        return float(row.get("imdb_score", 0) or 0)
    elif domain == "book":
        return float(row.get("google_rating", 0) or 0)
    elif domain == "music":
        return float(row.get("playcount", 0) or 0)
    return 0.0


def generate_diverse_recommendations(user_id: str, seen_items: list[str], top_per_domain: int = 5):
    """
    Genera recomendaciones simples por dominio:
      - filtra los items ya vistos
      - aplica boosting según calidad
      - selecciona top N por dominio
    """
    df = items_df.copy()
    df = df[~df["itemId"].isin(seen_items)]

    if df.empty:
        return []

    # agregar columna de calidad y boosted_score
    df["quality_score"] = df.apply(_get_quality_score, axis=1)
    df["boosted_score"] = (df.get("base_score", 1.0)) * (1 + df["quality_score"] / 10)

    # seleccionar top N diversificado por dominio
    recommendations = []
    for domain, group in df.groupby("domain"):
        top_items = group.nlargest(top_per_domain, "boosted_score")
        for _, row in top_items.iterrows():
            recommendations.append(
                RecItem(
                    item_id=row["itemId"],
                    title=row["title"],
                    distance=0.0,
                    image_url=row.get("image_url"),
                )
            )
    return recommendations

def create_session(user_id: str, domain: str) -> Tuple[str, RecItem]:
    session_id = str(uuid4())
    seed = generate_new_seed(domain)
    now = datetime.utcnow()
    sessions_col.insert_one({
        "session_id": session_id,
        "user_id": user_id,
        "domain": domain,
        "created_at": now,
        "last_item_id": seed.item_id,
        "iterations": 0,
        "limit": SESSION_ITER_LIMIT,
        "finished": False
    })
    session_feedback_col.insert_one({"session_id": session_id, "item_id": seed.item_id, "feedback": 0, "ts": now})
    return session_id, seed

def get_session(session_id: str):
    return sessions_col.find_one({"session_id": session_id})

def save_session_feedback(session_id: str, item_id: str, feedback: int):
    now = datetime.utcnow()
    session_feedback_col.insert_one({"session_id": session_id, "item_id": item_id, "feedback": feedback, "ts": now})
    sessions_col.update_one({"session_id": session_id}, {"$inc": {"iterations": 1}})

def get_session_history(session_id: str):
    docs = session_feedback_col.find({"session_id": session_id}).sort("ts", 1)
    return [(d["item_id"], d["feedback"]) for d in docs]

def reset_session(session_id: str):
    sessions_col.update_one({"session_id": session_id}, {"$set": {"iterations": 0, "finished": False}})
    session_feedback_col.delete_many({"session_id": session_id})



# ---------- Core algorithm: compute next seed from persisted history ----------
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
                    return RecItem(item_id=candidate_id, title=row["title"], distance=0.0, image_url=row.get("image_url"))
    # fallback: random outside shown
    candidates = items_df[items_df["domain"] == domain]
    candidates = candidates[~candidates["itemId"].isin(shown)]
    if not candidates.empty:
        row = candidates.sample(1).iloc[0]
        return RecItem(item_id=row["itemId"], title=row["title"], distance=0.0, image_url=row.get("image_url"))
    return None

def compute_next_seed_from_history(session_history: List[Tuple[str,int]], domain: str) -> Optional[RecItem]:
    shown = {item for item, _ in session_history}
    positives = [item for item, fb in session_history if fb > 0]
    negatives = {item for item, fb in session_history if fb < 0}

    if positives:
        base = positives[-1]
        idx = movieid_to_index.get(base)
        if idx is not None:
            neigh_idxs, _ = ann_index.get_nns_by_item(idx, 50, include_distances=True)
            for neigh_idx in neigh_idxs[1:]:
                row = items_df.iloc[neigh_idx]
                if row["domain"] == domain and row["itemId"] not in shown and row["itemId"] not in negatives:
                    return RecItem(item_id=row["itemId"], title=row["title"], distance=0.0, image_url=row.get("image_url"))
    # fallback random outside shown
    candidates = items_df[items_df["domain"] == domain]
    candidates = candidates[~candidates["itemId"].isin(shown)]
    if not candidates.empty:
        row = candidates.sample(1).iloc[0]
        return RecItem(item_id=row["itemId"], title=row["title"], distance=0.0, image_url=row.get("image_url"))
    return None


def _collect_candidates(domain: str, shown: set, positives: List[str],
                        k_neighbors: int = 50, exploration_sample: int = 200) -> dict:
    """
    Devuelve un dict item_id -> best_distance (menor) construyendo pool:
      - vecinos de los últimos positivos,
      - una muestra de exploración aleatoria del dominio.
    """
    candidates = {}
    # vecinos de positivos
    for base in positives:
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

    # pool de exploracion aleatoria
    pool = items_df[items_df["domain"] == domain]
    pool = pool[~pool["itemId"].isin(shown)]
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
    """
    Calcula score combinado y devuelve lista ordenada (desc) de dicts:
      {item_id, score, dist, genres, row}
    """
    scored = []
    for cid, dist in candidates.items():
        # row desde items_df
        try:
            row = items_df.loc[movieid_to_index[cid]]
        except Exception:
            continue
        # similitud: exponencial inversa de la distancia
        sim_score = math.exp(-dist) if dist < 900 else 0.01
        pop = _get_popularity(row)
        imdb = float(row.get("imdb_score", 0.0)) / 10.0 if "imdb_score" in row else 0.0
        raw = alpha_sim * sim_score + beta_pop * pop + gamma_imdb * imdb
        scored.append({
            "item_id": cid,
            "score": raw,
            "dist": dist,
            "genres": _genres_to_set(row.get("genres")),
            "row": row
        })
    # ordenar por score descendente
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def assemble_final_grid(user_id: str,
                        domain: str,
                        session_history: Optional[List[Tuple[str,int]]] = None,
                        target_n: int = TARGET_FINAL_N,
                        diversity_threshold: float = DIVERSITY_JACCARD_THRESHOLD) -> List[RecItem]:
    """
    Pipeline principal de Fase 3:
      1) Construye pool de candidatos (vecinos + exploración).
      2) Scoring combinado.
      3) Selección greedy con restricción de diversidad por Jaccard.
      4) Relleno aleatorio si no llega a target_n.
    Devuelve lista de RecItem (longitud <= target_n).
    """
    # shown items (si viene session_history priorizamos esa, sino historia global)
    if session_history is not None:
        shown = {item for item, _ in session_history}
        positives = [item for item, fb in session_history if fb > 0]
    else:
        hist = get_history(user_id, domain)
        shown = {item for item, _ in hist}
        positives = [item for item, fb in hist if fb > 0]

    # 1) pool
    candidates = _collect_candidates(domain, shown, positives, k_neighbors=K_VECINOS, exploration_sample=EXPLORATION_SAMPLE)
    if not candidates:
        return []

    # 2) scoring
    scored = _score_and_rank_candidates(candidates)

    # 3) greedy selection con diversidad
    selected = []
    selected_genres = []
    for c in scored:
        if len(selected) >= target_n:
            break
        cand_gen = c["genres"]
        max_j = 0.0
        for sg in selected_genres:
            max_j = max(max_j, jaccard(cand_gen, sg))
        # penalización por novedad (si demasiada superposición baja la prioridad)
        adjusted_score = c["score"] - DELTA_NOVELTY * max_j
        if max_j < diversity_threshold or len(selected) < 2:
            row = c["row"]
            selected.append(RecItem(item_id=row["itemId"], title=row["title"], distance=0.0, image_url=row.get("image_url", None)))
            selected_genres.append(cand_gen)

    # 4) rellenar si faltan
    if len(selected) < target_n:
        remaining = items_df[items_df["domain"] == domain]
        exclude = shown.union({r.item_id for r in selected})
        remaining = remaining[~remaining["itemId"].isin(exclude)]
        if not remaining.empty:
            need = target_n - len(selected)
            sample = remaining.sample(min(need, len(remaining)))
            for _, row in sample.iterrows():
                selected.append(RecItem(item_id=row["itemId"], title=row["title"], distance=0.0, image_url=row.get("image_url", None)))

    # asegurar unicidad
    unique = []
    seen = set()
    for r in selected:
        if r.item_id in seen:
            continue
        seen.add(r.item_id)
        unique.append(r)
        if len(unique) >= target_n:
            break

    logger.info("assemble_final_grid user=%s domain=%s -> returned=%d candidates=%d positives=%d shown=%d",
                user_id, domain, len(unique), len(candidates), len(positives), len(shown))
    return unique

# ---------- Endpoints: legacy recommend + seed + feedback + reset ----------
@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if req.item_id not in movieid_to_index:
        raise HTTPException(404, f"item_id '{req.item_id}' no encontrado")
    idx = movieid_to_index[req.item_id]
    neigh_idxs, dists = ann_index.get_nns_by_item(idx, req.top_n + 1, include_distances=True)
    recs = []
    for n_idx, dist in zip(neigh_idxs[1:], dists[1:]):
        row = items_df.iloc[n_idx]
        recs.append(RecItem(item_id=row["itemId"], title=row["title"], distance=dist, image_url=row.get("image_url")))
    return RecommendResponse(item_id=req.item_id, recommendations=recs)

@app.post("/recommend/diverse")
def recommend_diverse(user_id: str = Depends(get_user_id_from_jwt)):
    # obtener historial de Mongo (todos los dominios)
    seen_items = []
    for dom in ["movie", "book", "music"]:
        seen_items.extend([item_id for item_id, _ in get_history(user_id, dom)])

    recs = generate_diverse_recommendations(user_id, seen_items, top_per_domain=5)
    return {"user_id": user_id, "recommendations": recs}

@app.get("/seed/{domain}", response_model=SeedResponse)
def get_initial_seed(domain: Domain, user_id: str = Depends(get_user_id_from_jwt)):
    dom = domain.value
    history = get_history(user_id, dom)
    if history:
        last_item_id, _ = history[-1]
        if last_item_id in movieid_to_index:
            row = items_df.loc[movieid_to_index[last_item_id]]
            return SeedResponse(seed_item=RecItem(item_id=row["itemId"], title=row["title"], distance=0.0, image_url=row.get("image_url")))
        else:
            clear_history(user_id, dom)
    seed = generate_new_seed(dom)
    save_feedback(user_id, dom, seed.item_id, 0)
    return SeedResponse(seed_item=seed)

@app.post("/feedback/{domain}", response_model=SeedResponse)
def handle_feedback(domain: Domain, req: FeedbackRequest, user_id: str = Depends(get_user_id_from_jwt)):
    dom = domain.value
    save_feedback(user_id, dom, req.item_id, req.feedback)
    new_seed = compute_next_seed(user_id, dom)
    if new_seed is None:
        raise HTTPException(404, "No se pudo generar nuevo ítem semilla")
    last = feedback_col.find_one({"user_id": user_id, "domain": dom}, sort=[("ts", -1)])
    if not last or last.get("item_id") != new_seed.item_id:
        save_feedback(user_id, dom, new_seed.item_id, 0)
    return SeedResponse(seed_item=new_seed)

@app.post("/reset/{domain}", response_model=SeedResponse)
def reset_recs(domain: Domain, user_id: str = Depends(get_user_id_from_jwt)):
    dom = domain.value
    clear_history(user_id, dom)
    seed = generate_new_seed(dom)
    save_feedback(user_id, dom, seed.item_id, 0)
    return SeedResponse(seed_item=seed)

# ---------- Session endpoints (new flow) ----------
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
    if s.get("last_item_id"):
        idx = movieid_to_index.get(s["last_item_id"])
        if idx is not None:
            row = items_df.iloc[idx]
            last_item = RecItem(item_id=row["itemId"], title=row["title"], distance=0.0, image_url=row.get("image_url"))
    finished = s.get("finished", False) or (s.get("iterations", 0) >= s.get("limit", SESSION_ITER_LIMIT))
    return SessionStateResponse(session_id=session_id, domain=s["domain"], last_item=last_item, iterations=s.get("iterations", 0), limit=s.get("limit", SESSION_ITER_LIMIT), finished=finished)

@app.post("/session/{session_id}/feedback", response_model=SeedResponse)
def api_session_feedback(session_id: str, req: SessionFeedbackRequest, user_id: str = Depends(get_user_id_from_jwt)):
    s = get_session(session_id)
    if not s or s["user_id"] != user_id:
        raise HTTPException(404, "Session not found or unauthorized")
    domain = s["domain"]
    save_session_feedback(session_id, req.item_id, req.feedback)
    s_new = get_session(session_id)
    if s_new["iterations"] >= s_new.get("limit", SESSION_ITER_LIMIT):
        sessions_col.update_one({"session_id": session_id}, {"$set": {"finished": True}})
    session_history = get_session_history(session_id)
    new_seed = compute_next_seed_from_history(session_history, domain)
    if new_seed is None:
        raise HTTPException(404, "No se pudo generar nuevo seed")
    sessions_col.update_one({"session_id": session_id}, {"$set": {"last_item_id": new_seed.item_id}})
    session_feedback_col.insert_one({"session_id": session_id, "item_id": new_seed.item_id, "feedback": 0, "ts": datetime.utcnow()})
    return SeedResponse(seed_item=new_seed)

@app.post("/session/{session_id}/reset", response_model=SeedResponse)
def api_session_reset(session_id: str, user_id: str = Depends(get_user_id_from_jwt)):
    s = get_session(session_id)
    if not s or s["user_id"] != user_id:
        raise HTTPException(404, "Session not found or unauthorized")
    reset_session(session_id)
    seed = generate_new_seed(s["domain"])
    sessions_col.update_one({"session_id": session_id}, {"$set": {"last_item_id": seed.item_id}})
    session_feedback_col.insert_one({"session_id": session_id, "item_id": seed.item_id, "feedback": 0, "ts": datetime.utcnow()})
    return SeedResponse(seed_item=seed)

@app.post("/session/{session_id}/finalize", response_model=FinalListResponse)
def api_session_finalize(session_id: str, user_id: str = Depends(get_user_id_from_jwt)):
    s = get_session(session_id)
    if not s or s["user_id"] != user_id:
        raise HTTPException(404, "Session not found or unauthorized")
    # verificar que la sesión haya terminado (o forzar)
    if not s.get("finished") and s.get("iterations", 0) < s.get("limit", SESSION_ITER_LIMIT):
        raise HTTPException(400, "Session not finished yet")

    # obtener history de la sesión
    session_history = get_session_history(session_id)
    recs = assemble_final_grid(user_id, s["domain"], session_history=session_history, target_n=TARGET_FINAL_N)
    return FinalListResponse(recommendations=recs)

@app.get("/health")
def health():
    return {"status": "ok"}
