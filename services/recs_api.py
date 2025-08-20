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
    feedback_col.create_index([("user_id", 1), ("domain", 1), ("item_id", 1)], unique=False)
    sessions_col.create_index([("session_id", 1)], unique=True)
    session_feedback_col.create_index([("session_id", 1), ("ts", 1)])
except Exception as e:
    logger.warning("No se pudieron crear índices (ya existen?): %s", e)

# ---------- assets vectorizados (items + embeddings + annoy) ----------
BASE_DIR = Path(__file__).resolve().parents[1]
VECT_DIR = BASE_DIR / "data" / "vectorized"

items_df = pd.read_parquet(VECT_DIR / "items.parquet")
movieid_to_index = {row["itemId"]: idx for idx, row in items_df.iterrows()}

embeds = np.load(VECT_DIR / "items_embeds.npz")["embeds"]
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

def generate_new_seed(domain: str) -> RecItem:
    candidates = items_df[items_df["domain"] == domain]
    if candidates.empty:
        raise HTTPException(404, "no hay items para el dominio")
    row = candidates.sample(1).iloc[0]
    return RecItem(item_id=row["itemId"], title=row["title"], distance=0.0, image_url=row.get("image_url"))

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
@app.post("/session/{domain}", response_model=SessionCreateResponse)
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

@app.get("/health")
def health():
    return {"status": "ok"}
