# services/recs_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import numpy as np
from annoy import AnnoyIndex

app = FastAPI(title="Recommendation Service")

# Rutas base
BASE_DIR = Path(__file__).resolve().parents[1]
VECT_DIR = BASE_DIR / "data" / "vectorized"

# 1. Carga catálogo y construye map itemId→índice
items_df = pd.read_parquet(VECT_DIR / "items.parquet")
movieid_to_index = {
    row["itemId"]: idx
    for idx, row in items_df.iterrows()
}

# 2. Carga embeddings
embeds = np.load(VECT_DIR / "items_embeds.npz")["embeds"]

# 3. Carga índice Annoy
dim = embeds.shape[1]
ann_index = AnnoyIndex(dim, metric="angular")
ann_index.load(str(VECT_DIR / "items_index.ann"))

# 4. Pydantic models
class RecItem(BaseModel):
    item_id: str
    title: str
    distance: float
    image_url: str | None = None

class RecommendRequest(BaseModel):
    item_id: str
    top_n: int = 5

class RecommendResponse(BaseModel):
    item_id: str
    recommendations: list[RecItem]


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    # 1. Validar item_id
    if req.item_id not in movieid_to_index:
        raise HTTPException(404, f"item_id '{req.item_id}' no encontrado")

    idx = movieid_to_index[req.item_id]
    # 2. Consultar Annoy (K+1 porque incluye a sí mismo)
    neigh_idxs, dists = ann_index.get_nns_by_item(
        idx, req.top_n + 1, include_distances=True
    )

    recs = []
    for neigh_idx, dist in zip(neigh_idxs[1:], dists[1:]):
        row = items_df.iloc[neigh_idx]
        recs.append(RecItem(
            item_id=row["itemId"],
            title=row["title"],
            distance=dist,
            image_url=row.get("image_url", None)
        ))

    return RecommendResponse(item_id=req.item_id, recommendations=recs)
