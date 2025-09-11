import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter()

TMDB_API_KEY = "647d6ce7695cbea9dafc5c77378bac2b"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"


async def fetch_movie_poster(title: str) -> str | None:
    """
    Consulta TMDB por el título de la película y devuelve la URL del póster.
    """
    if not TMDB_API_KEY:
        raise RuntimeError("TMDB_API_KEY no está configurada")

    async with httpx.AsyncClient() as client:
        url = f"{TMDB_BASE_URL}/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": title}
        resp = await client.get(url, params=params)

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Error al consultar TMDB")

    data = resp.json()
    results = data.get("results", [])
    if not results:
        return None

    poster_path = results[0].get("poster_path")
    return f"{TMDB_IMG_BASE}{poster_path}" if poster_path else None


@router.get("/movies/poster")
async def get_movie_poster(title: str):
    """
    Endpoint externo: recibe un título y devuelve el póster.
    """
    poster_url = await fetch_movie_poster(title)
    return {"poster": poster_url}