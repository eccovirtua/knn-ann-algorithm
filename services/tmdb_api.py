import httpx
import re
from fastapi import APIRouter, HTTPException
import re
import httpx
router = APIRouter()

TMDB_API_KEY = "647d6ce7695cbea9dafc5c77378bac2b"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"


async def fetch_movie_enrichment_data(raw_title: str) -> dict | None:
    """
    Recibe un título como 'Child's Play 2 (1990)', lo limpia, consulta TMDB 
    y devuelve un diccionario con toda la info enriquecida en español.
    """
    if not TMDB_API_KEY:
        raise RuntimeError("TMDB_API_KEY no está configurada")

    # 1. Limpiar el título y aislar el año (Magia con Regex)
    # Busca todo hasta que encuentra un paréntesis con 4 números al final
    match = re.search(r"^(.*?)\s*\((\d{4})\)", raw_title)
    if match:
        clean_title = match.group(1).strip()
        year = match.group(2)
    else:
        clean_title = raw_title.strip()
        year = None

    async with httpx.AsyncClient() as client:
        # 2. Buscar la película para obtener su TMDB ID
        search_url = f"{TMDB_BASE_URL}/search/movie"
        search_params = {"api_key": TMDB_API_KEY, "query": clean_title}
        if year:
            search_params["primary_release_year"] = year

        search_resp = await client.get(search_url, params=search_params)
        
        if search_resp.status_code != 200 or not search_resp.json().get("results"):
            return None

        # Tomamos el ID del primer resultado (el más relevante)
        movie_id = search_resp.json()["results"][0]["id"]

        # 3. Pedir TODOS los detalles de una sola vez usando el ID
        details_url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        details_params = {
            "api_key": TMDB_API_KEY,
            "language": "es-MX", # Español de Hispanoamérica (Títulos y sinopsis)
            "append_to_response": "credits,keywords" # Trae directores y descriptores
        }
        details_resp = await client.get(details_url, params=details_params)
        
        if details_resp.status_code != 200:
            return None

        data = details_resp.json()

        # --- 4. Extraer exactamente los datos que necesitas ---
        
        # Póster
        poster_path = data.get("poster_path")
        poster_url = f"{TMDB_IMG_BASE}{poster_path}" if poster_path else None

        # Géneros (TMDB los trae como lista de diccionarios, sacamos solo los nombres y máximo 3)
        genres_list = [g["name"] for g in data.get("genres", [])]
        genres = genres_list[:3]

        # Tipo (Película vs Documental)
        item_type = "Documental" if "Documental" in genres_list else "Película"

        # Director (Buscamos en el 'crew' el que tenga el trabajo de 'Director')
        director = None
        for crew_member in data.get("credits", {}).get("crew", []):
            if crew_member.get("job") == "Director":
                director = crew_member.get("name")
                break

        # Descriptores/Keywords (sacamos máximo 6 palabras clave)
        keywords = [k["name"] for k in data.get("keywords", {}).get("keywords", [])][:6]

        # Armamos el paquete final
        return {
            "title_es": data.get("title", clean_title), # Título en español!
            "overview": data.get("overview", ""),       # Sinopsis
            "poster_url": poster_url,
            "director": director,
            "type": item_type,
            "genres": genres,
            "keywords": keywords
        }


@router.get("/movies/enrich")
async def get_movie_enrichment(title: str):
    """
    Endpoint temporal de prueba. Úsalo para probar con una sola película.
    """
    data = await fetch_movie_enrichment_data(title)
    if not data:
        raise HTTPException(status_code=404, detail="Película no encontrada en TMDB")
    return data