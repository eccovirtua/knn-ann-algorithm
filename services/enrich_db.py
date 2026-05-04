import asyncio
import re
from motor.motor_asyncio import AsyncIOMotorClient
import httpx
from googletrans import Translator
import logging
import os
from dotenv import load_dotenv

# --- Configuración y Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI")  # Asegúrate de que esto esté en tu .env
TMDB_API_KEY = "647d6ce7695cbea9dafc5c77378bac2b"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

translator = Translator()


async def fetch_movie_enrichment_data(client: httpx.AsyncClient, raw_title: str) -> dict | None:
    match = re.search(r"^(.*?)\s*\((\d{4})\)", raw_title)
    if match:
        clean_title = match.group(1).strip()
        year = match.group(2)
    else:
        clean_title = raw_title.strip()
        year = None

    search_url = f"{TMDB_BASE_URL}/search/movie"
    search_params = {"api_key": TMDB_API_KEY, "query": clean_title}
    if year:
        search_params["primary_release_year"] = year

    try:
        search_resp = await client.get(search_url, params=search_params)
        if search_resp.status_code != 200 or not search_resp.json().get("results"):
            return None

        movie_id = search_resp.json()["results"][0]["id"]

        details_url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        details_params = {
            "api_key": TMDB_API_KEY,
            "language": "es-MX",
            "append_to_response": "credits,keywords"
        }
        details_resp = await client.get(details_url, params=details_params)
        
        if details_resp.status_code != 200:
            return None

        data = details_resp.json()

        poster_path = data.get("poster_path")
        poster_url = f"{TMDB_IMG_BASE}{poster_path}" if poster_path else None
        
        genres_list = [g["name"] for g in data.get("genres", [])]
        genres = genres_list[:3]
        item_type = "Documental" if "Documental" in genres_list else "Película"

        director = None
        for crew_member in data.get("credits", {}).get("crew", []):
            if crew_member.get("job") == "Director":
                director = crew_member.get("name")
                break

        keywords_en = [k["name"] for k in data.get("keywords", {}).get("keywords", [])][:6]
        keywords_es = []
        if keywords_en:
            try:
                text_to_translate = ", ".join(keywords_en)
                translation = translator.translate(text_to_translate, src='en', dest='es')
                keywords_es = [k.strip() for k in translation.text.split(",")]
            except Exception as e:
                keywords_es = keywords_en

        return {
            "title": data.get("title", clean_title),
            "overview": data.get("overview", ""),
            "image_url": poster_url,
            "director": director,
            "domain_type": item_type,
            "genres": genres,
            "keywords": keywords_es
        }

    except Exception as e:
        return None

async def process_single_movie(doc, client, items_col, sem):
    """Procesa una sola película controlando la concurrencia con el Semáforo"""
    async with sem:  # Espera su turno si ya hay 20 ejecutándose
        raw_title = doc.get("title")
        item_id = doc.get("itemId")

        if not raw_title:
            return

        enrichment_data = await fetch_movie_enrichment_data(client, raw_title)

        if enrichment_data:
            await items_col.update_one(
                {"itemId": item_id},
                {"$set": enrichment_data}
            )
            logger.info(f"✅ Éxito: {enrichment_data['title']} ({item_id})")
        else:
            logger.warning(f"❌ Fallo (TMDB no encontró): {raw_title} ({item_id})")
            
        # Pausa microscópica para evitar picos abruptos en TMDB
        await asyncio.sleep(0.05)

async def main():
    if not MONGO_URI:
        logger.error("Falta MONGODB_URI en el archivo .env")
        return

    mongo_client = AsyncIOMotorClient(MONGO_URI)
    db = mongo_client.get_database()
    items_col = db.get_collection("items_col")

    # 1. Traer TODA la lista a la memoria RAM de inmediato (evita el CursorNotFound)
    cursor = items_col.find({"domain": "movie", "director": {"$exists": False}})
    movies_to_process = await cursor.to_list(length=None)
    total_movies = len(movies_to_process)
    
    if total_movies == 0:
        logger.info("¡Todas las películas ya están enriquecidas!")
        return

    logger.info(f"Comenzando el enriquecimiento súper rápido de {total_movies} películas restantes...")

    # 2. Semáforo: Máximo 20 peticiones a la vez (TMDB permite 40/s, somos precavidos)
    sem = asyncio.Semaphore(20)

    # 3. Lanzar todas las tareas concurrentemente
    async with httpx.AsyncClient(timeout=20.0) as http_client:
        tasks = [
            process_single_movie(doc, http_client, items_col, sem) 
            for doc in movies_to_process
        ]
        # Esperamos a que todas terminen
        await asyncio.gather(*tasks)

    logger.info("¡Enriquecimiento completado al 100%!")

if __name__ == "__main__":
    asyncio.run(main())