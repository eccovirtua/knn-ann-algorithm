import asyncio
import re
import logging
import os
import httpx
from motor.motor_asyncio import AsyncIOMotorClient
from googletrans import Translator
from dotenv import load_dotenv

# --- Configuración ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI")
TMDB_API_KEY = "647d6ce7695cbea9dafc5c77378bac2b"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

translator = Translator()

async def fetch_movie_enrichment_data(client: httpx.AsyncClient, raw_title: str) -> dict | None:
    # 1. Limpiar el título y aislar el año
    match = re.search(r"^(.*?)\s*\((\d{4})\)", raw_title)
    if match:
        clean_title = match.group(1).strip()
        year = match.group(2)
    else:
        clean_title = raw_title.strip()
        year = None

    search_url = f"{TMDB_BASE_URL}/search/movie"
    search_params = {"api_key": TMDB_API_KEY, "query": clean_title, "language": "es-MX"}
    if year:
        search_params["primary_release_year"] = year

    try:
        # 2. Buscar ID de la película
        search_resp = await client.get(search_url, params=search_params)
        if search_resp.status_code != 200 or not search_resp.json().get("results"):
            return None

        movie_id = search_resp.json()["results"][0]["id"]

        # 3. Traer detalles completos
        details_url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        details_params = {
            "api_key": TMDB_API_KEY,
            "language": "es-MX", # Trae géneros, título y overview en ESPAÑOL
            "append_to_response": "credits,keywords"
        }
        details_resp = await client.get(details_url, params=details_params)
        
        if details_resp.status_code != 200:
            return None

        data = details_resp.json()

        # 4. Extraer y procesar datos
        poster_path = data.get("poster_path")
        poster_url = f"{TMDB_IMG_BASE}{poster_path}" if poster_path else None
        
        genres_list = [g["name"] for g in data.get("genres", [])]
        item_type = "Documental" if "Documental" in genres_list else "Película"

        director = next((crew.get("name") for crew in data.get("credits", {}).get("crew", []) if crew.get("job") == "Director"), None)

        # 5. Traducir Keywords al Español
        keywords_en = [k["name"] for k in data.get("keywords", {}).get("keywords", [])][:6]
        keywords_es = []
        if keywords_en:
            try:
                text_to_translate = ", ".join(keywords_en)
                translation = translator.translate(text_to_translate, src='en', dest='es')
                keywords_es = [k.strip().lower() for k in translation.text.split(",")]
            except Exception:
                keywords_es = keywords_en # Fallback si falla el traductor

        # 6. Retornar Mega-Paquete
        return {
            "title_es": data.get("title", clean_title),
            "overview": data.get("overview", ""),
            "image_url": poster_url,
            "director": director,
            "domain_type": item_type,
            "genres_es": genres_list, # Guardamos la lista completa
            "keywords_es": keywords_es
        }

    except Exception as e:
        logger.error(f"Error procesando {raw_title}: {e}")
        return None

async def process_single_movie(doc, client, items_col, sem):
    async with sem: 
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
            logger.info(f"✅ Éxito: {enrichment_data['title_es']} ({item_id})")
        else:
            logger.warning(f"❌ Fallo: {raw_title} ({item_id})")
            
        await asyncio.sleep(0.05) # Pausa amigable para la API

async def main():
    mongo_client = AsyncIOMotorClient(MONGO_URI)
    db = mongo_client.get_database()
    items_col = db.get_collection("items_col")

    query = {
    "domain": "movie",
    "$or": [
        {"title": {"$exists": False}},
        {"director": {"$exists": False}},
        {"genres_es": {"$exists": False}},
        {"keywords_es": {"$exists": False}},
        {"overview": {"$exists": False}}
    ]
}

    # Busca películas que NO tengan el title_es (indicador de que no han pasado por el script)
    cursor = items_col.find(query)
    movies_to_process = await cursor.to_list(length=None)
    
    if not movies_to_process:
        logger.info("¡Todas las películas ya están enriquecidas!")
        return

    logger.info(f"Procesando {len(movies_to_process)} películas...")
    sem = asyncio.Semaphore(20)

    async with httpx.AsyncClient(timeout=20.0) as http_client:
        tasks = [process_single_movie(doc, http_client, items_col, sem) for doc in movies_to_process]
        await asyncio.gather(*tasks)

    logger.info("¡Enriquecimiento completado!")

if __name__ == "__main__":
    asyncio.run(main()) 