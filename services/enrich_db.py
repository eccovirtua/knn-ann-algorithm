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

# ==============================================================================
# FUNCIÓN PRINCIPAL DE ENRIQUECIMIENTO (TMDB)
# ==============================================================================
async def fetch_movie_enrichment_data(client: httpx.AsyncClient, raw_title: str) -> dict | None:
    if not TMDB_API_KEY or TMDB_API_KEY == "TU_API_KEY_DE_TMDB_AQUI":
        raise RuntimeError("TMDB_API_KEY no está configurada correctamente.")

    # =================================================================
    # 1. LIMPIEZA NIVEL DIOS Y TRUCO MOVIELENS
    # =================================================================
    # Aislar el año de estreno
    match = re.search(r"^(.*?)\s*\((\d{4})\)", raw_title)
    if match:
        base_title = match.group(1).strip()
        year = match.group(2)
    else:
        base_title = raw_title.strip()
        year = None

    # Eliminar textos basura entre paréntesis tipo "(a.k.a. ...)" o "(Kaidan)"
    clean_title = re.sub(r"\(.*?\)", "", base_title).strip()

    # INVERTIR EL ARTÍCULO: Convierte "Lost Skeleton of Cadavra, The" en "The Lost Skeleton of Cadavra"
    article_match = re.search(r"^(.*?),\s*(The|A|An|El|La|Los|Las|Le|Les|L'|Il|Der|Die|Das)$", clean_title, flags=re.IGNORECASE)
    if article_match:
        clean_title = f"{article_match.group(2)} {article_match.group(1)}".strip()
        logger.info(f"✨ Título invertido para búsqueda: '{clean_title}'")

    # =================================================================
    # 2. BÚSQUEDA CON REINTENTO INTELIGENTE Y CHISMOSO
    # =================================================================
    search_url = f"{TMDB_BASE_URL}/search/movie"
    search_params = {"api_key": TMDB_API_KEY, "query": clean_title, "language": "es-MX"}
    if year:
        search_params["primary_release_year"] = year

    try:
        search_resp = await client.get(search_url, params=search_params)
        
        # Chismoso 1: ¿Nos bloqueó TMDB por ir muy rápido?
        if search_resp.status_code == 429:
            logger.warning(f"⚠️ RATE LIMIT (429) alcanzado al buscar: {clean_title}")
            return None
        elif search_resp.status_code != 200:
            logger.warning(f"⚠️ Error {search_resp.status_code} en TMDB para: {clean_title}")
            return None

        results = search_resp.json().get("results", [])

        # FALLBACK: Si no encuentra nada con el año, intenta buscar SIN el año
        if not results and year:
            logger.info(f"🔄 Reintentando '{clean_title}' sin el año {year}...")
            search_params.pop("primary_release_year", None)
            search_resp = await client.get(search_url, params=search_params)
            results = search_resp.json().get("results", [])

        # Chismoso 2: ¿De verdad no existe ni con el título limpio?
        if not results:
            logger.error(f"🔍 CERO RESULTADOS en TMDB para: '{clean_title}' (Original: {raw_title})")
            return None

        movie_id = results[0]["id"]

        # =================================================================
        # 3. TRAER DETALLES COMPLETOS (Para Sinopsis, Score y Directores)
        # =================================================================
        details_url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        details_params = {
            "api_key": TMDB_API_KEY,
            "language": "es-MX",
            "append_to_response": "credits,keywords"
        }
        details_resp = await client.get(details_url, params=details_params)
        
        if details_resp.status_code != 200:
            logger.warning(f"⚠️ Error {details_resp.status_code} al traer detalles del ID {movie_id}")
            return None

        data = details_resp.json()

        # =================================================================
        # 4. EXTRAER Y PROCESAR DATOS
        # =================================================================
        poster_path = data.get("poster_path")
        poster_url = f"{TMDB_IMG_BASE}{poster_path}" if poster_path else None
        
        genres_list = [g["name"] for g in data.get("genres", [])]
        item_type = "Documental" if "Documental" in genres_list else "Película"

        director = next((crew.get("name") for crew in data.get("credits", {}).get("crew", []) if crew.get("job") == "Director"), None)

        vote_average = data.get("vote_average", 0.0)
        imdb_score = round(float(vote_average), 1) if vote_average else 0.0

        keywords_en = [k["name"] for k in data.get("keywords", {}).get("keywords", [])][:6]
        keywords_es = []
        if keywords_en:
            try:
                text_to_translate = ", ".join(keywords_en)
                translation = translator.translate(text_to_translate, src='en', dest='es')
                keywords_es = [k.strip().lower() for k in translation.text.split(",")]
            except Exception:
                keywords_es = keywords_en

        # =================================================================
        # 5. DEVOLVER EL MEGA-PAQUETE A LA BASE DE DATOS
        # =================================================================
        return {
            "title_es": data.get("title", clean_title),
            "overview": data.get("overview", ""),
            "image_url": poster_url,
            "director": director,
            "domain_type": item_type,
            "genres_es": genres_list,
            "keywords_es": keywords_es,
            "imdb_score": imdb_score
        }

    except Exception as e:
        # Chismoso 3: ¿Se cayó la conexión o hubo un error inesperado de Python?
        logger.error(f"💥 EXCEPCIÓN procesando {raw_title}: {str(e)}")
        return None
# ==============================================================================
# PROCESAMIENTO INDIVIDUAL EN BASE DE DATOS
# ==============================================================================
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
            logger.info(f"✅ Éxito: {enrichment_data['title_es']} | Tipo: {enrichment_data['domain_type']} | Score: {enrichment_data['imdb_score']}")
        else:
            logger.warning(f"❌ Fallo al buscar datos: {raw_title} ({item_id})")
            
        await asyncio.sleep(0.02) # Pausa amigable para no saturar la API de TMDB

# ==============================================================================
# MOTOR PRINCIPAL
# ==============================================================================
async def main():
    mongo_client = AsyncIOMotorClient(MONGO_URI)
    db = mongo_client.get_database() 
    items_col = db.get_collection("items_col") # Reemplaza con el nombre real de tu colección

    # =====================================================================
    # QUERY MEJORADO: Busca películas a las que les falte cualquier dato
    # =====================================================================
    query = {
        "domain": "movie",
        "$or": [
            {"title_es": {"$exists": False}},
            {"director": {"$exists": False}},
            {"genres_es": {"$exists": False}},
            {"keywords_es": {"$exists": False}},
            {"overview": {"$exists": False}},
            {"imdb_score": {"$exists": False}},
            {"image_url": {"$exists": False}},
            {"domain_type": {"$exists": False}}
        ]
    }

    cursor = items_col.find(query)
    movies_to_process = await cursor.to_list(length=None)
    
    if not movies_to_process:
        logger.info("¡Todas las películas ya están enriquecidas con todos los campos requeridos!")
        return

    logger.info(f"Procesando {len(movies_to_process)} películas incompletas...")
    
    # Controlamos cuántas peticiones simultáneas hacemos (20 a la vez)
    sem = asyncio.Semaphore(35)

    async with httpx.AsyncClient(timeout=20.0) as http_client:
        tasks = [process_single_movie(doc, http_client, items_col, sem) for doc in movies_to_process]
        await asyncio.gather(*tasks)

    logger.info("¡Enriquecimiento completado al 100%!")

if __name__ == "__main__":
    asyncio.run(main())
