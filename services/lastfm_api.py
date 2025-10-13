import httpx
import re
import json
import logging

# Configurar el logger para que Uvicorn lo capture
logger = logging.getLogger("uvicorn")

LASTFM_API_KEY = "35cb090bb75bc5d49c342c37d0d5e625"
LASTFM_BASE_URL = "http://ws.audioscrobbler.com/2.0/"
PLACEHOLDER = "https://lastfm.freetls.fastly.net/i/u/300x300/2a96cbd8b46e442fc41c2b86b821562f.png"


def clean_name(text: str) -> str:
    """Limpia nombres de artista o canción para mejorar coincidencia de forma segura."""

    # 1. Eliminar contenido entre paréntesis y corchetes (versiones/ediciones)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]", "", text)

    # 2. Eliminar 'featuring'
    text = re.sub(r"feat\.?.*|ft\.?.*", "", text, flags=re.IGNORECASE)

    # 3. Reemplazar caracteres especiales de unión por un espacio
    text = re.sub(r"[\-&/]", " ", text)

    # 4. Eliminar espacios duplicados y limpiar extremos.
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_image_url(data: dict, path: list) -> str | None:
    """Función auxiliar para extraer la URL de la imagen en diferentes rutas."""
    try:
        current = data
        for key in path:
            current = current[key]

        # current debe ser una lista de objetos de imagen de Last.fm
        if not current or not isinstance(current, list):
            return None

        # Buscamos la URL real, empezando por la más grande (al final de la lista)
        for img in reversed(current):
            url = img.get("#text")
            if url and PLACEHOLDER not in url:
                return url

    except (KeyError, TypeError):
        return None

    return None


def get_album_art(artist: str, track: str) -> str | None:
    if not LASTFM_API_KEY:
        return None

    artist_clean = clean_name(artist)
    track_clean = clean_name(track)

    print(f"DEBUG_CLEAN: API CALLING with: Artist='{artist_clean}', Track='{track_clean}'")

    logger.info(f"--- 🔎 Iniciando búsqueda para: {artist_clean} - {track_clean} ---")

    with httpx.Client(timeout=10) as client:
        # --- Intento 1: track.getInfo (Más preciso) ---
        url = None  # Inicializamos la URL para el primer intento
        try:
            params_track = {
                "method": "track.getInfo",
                "api_key": LASTFM_API_KEY,
                "artist": artist_clean,
                "track": track_clean,
                "autocorrect": 1,
                "format": "json",
            }
            resp = client.get(LASTFM_BASE_URL, params=params_track)
            resp.raise_for_status()
            data = resp.json()

            url = extract_image_url(data, ["track", "album", "image"])
            if url:
                logger.info(f"✅ [ÉXITO 1 - track.getInfo] Portada encontrada: {url}")
                return url

            logger.warning(f"❌ [FALLO 1 - track.getInfo] No se encontró URL real.")

        except httpx.HTTPStatusError as e:
            logger.error(f"❌ [FALLO 1a - track.getInfo] HTTP Error {e.response.status_code}. Pista no encontrada.")
        except Exception as e:
            logger.error(f"⚠️ [ERROR 1b - track.getInfo] Error inesperado: {e}")

        # --- Intento 2: album.search (Fallback menos preciso) ---
        url = None  # Reiniciamos la URL para el segundo intento
        try:
            params_search = {
                "method": "album.search",
                "api_key": LASTFM_API_KEY,
                "album": f"{artist_clean} - {track_clean}",
                "format": "json",
            }
            resp = client.get(LASTFM_BASE_URL, params=params_search)
            resp.raise_for_status()
            data = resp.json()

            # Lógica corregida para manejo de JSON vacío y evitar 'list index out of range'
            album_matches = data.get("results", {}).get("albummatches", {}).get("album")

            if album_matches and isinstance(album_matches, list) and len(album_matches) > 0:
                image_list = album_matches[0].get("image")
                if image_list:
                    url = extract_image_url({"image_list": image_list}, ["image_list"])

                if url:
                    print(f"✅ [ÉXITO 2 - album.search] Portada encontrada (Fallback): {url}")
                    return url

            print(f"❌ [FALLO 2 - album.search] No se encontró URL real en el fallback.")

            # 💡 ¡ESTO ES LO IMPORTANTE! Imprimir la respuesta cruda
            print(f"--- 🚨 RESPUESTA CRUDA DE FALLBACK para {artist_clean} - {track_clean} ---")
            print(json.dumps(data, indent=2))

        except Exception as e:
            print(f"⚠️ [ERROR 2 - album.search] Error inesperado en fallback: {e}")

        print(f"🛑 FALLO TOTAL para {artist_clean} - {track_clean}. Usando URL por defecto.")
        return None