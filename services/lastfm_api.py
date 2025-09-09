# services/lastfm_api.py
import httpx

LASTFM_API_KEY = "35cb090bb75bc5d49c342c37d0d5e625"
LASTFM_BASE_URL = "https://ws.audioscrobbler.com/2.0/"

async def get_album_art(artist: str, track: str) -> str | None:
    """
    Busca la portada de un track en Last.fm usando artista + track.
    Retorna la URL de la imagen o None si no hay resultados.
    """
    if not LASTFM_API_KEY:
        return None

    params = {
        "method": "track.getInfo",
        "api_key": LASTFM_API_KEY,
        "artist": artist,
        "track": track,
        "format": "json",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(LASTFM_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            # Navegamos en la respuesta
            image_list = data.get("track", {}).get("album", {}).get("image", [])
            if image_list:
                # Tomamos la última (generalmente la de mayor resolución)
                return image_list[-1].get("#text")

    except Exception as e:
        print(f"Error Last.fm: {e}")

    return None
