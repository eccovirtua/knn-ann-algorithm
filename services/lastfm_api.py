import httpx
import re

LASTFM_API_KEY = "35cb090bb75bc5d49c342c37d0d5e625"
LASTFM_BASE_URL = "https://ws.audioscrobbler.com/2.0/"
PLACEHOLDER = "2a96cbd8b46e442fc41c2b86b821562f.png"

def clean_name(text: str) -> str:
    """Limpia nombres de artista o canción para mejorar coincidencia"""
    text = re.sub(r"\(.*?\)", "", text)  # elimina paréntesis
    text = re.sub(r"feat\.?.*|ft\.?.*", "", text, flags=re.IGNORECASE)  # elimina featuring
    return text.strip()

async def get_album_art(artist: str, track: str) -> str | None:
    if not LASTFM_API_KEY:
        return None

    artist = clean_name(artist)
    track = clean_name(track)

    async with httpx.AsyncClient(timeout=10) as client:
        # --- Nuevo Intento con track.getInfo ---
        try:
            params = {
                "method": "track.getInfo", # <-- ¡Cambio aquí!
                "api_key": LASTFM_API_KEY,
                "artist": artist,
                "track": track,           # <-- ¡Cambio aquí!
                "autocorrect": 1,         # <-- ¡Muy útil para coincidencias!
                "format": "json",
            }
            resp = await client.get(LASTFM_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            # La imagen ahora estará dentro de track.album.image
            album = data.get("track", {}).get("album", {})
            if album and "image" in album:
                # La lógica de Last.fm generalmente pone la imagen más grande al final (reversed)
                for img in reversed(album["image"]):
                    url = img.get("#text")
                    if url and PLACEHOLDER not in url:
                        return url
        except Exception as e:
            # Revisa la respuesta de Last.fm; si el track no existe, fallará.
            print(f"[LastFM track.getInfo] Error: {e}")

    # Si no se encuentra una imagen válida:
    return None