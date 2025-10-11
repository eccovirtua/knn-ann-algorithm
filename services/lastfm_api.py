# services/lastfm_api.py
import asyncio

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
        # --- 1️⃣ Primer intento: track.getInfo ---
        try:
            params = {
                "method": "track.getInfo",
                "api_key": LASTFM_API_KEY,
                "artist": artist,
                "track": track,
                "format": "json",
            }
            resp = await client.get(LASTFM_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()


            album = data.get("track", {}).get("album")
            if album and "image" in album:
                for img in reversed(album["image"]):
                    url = img.get("#text")
                    if url and PLACEHOLDER not in url:
                        return url
        except Exception as e:
            print(f"[LastFM track.getInfo] Error: {e}")

        # --- 2️⃣ Segundo intento: track.search ---
        try:
            params = {
                "method": "track.search",
                "api_key": LASTFM_API_KEY,
                "track": f"{artist} {track}",
                "format": "json",
                "limit": 1,
            }
            resp = await client.get(LASTFM_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            matches = data.get("results", {}).get("trackmatches", {}).get("track", [])
            if matches:
                img_list = matches[0].get("image", [])
                for img in reversed(img_list):
                    url = img.get("#text")
                    if url and PLACEHOLDER not in url:
                        return url
        except Exception as e:
            print(f"[LastFM track.search] Error: {e}")

        # --- 3️⃣ Último intento: buscar por artista ---
        try:
            params = {
                "method": "artist.getTopAlbums",
                "api_key": LASTFM_API_KEY,
                "artist": artist,
                "format": "json",
                "limit": 1,
            }
            resp = await client.get(LASTFM_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            albums = data.get("topalbums", {}).get("album", [])
            if albums:
                img_list = albums[0].get("image", [])
                for img in reversed(img_list):
                    url = img.get("#text")
                    if url and PLACEHOLDER not in url:
                        return url
        except Exception as e:
            print(f"[LastFM artist.getTopAlbums] Error: {e}")

    # Si nada funcionó:
    return None
print(asyncio.run(get_album_art("Twenty One Pilots", "Tally")))
print(asyncio.run(get_album_art("The Weeknd", "Blinding Lights")))
print(asyncio.run(get_album_art("Dua Lipa ft. DaBaby", "Levitating")))
print(asyncio.run(get_album_art("Coldplay", "Yellow")))
