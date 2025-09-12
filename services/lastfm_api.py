# services/lastfm_api.py
import httpx
# import requests

LASTFM_API_KEY = "35cb090bb75bc5d49c342c37d0d5e625"
LASTFM_BASE_URL = "https://ws.audioscrobbler.com/2.0/"

async def get_album_art(artist: str, track: str) -> str | None:
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

            album = data.get("track", {}).get("album")
            if not album:
                return None  # si no hay album, no hay portada

            image_list = album.get("image", [])
            for img in reversed(image_list):  # revisa de mayor a menor tamaño
                url = img.get("#text")
                if url and "2a96cbd8b46e442fc41c2b86b821562f.png" not in url:
                    return url

    except Exception as e:
        print(f"Error Last.fm: {e}")

    return None

# API_KEY = LASTFM_API_KEY
# artist = "Coldplay"
# track = "Yellow"
#
# url = f"http://ws.audioscrobbler.com/2.0/?method=track.getInfo&api_key={API_KEY}&artist={artist}&track={track}&format=json"
# resp = requests.get(url)
# print(resp.json())