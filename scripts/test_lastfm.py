# Importación correcta: Python usa la estructura de carpetas para módulos.
# Asume que estás ejecutando el script desde el directorio superior a 'services'.
from services.lastfm_api import get_album_art
import logging

# Configura el logging básico para que los mensajes de depuración sean visibles
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- El código de prueba es SÍNCRONO porque get_album_art es SÍNCRONA ---

if __name__ == "__main__":
    print("\n=======================================================")
    print(" 🛠️ Iniciando prueba directa de Last.fm API (Depuración) 🛠️")
    print("=======================================================")

    # 1. PRUEBA DE FALLO CONOCIDO (Lorde - Shapeshifter)
    print("\n--- PRUEBA 1: Lorde - Shapeshifter (Fallo/Placeholder esperado) ---")
    artist_fail = "Lorde"
    track_fail = "Shapeshifter"
    url_fail = get_album_art(artist_fail, track_fail)
    print(f"\n[TEST RESULTADO] Lorde - Shapeshifter: {url_fail}")

    # 2. PRUEBA DE ÉXITO CONOCIDO (The Weeknd - Blinding Lights)
    print("\n--- PRUEBA 2: The Weeknd - Blinding Lights (Éxito esperado) ---")
    artist_success = "The Weeknd"
    track_success = "Blinding Lights"
    url_success = get_album_art(artist_success, track_success)
    print(f"\n[TEST RESULTADO] The Weeknd - Blinding Lights: {url_success}")

    print("\n=======================================================\n")    