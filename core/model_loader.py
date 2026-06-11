import urllib.request
import sys
from pathlib import Path
from core.config import YUNET_MODEL_PATH, SFACE_MODEL_PATH, YUNET_DOWNLOAD_URL, SFACE_DOWNLOAD_URL

def download_progress(block_num, block_size, total_size):
    """Callback para mostrar progreso de descarga en consola."""
    downloaded = block_num * block_size
    percent = min(100, (downloaded * 100) / total_size) if total_size > 0 else 0
    sys.stdout.write(f"\rDescargando... {percent:.1f}% ({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
    sys.stdout.flush()

def ensure_models_exist():
    """Verifica la existencia de los modelos YuNet y SFace, descargándolos si es necesario."""
    print("Verificando modelos de Deep Learning locales...")
    
    # Asegurar que el directorio de modelos existe
    YUNET_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Verificar YuNet (Detección Facial)
    if not YUNET_MODEL_PATH.exists():
        print(f"\n[YuNet] Modelo de detección no encontrado. Descargando desde la fuente oficial...")
        try:
            urllib.request.urlretrieve(YUNET_DOWNLOAD_URL, YUNET_MODEL_PATH, download_progress)
            print(f"\n[YuNet] Descargado correctamente en: {YUNET_MODEL_PATH.name}")
        except Exception as e:
            print(f"\nError al descargar YuNet: {e}")
            if YUNET_MODEL_PATH.exists():
                YUNET_MODEL_PATH.unlink()
            raise e
    else:
        print(f"[YuNet] Modelo de detección facial listo ({YUNET_MODEL_PATH.name}).")

    # 2. Verificar SFace (Reconocimiento Facial)
    if not SFACE_MODEL_PATH.exists():
        print(f"\n[SFace] Modelo de reconocimiento no encontrado. Descargando desde la fuente oficial...")
        try:
            urllib.request.urlretrieve(SFACE_DOWNLOAD_URL, SFACE_MODEL_PATH, download_progress)
            print(f"\n[SFace] Descargado correctamente en: {SFACE_MODEL_PATH.name}")
        except Exception as e:
            print(f"\nError al descargar SFace: {e}")
            if SFACE_MODEL_PATH.exists():
                SFACE_MODEL_PATH.unlink()
            raise e
    else:
        print(f"[SFace] Modelo de reconocimiento facial listo ({SFACE_MODEL_PATH.name}).")
    
    print("Todos los modelos de Visión Artificial están en local y listos.\n")
