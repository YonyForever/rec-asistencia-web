import os
from pathlib import Path

# Directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Ruta de la Base de Datos (SQLite local como fallback, PostgreSQL de Supabase en producción)
DB_PATH = BASE_DIR / "asistencia.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# Adaptar el prefijo de conexión si viene de servicios cloud como Render/Supabase
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Limpiar parámetros de pgbouncer/options conflictivos para psycopg2 (común en Supabase)
if "?" in DATABASE_URL:
    import urllib.parse
    parsed = urllib.parse.urlparse(DATABASE_URL)
    query_params = urllib.parse.parse_qsl(parsed.query)
    # Filtrar cualquier parámetro que sea 'options' o contenga 'pgbouncer'
    filtered_params = [
        (k, v) for k, v in query_params
        if k != "options" and "pgbouncer" not in v and "pgbouncer" not in k
    ]
    new_query = urllib.parse.urlencode(filtered_params)
    parsed = parsed._replace(query=new_query)
    DATABASE_URL = urllib.parse.urlunparse(parsed)


# Directorio de Modelos de Inteligencia Artificial (ONNX)
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# URL de descarga para los modelos de OpenCV Zoo (YuNet y SFace)
YUNET_DOWNLOAD_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_DOWNLOAD_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


# Configuración de Contingencia (QR / TOTP)
TOTP_INTERVAL = 60  # Segundos de validez para el QR dinámico

# Configuración de Hardware (Cámara)
CAMERA_INDEX = 0  # Cambiar a 1, 2 para cámaras USB externas, o una URL RTSP para cámaras IP

# Configuración de Telegram (se puede sobreescribir con variables de entorno o desde la BD)
TELEGRAM_BOT_TOKEN = "8903383252:AAFT1ts6JCssMPrHeqhJJxTe9asJ2OXKtvA"
TELEGRAM_CHAT_ID   = "6344059137"
