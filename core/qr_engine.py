import cv2
import pyotp
from database.connection import get_session
from database.models import Alumno
from core.config import TOTP_INTERVAL

class QREngine:
    def __init__(self):
        """Inicializa el motor de detección de códigos QR de OpenCV."""
        print("[QREngine] Inicializando detector de códigos QR...")
        self.qr_detector = cv2.QRCodeDetector()

    def detect_and_decode(self, frame):
        """
        Detecta y decodifica un código QR presente en el frame.
        Retorna: (decoded_text, points) si detecta uno, o (None, None)
        """
        # detectAndDecode retorna: (decoded_info, points, straight_qrcode) en los bindings de Python
        decoded_info, points, _ = self.qr_detector.detectAndDecode(frame)
        
        if decoded_info:
            return decoded_info, points
        return None, None

    def validate_student_qr(self, decoded_info):
        """
        Valida la información del código QR decodificado.
        Formato esperado: "ALUMNO_ID:TOTP_TOKEN" (ej: "20230045:128493")
        
        Retorna: (is_valid, alumno_id, alumno_nombre, mensaje)
        """
        if not decoded_info or ":" not in decoded_info:
            return False, None, None, "Formato de QR inválido (sin delimitador ':' o vacío)"
            
        parts = decoded_info.split(":", 1)
        alumno_id = parts[0].strip()
        totp_token = parts[1].strip()
        
        if not alumno_id or not totp_token:
            return False, None, None, "Código de alumno o token TOTP vacíos"
            
        session = get_session()
        try:
            # Buscar el alumno en la base de datos
            alumno = session.query(Alumno).filter_by(id=alumno_id).first()
            if not alumno:
                return False, None, None, f"Alumno con ID {alumno_id} no está registrado en el sistema"
                
            # Configurar el validador TOTP de 15 segundos para el alumno
            totp = pyotp.TOTP(alumno.totp_secret, interval=TOTP_INTERVAL)
            
            # Validar el token actual
            # valid_window=2 permite dos tokens de desfase (atrás o adelante) para facilitar pruebas y presentaciones sin perder seguridad
            is_valid = totp.verify(totp_token, valid_window=1)
            
            if is_valid:
                return True, alumno.id, alumno.nombre, "Código QR dinámico validado correctamente"
            else:
                return False, alumno.id, alumno.nombre, "Token TOTP inválido o expirado (fraude de captura detectado)"
                
        except Exception as e:
            return False, None, None, f"Error en validación de base de datos: {e}"
        finally:
            session.close()
