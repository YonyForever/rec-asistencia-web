import cv2
import numpy as np
from database.connection import get_session
from database.models import Alumno
from core.config import YUNET_MODEL_PATH, SFACE_MODEL_PATH

class FaceEngine:
    def __init__(self):
        """Inicializa el motor de detección y reconocimiento facial de OpenCV ONNX."""
        print("[FaceEngine] Cargando redes neuronales en memoria...")
        
        # Detector Facial YuNet
        self.detector = cv2.FaceDetectorYN.create(
            model=str(YUNET_MODEL_PATH),
            config="",
            input_size=(320, 240),  # Tamaño base, se actualiza dinámicamente con cada frame
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
        
        # Reconocedor Facial SFace
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=str(SFACE_MODEL_PATH),
            config="",
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
        
        # Estructuras de memoria para emparejamiento ultra rápido
        self.alumnos_ids = []
        self.alumnos_nombres = []
        self.embeddings_matrix = None  # Matriz NumPy (N, 128)
        
        # Umbral oficial de SFace para Similitud de Coseno
        # Valor recomendado >= 0.363 para un emparejamiento confiable
        self.cosine_threshold = 0.363

    def reload_students_cache(self):
        """Carga todos los embeddings biométricos desde la BD a la memoria RAM."""
        print("[FaceEngine] Cargando base de datos a memoria RAM...")
        session = get_session()
        try:
            alumnos = session.query(Alumno).all()
            if not alumnos:
                self.alumnos_ids = []
                self.alumnos_nombres = []
                self.embeddings_matrix = None
                print("[FaceEngine] Cache vacío: No hay alumnos registrados.")
                return False
                
            temp_embeddings = []
            self.alumnos_ids = []
            self.alumnos_nombres = []
            
            for alumno in alumnos:
                vector = alumno.get_embedding()
                if vector.size == 128:
                    self.alumnos_ids.append(alumno.id)
                    self.alumnos_nombres.append(alumno.nombre)
                    temp_embeddings.append(vector)
            
            if temp_embeddings:
                self.embeddings_matrix = np.array(temp_embeddings, dtype=np.float32)
                print(f"[FaceEngine] Cache cargado con éxito. Alumnos activos en RAM: {len(self.alumnos_ids)}")
                return True
            else:
                self.embeddings_matrix = None
                print("[FaceEngine] Advertencia: Alumnos encontrados pero sin firmas biométricas válidas.")
                return False
        except Exception as e:
            print(f"[FaceEngine] Error al recargar cache de alumnos: {e}")
            return False
        finally:
            session.close()

    def detect_face(self, frame):
        """
        Detecta si hay un rostro en el frame y devuelve el primer rostro detectado.
        Actualiza el tamaño de entrada de YuNet dinámicamente si cambia la resolución.
        """
        h, w, _ = frame.shape
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        
        if faces is not None and len(faces) > 0:
            return faces[0]  # Retornar el primer rostro detectado
        return None

    def extract_embedding(self, frame, face_data):
        """
        Alinea el rostro detectado y extrae su vector biométrico de 128 floats.
        """
        # Alinear y recortar la imagen del rostro
        face_aligned = self.recognizer.alignCrop(frame, face_data)
        
        # Generar firma biométrica
        feat = self.recognizer.feature(face_aligned)
        embedding = feat[0].copy()
        
        # Cumplir regla estricta de privacidad: borrar variables de imagen inmediatamente
        del face_aligned
        
        return embedding

    def match_face(self, current_embedding):
        """
        Compara el embedding actual contra toda la base de datos en memoria RAM 
        utilizando similitud de coseno vectorizada.
        Match Time: < 0.005 segundos.
        
        Retorna: (alumno_id, nombre, score) si hace match, o (None, None, score)
        """
        if self.embeddings_matrix is None or len(self.alumnos_ids) == 0:
            return None, None, 0.0
            
        # Calcular normas vectoriales
        norms_registered = np.linalg.norm(self.embeddings_matrix, axis=1)
        norm_current = np.linalg.norm(current_embedding)
        
        if norm_current == 0:
            return None, None, 0.0
            
        # Similitud de Coseno en una sola operación NumPy para todos los registros a la vez
        similarities = np.dot(self.embeddings_matrix, current_embedding) / (norms_registered * norm_current)
        
        # Buscar el índice con mayor coincidencia
        best_idx = np.argmax(similarities)
        best_score = float(similarities[best_idx])
        
        if best_score >= self.cosine_threshold:
            return self.alumnos_ids[best_idx], self.alumnos_nombres[best_idx], best_score
            
        return None, None, best_score
