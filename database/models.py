from sqlalchemy import Column, String, LargeBinary, Integer, Time, Date, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.sql import func
from database.connection import Base
import numpy as np

class Alumno(Base):
    __tablename__ = "alumnos"

    id = Column(String, primary_key=True)               # Código único del estudiante (ej: "20230045")
    nombre = Column(String, nullable=False)              # Nombre completo
    embedding = Column(LargeBinary, nullable=False)      # Vector biométrico guardado como bytes de float32
    totp_secret = Column(String, nullable=False)         # Secreto Base32 para validación TOTP
    creado_en = Column(DateTime, server_default=func.now())

    def set_embedding(self, embedding_list):
        """Convierte una lista o array numpy de floats a bytes compactos float32 para SQLite."""
        arr = np.array(embedding_list, dtype=np.float32)
        self.embedding = arr.tobytes()

    def get_embedding(self) -> np.ndarray:
        """Reconstruye el array numpy float32 original a partir de los bytes guardados en BD."""
        if not self.embedding:
            return np.array([], dtype=np.float32)
        return np.frombuffer(self.embedding, dtype=np.float32)


class ClaseConfig(Base):
    __tablename__ = "clases_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    nombre_materia = Column(String, nullable=False)
    hora_inicio = Column(String, nullable=False)         # Almacenado como "HH:MM:SS" (ej. "09:00:00")
    limite_presente = Column(Integer, nullable=False)    # Tolerancia en minutos para "Presente" (ej: 5)
    limite_tarde = Column(Integer, nullable=False)       # Tolerancia en minutos para "Tarde" (ej: 20)
    activo = Column(Integer, default=1)                  # 1 = Activo, 0 = Inactivo


class Sesion(Base):
    __tablename__ = "sesiones"

    id = Column(Integer, primary_key=True, autoincrement=True)
    clase_config_id = Column(Integer, ForeignKey("clases_config.id"), nullable=False)
    fecha = Column(Date, server_default=func.current_date())
    hora_apertura = Column(DateTime, server_default=func.now())
    hora_cierre = Column(DateTime, nullable=True)
    estado = Column(String, default="ABIERTA")           # "ABIERTA" o "CERRADA"


class Asistencia(Base):
    __tablename__ = "asistencias"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sesion_id = Column(Integer, ForeignKey("sesiones.id"), nullable=False)
    alumno_id = Column(String, ForeignKey("alumnos.id"), nullable=False)
    timestamp = Column(DateTime, server_default=func.now())
    metodo = Column(String, nullable=False)              # "FACIAL" o "QR"
    estado = Column(String, nullable=False)              # "PRESENTE", "TARDE", "FALTA"

    __table_args__ = (
        UniqueConstraint("sesion_id", "alumno_id", name="uq_sesion_alumno"),  # Evitar registros duplicados
    )
