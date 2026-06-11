import cv2
import numpy as np
import pyotp
import argparse
import sys
from pathlib import Path

from core.config import CAMERA_INDEX, TOTP_INTERVAL
from core.model_loader import ensure_models_exist
from core.face_engine import FaceEngine
from database.connection import get_session
from database.models import Alumno

def parse_args():
    parser = argparse.ArgumentParser(description="Enrolamiento de Alumnos - Sistema Biométrico")
    parser.add_argument("--id", type=str, help="Código o Matrícula del Alumno (ej: 20230045)")
    parser.add_argument("--nombre", type=str, help="Nombre completo del Alumno")
    return parser.parse_args()

def register_student(alumno_id, nombre, embedding_vector, totp_secret):
    """Guarda el alumno con su embedding y llave TOTP en SQLite."""
    session = get_session()
    try:
        # Verificar si ya existe
        existente = session.query(Alumno).filter_by(id=alumno_id).first()
        if existente:
            print(f"\n[BD] El alumno con ID {alumno_id} ya existe. Se sobrescribirá su registro biométrico.")
            existente.nombre = nombre
            existente.set_embedding(embedding_vector)
            existente.totp_secret = totp_secret
        else:
            nuevo_alumno = Alumno(id=alumno_id, nombre=nombre)
            nuevo_alumno.set_embedding(embedding_vector)
            nuevo_alumno.totp_secret = totp_secret
            session.add(nuevo_alumno)
        
        session.commit()
        print(f"\n[ÉXITO] Alumno '{nombre}' ({alumno_id}) registrado correctamente en SQLite.")
        return True
    except Exception as e:
        session.rollback()
        print(f"\n[ERROR] No se pudo guardar en la base de datos: {e}")
        return False
    finally:
        session.close()

def main():
    args = parse_args()
    
    # Si no se proveen por consola, pedir interactivamente
    alumno_id = args.id
    nombre = args.nombre
    
    print("=" * 60)
    print("   SISTEMA DE ASISTENCIA BIOMÉTRICA - MÓDULO DE ENROLAMIENTO")
    print("=" * 60)
    
    if not alumno_id:
        alumno_id = input("Ingrese el Código/Matrícula del Alumno: ").strip()
    if not nombre:
        nombre = input("Ingrese el Nombre Completo del Alumno: ").strip()
        
    if not alumno_id or not nombre:
        print("[ERROR] El ID y el Nombre son obligatorios para el enrolamiento.")
        sys.exit(1)
        
    # Asegurar que los modelos ONNX estén listos
    ensure_models_exist()
    
    # Generar secreto TOTP único para el alumno (usando el intervalo configurado)
    totp_secret = pyotp.random_base32()
    
    # Inicializar el motor facial centralizado (que carga YuNet y SFace)
    face_engine = FaceEngine()
    
    # Iniciar captura de cámara local
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara. Verifique que esté conectada.")
        sys.exit(1)
        
    # Configurar resolución de cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    face_engine.detector.setInputSize((w, h))
    
    print("\n" + "*" * 50)
    print(" INSTRUCCIONES EN PANTALLA:")
    print(" - Asegúrese de tener el rostro despejado (SIN GORRAS, SIN LENTES).")
    print(" - Presione la tecla 's' para capturar el rostro y guardar.")
    print(" - Presione la tecla 'q' para cancelar y salir.")
    print("*" * 50 + "\n")
    
    embedding_capturado = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Falla al leer frame de la cámara.")
            break
            
        # Espejar el frame para comodidad del usuario
        frame = cv2.flip(frame, 1)
        
        # Detectar rostros
        _, faces = face_engine.detector.detect(frame)
        
        # Copia del frame para dibujar los overlays
        display_frame = frame.copy()
        
        # Dibujar pautas del aula en pantalla
        cv2.putText(display_frame, "REGISTRO BIOMETRICO", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, "REGLA: Rostro despejado (Sin gorra/lentes)", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(display_frame, f"Alumno: {nombre} ({alumno_id})", (20, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        rostro_detectado = False
        box_color = (0, 0, 255) # Rojo por defecto (sin rostros o no apto)
        
        if faces is not None and len(faces) > 0:
            rostro_detectado = True
            face = faces[0]  # Tomar el primer rostro detectado
            
            # Obtener caja delimitadora
            box = list(map(int, face[0:4]))
            x, y, w_box, h_box = box[0], box[1], box[2], box[3]
            
            # Si el rostro está en un rango adecuado, colorear en verde
            box_color = (0, 255, 0)
            cv2.rectangle(display_frame, (x, y), (x + w_box, y + h_box), box_color, 2)
            
            # Dibujar puntos clave (landmarks: ojos, nariz, comisuras de boca)
            landmarks = list(map(int, face[4:14]))
            for i in range(5):
                cv2.circle(display_frame, (landmarks[i*2], landmarks[i*2+1]), 3, (255, 0, 255), -1)
                
            cv2.putText(display_frame, "Listo para Captura ('s')", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        else:
            cv2.putText(display_frame, "Buscando rostro...", (20, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        cv2.imshow("Enrolamiento Biometrico", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Capturar y procesar rostro
        if key == ord('s'):
            if rostro_detectado:
                print("\nProcesando rostro capturado biométricamente...")
                try:
                    # Alinear y recortar rostro
                    face_aligned = face_engine.recognizer.alignCrop(frame, faces[0])
                    
                    # Generar vector numérico (embedding)
                    feat = face_engine.recognizer.feature(face_aligned)
                    embedding_capturado = feat[0].copy()
                    
                    # CUMPLIR REGLA DE PRIVACIDAD: Borrar inmediatamente imágenes de RAM
                    del frame
                    del display_frame
                    del face_aligned
                    cv2.destroyAllWindows()
                    break
                except Exception as ex:
                    print(f"[ERROR] Error al extraer embedding facial: {ex}")
            else:
                print("\n[ALERTA] No se detecta ningún rostro despejado en el cuadrante. Intente de nuevo.")
                
        elif key == ord('q'):
            print("\nEnrolamiento cancelado por el usuario.")
            cv2.destroyAllWindows()
            break
            
    cap.release()
    
    if embedding_capturado is not None:
        # Registrar en la base de datos
        exito = register_student(alumno_id, nombre, embedding_capturado, totp_secret)
        
        if exito:
            # Mostrar URI de configuración TOTP para apps móviles
            totp = pyotp.TOTP(totp_secret, interval=TOTP_INTERVAL)
            uri = totp.provisioning_uri(name=alumno_id, issuer_name="REC-Asistencia")
            
            print("\n" + "=" * 60)
            print("   DATOS DE CONFIGURACIÓN TOTP DEL ALUMNO")
            print("=" * 60)
            print(f"Semilla Secreta (Base32): {totp_secret}")
            print(f"Enlace de aprovisionamiento QR (para app móvil):")
            print(f"{uri}")
            print("=" * 60)
            print("Escanee este URI en un generador TOTP o use la App de Contingencia del Alumno.")
            print("=" * 60 + "\n")
    else:
        print("[ALERTA] No se capturó ninguna firma biométrica. Registro incompleto.")

if __name__ == "__main__":
    main()
