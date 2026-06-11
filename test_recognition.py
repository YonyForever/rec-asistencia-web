import cv2
import time
import sys
from core.face_engine import FaceEngine
from core.config import CAMERA_INDEX

def main():
    print("=" * 60)
    print("   SISTEMA DE ASISTENCIA BIOMÉTRICA - PRUEBA DE RECONOCIMIENTO")
    print("=" * 60)

    # 1. Inicializar el motor facial
    engine = FaceEngine()
    
    # 2. Cargar cache de alumnos
    exito = engine.reload_students_cache()
    if not exito:
        print("[ADVERTENCIA] No hay alumnos cargados en memoria RAM. Registre al menos uno ejecutando enroll.py.")

    # 3. Inicializar captura de cámara
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara local.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    engine.detector.setInputSize((w, h))

    print("\n" + "*" * 50)
    print(" INSTRUCCIONES DE RECONOCIMIENTO:")
    print(" - Mire a la cámara con el rostro despejado.")
    print(" - El sistema buscará coincidencias en SQLite en tiempo real.")
    print(" - Presione 'q' para salir de la prueba.")
    print("*" * 50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Error al leer frame de la cámara.")
            break

        # Espejar el frame para comodidad del docente
        frame = cv2.flip(frame, 1)
        
        # Copia del frame para dibujar las alertas gráficas
        display_frame = frame.copy()

        # Dibujar pautas del aula en pantalla
        cv2.putText(display_frame, "MODULO DE RECONOCIMIENTO ACTIVO", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Detectar rostro
        face_data = engine.detect_face(frame)

        if face_data is not None:
            # 1. Obtener coordenadas del rostro
            box = list(map(int, face_data[0:4]))
            x, y, w_box, h_box = box[0], box[1], box[2], box[3]

            # 2. Extraer firma biométrica y medir tiempo de emparejamiento
            t_start = time.perf_counter()
            try:
                # Extrae el vector de 128 floats
                current_embedding = engine.extract_embedding(frame, face_data)
                
                # CUMPLIMIENTO ESTRICTO DE PRIVACIDAD: 
                # Liberar inmediatamente el frame original y la memoria RAM del procesamiento
                del frame
                
                # Realizar el emparejamiento contra la RAM
                alumno_id, nombre, score = engine.match_face(current_embedding)
                t_end = time.perf_counter()
                
                match_time_ms = (t_end - t_start) * 1000

                if alumno_id is not None:
                    # Coincidencia exitosa (Match)
                    box_color = (0, 255, 0) # Verde
                    lbl_id = f"ID: {alumno_id}"
                    lbl_name = nombre
                    lbl_status = f"MATCH: {score:.3f} | {match_time_ms:.2f}ms"
                else:
                    # Desconocido
                    box_color = (0, 165, 255) # Naranja / Amarillo
                    lbl_id = "Desconocido"
                    lbl_name = "Sin registro biometrico"
                    lbl_status = f"No Match: {score:.3f} | {match_time_ms:.2f}ms"

            except Exception as e:
                box_color = (0, 0, 255) # Rojo
                lbl_id = "Error de procesamiento"
                lbl_name = str(e)
                lbl_status = ""

            # Dibujar caja delimitadora sobre el clon de pantalla
            cv2.rectangle(display_frame, (x, y), (x + w_box, y + h_box), box_color, 2)

            # Dibujar puntos clave (landmarks)
            landmarks = list(map(int, face_data[4:14]))
            for i in range(5):
                cv2.circle(display_frame, (landmarks[i*2], landmarks[i*2+1]), 3, (255, 0, 255), -1)

            # Dibujar textos sobre el rostro
            cv2.putText(display_frame, lbl_id, (x, y - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            cv2.putText(display_frame, lbl_name, (x, y - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            cv2.putText(display_frame, lbl_status, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            # No hay rostros en escena
            cv2.putText(display_frame, "Buscando rostro en escena...", (20, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Mostrar frame
        cv2.imshow("Prueba de Reconocimiento Facial (Fase 2)", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nPrueba de reconocimiento finalizada.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
