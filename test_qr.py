import cv2
import numpy as np
import sys
from core.qr_engine import QREngine
from core.config import CAMERA_INDEX

def main():
    print("=" * 60)
    print("   SISTEMA DE ASISTENCIA BIOMÉTRICA - PRUEBA DE LECTOR QR / TOTP")
    print("=" * 60)

    # 1. Inicializar el motor QR
    qr_engine = QREngine()

   # 2. Inicializar captura de cámara (Usa el origen configurado en core/config.py)
    try:
        cam_source = int(CAMERA_INDEX)
    except ValueError:
        cam_source = CAMERA_INDEX

    cap = cv2.VideoCapture(cam_source)
    
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la transmisión de la cámara en el origen: {cam_source}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "*" * 50)
    print(" INSTRUCCIONES DE LECTOR QR:")
    print(" - Abra 'student_app.html' en su celular.")
    print(" - Acerque la pantalla de su celular con el QR a la cámara.")
    print(" - El sistema validará la autenticidad y el tiempo del token.")
    print(" - Presione 'q' para salir de la prueba.")
    print("*" * 50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Error al leer frame de la cámara.")
            break

        # Espejar para comodidad del docente
        frame = cv2.flip(frame, 1)
        
        # Copia del frame para dibujar los overlays gráficos
        display_frame = frame.copy()

        # Dibujar pautas en pantalla
        cv2.putText(display_frame, "MODULO DE LECTOR QR ACTIVO", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, "Acerque el QR dinamico del celular", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Detectar y decodificar código QR
        decoded_info, points = qr_engine.detect_and_decode(frame)

        if decoded_info is not None:
            # 1. Validar la firma del QR
            success, alumno_id, nombre, message = qr_engine.validate_student_qr(decoded_info)
            
            # Definir colores y textos según el resultado de la validación
            if success:
                box_color = (0, 255, 0)  # Verde: Auténtico e instantáneo
                status_text = "ACCESO CONCEDIDO (QR VALIDO)"
                detail_text = f"{nombre} ({alumno_id})"
            else:
                box_color = (0, 0, 255)  # Rojo: Fallo en token o no registrado
                status_text = "ACCESO DENEGADO"
                detail_text = message

            # 2. Dibujar polígono delimitador alrededor del QR
            # points tiene forma (1, 4, 2)
            if points is not None and len(points) > 0:
                pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], True, box_color, 3)
                
                # Obtener la esquina superior izquierda del QR para pintar los textos cerca
                # points[0][0] contiene [x, y]
                x_pts = [p[0] for p in points[0]]
                y_pts = [p[1] for p in points[0]]
                min_x = int(min(x_pts))
                min_y = int(min(y_pts))
                
                # Dibujar textos cerca del QR
                cv2.putText(display_frame, status_text, (min_x, max(15, min_y - 30)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                cv2.putText(display_frame, detail_text, (min_x, max(15, min_y - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Imprimir en consola para depuración
            print(f"\r[LECTOR QR] Decodificado: '{decoded_info}' | {status_text} - {detail_text}", end="")
            sys.stdout.flush()
            
        # Mostrar el frame de video
        cv2.imshow("Prueba de Lector QR y Contingencia (Fase 3)", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nPrueba de lector QR finalizada.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
