import os
import sys
import time
import queue
import threading
import datetime
from pathlib import Path
from PIL import Image, ImageTk

import cv2
import numpy as np
import pyotp
import customtkinter as ctk
import telebot  # pyTelegramBotAPI
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    EXCEL_DISPONIBLE = True
except ImportError:
    EXCEL_DISPONIBLE = False

from core.config import (
    CAMERA_INDEX,
    TOTP_INTERVAL,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    BASE_DIR
)
from core.face_engine import FaceEngine
from core.qr_engine import QREngine
from database.connection import get_session, init_db
from database.models import Alumno, ClaseConfig, Sesion, Asistencia

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Paleta de colores centralizada ──────────────────────────────────────────
C = {
    "bg":          "#0d0d14",   # fondo global
    "surface":     "#12121e",   # superficies / cards
    "surface2":    "#1a1a2e",   # inputs / filas
    "border":      "#1e1e35",   # bordes sutiles
    "border2":     "#252540",   # bordes secundarios
    "accent":      "#6d28d9",   # violeta principal
    "accent_h":    "#7c3aed",   # hover del acento
    "accent_soft": "#1e1065",   # fondo badge facial
    "text":        "#e2e8f0",   # texto principal
    "muted":       "#6b7280",   # texto secundario
    "green":       "#22c55e",   # presente
    "green_bg":    "#0d2a1a",   # fondo presente
    "yellow":      "#f59e0b",   # retardo
    "yellow_bg":   "#2a1f00",   # fondo retardo
    "red":         "#ef4444",   # ausente / error
    "red_bg":      "#2a0d0d",   # fondo ausente
    "qr_text":     "#34d399",   # badge QR
    "qr_bg":       "#0c2a1f",   # fondo badge QR
    "indigo_text": "#818cf8",   # badge FACIAL
}


class CameraScannerThread(threading.Thread):
    """
    Hilo de background para capturar la cámara a 30 FPS,
    procesar detección facial con YuNet + SFace y QR con OpenCV,
    y enviar frames + resultados al hilo principal (GUI).
    """
    def __init__(self, camera_index, face_engine, qr_engine, result_queue):
        super().__init__()
        self.daemon = True
        self.camera_index = camera_index
        self.face_engine = face_engine
        self.qr_engine = qr_engine
        self.result_queue = result_queue
        self.running = False
        self.cap = None

    def run(self):
        self.running = True
        try:
            cam_source = int(self.camera_index)
        except ValueError:
            cam_source = self.camera_index

        print(f"[Cámara] Inicializando desde: {cam_source}")
        self.cap = cv2.VideoCapture(cam_source)
        if not self.cap.isOpened():
            self.result_queue.put(("error", f"No se pudo conectar: {cam_source}"))
            self.running = False
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.face_engine.detector.setInputSize((w, h))

        last_match_times = {}

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            detection_result = None

            # QR
            decoded_info, points = self.qr_engine.detect_and_decode(frame)
            if decoded_info:
                success, alumno_id, nombre, message = self.qr_engine.validate_student_qr(decoded_info)
                if points is not None and len(points) > 0:
                    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                    box_color = (52, 211, 153) if success else (239, 68, 68)
                    cv2.polylines(display_frame, [pts], True, box_color, 3)
                now = time.time()
                key = f"qr_{alumno_id or 'unknown'}"
                if now - last_match_times.get(key, 0) > 3.0:
                    last_match_times[key] = now
                    detection_result = {
                        "metodo": "QR",
                        "exito": success,
                        "alumno_id": alumno_id,
                        "nombre": nombre,
                        "mensaje": message
                    }
            else:
                # FACIAL
                face_data = self.face_engine.detect_face(frame)
                if face_data is not None:
                    box = list(map(int, face_data[0:4]))
                    x, y, w_box, h_box = box[0], box[1], box[2], box[3]
                    try:
                        embedding = self.face_engine.extract_embedding(frame, face_data)
                        alumno_id, nombre, score = self.face_engine.match_face(embedding)
                        box_color = (52, 211, 153) if alumno_id else (245, 158, 11)
                        cv2.rectangle(display_frame, (x, y), (x + w_box, y + h_box), box_color, 2)
                        landmarks = list(map(int, face_data[4:14]))
                        for i in range(5):
                            cv2.circle(display_frame, (landmarks[i*2], landmarks[i*2+1]), 3, (129, 140, 248), -1)
                        now = time.time()
                        key = f"facial_{alumno_id or 'unknown'}"
                        if now - last_match_times.get(key, 0) > 3.0:
                            last_match_times[key] = now
                            if alumno_id:
                                detection_result = {
                                    "metodo": "FACIAL",
                                    "exito": True,
                                    "alumno_id": alumno_id,
                                    "nombre": nombre,
                                    "score": score
                                }
                            else:
                                detection_result = {"metodo": "FACIAL", "exito": False, "score": score}
                    except Exception as e:
                        print(f"[ScannerThread] Error facial: {e}")

            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            self.result_queue.put(("frame", (frame_rgb, detection_result)))
            time.sleep(0.01)

        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass


class RecDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("REC — Sistema Híbrido de Asistencia Biométrica")
        self.geometry("1400x860")
        self.resizable(False, False)
        self.attributes("-fullscreen", True)
        self.configure(fg_color=C["bg"])

        self.face_engine = FaceEngine()
        self.face_engine.reload_students_cache()
        self.qr_engine = QREngine()

        self.session_active = False
        self.active_session_id = None
        self.active_class_name = ""
        self.hora_apertura = None

        self.result_queue = queue.Queue()
        self.scanner_thread = None

        self.setup_ui()
        self.load_classes_into_dropdown()
        self.add_log("Sistema inicializado. Listo para abrir sesión de asistencia.")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ── UI ──────────────────────────────────────────────────────────────────

    def setup_ui(self):
        # ── TOPBAR ──────────────────────────────────────────────────────────
        topbar = ctk.CTkFrame(self, fg_color=C["surface"], corner_radius=0, height=64)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        # Logo / marca
        logo_box = ctk.CTkFrame(topbar, fg_color=C["accent"], corner_radius=10, width=40, height=40)
        logo_box.place(x=18, y=12)
        logo_box.pack_propagate(False)
        ctk.CTkLabel(
            logo_box, text="REC",
            font=ctk.CTkFont(family="Consolas", size=13, weight="bold"),
            text_color="#ffffff"
        ).place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            topbar,
            text="Asistencia Híbrida · Facial / QR",
            font=ctk.CTkFont(family="Outfit", size=17, weight="bold"),
            text_color=C["text"]
        ).place(x=68, y=10)

        ctk.CTkLabel(
            topbar,
            text="Panel local del docente · Sistema REC",
            font=ctk.CTkFont(family="Outfit", size=11),
            text_color=C["muted"]
        ).place(x=69, y=36)

        # Indicador de sesión
        self.indicator_canvas = ctk.CTkLabel(
            topbar,
            text="● SESIÓN CERRADA",
            font=ctk.CTkFont(family="Outfit", size=13, weight="bold"),
            text_color=C["red"]
        )
        self.indicator_canvas.place(relx=0.72, rely=0.5, anchor="center")

        # Botón enrolar
        self.btn_enroll = ctk.CTkButton(
            topbar,
            text="+ Enrolar alumno",
            fg_color=C["accent"],
            hover_color=C["accent_h"],
            text_color="#ffffff",
            font=ctk.CTkFont(family="Outfit", size=12, weight="bold"),
            height=34, width=148,
            corner_radius=8,
            command=self.open_enrollment_window
        )
        self.btn_enroll.place(relx=0.88, rely=0.5, anchor="center")

        # ── CUERPO PRINCIPAL ─────────────────────────────────────────────────
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=14, pady=12)
        body.grid_columnconfigure(0, weight=7)
        body.grid_columnconfigure(1, weight=3)
        body.grid_rowconfigure(0, weight=1)

        # ── COLUMNA IZQUIERDA: cámara ────────────────────────────────────────
        left = ctk.CTkFrame(body, fg_color="transparent")
        left.grid(row=0, column=0, padx=(0, 10), sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        # Card cámara
        cam_card = ctk.CTkFrame(left, fg_color=C["surface"], corner_radius=16, border_width=1, border_color=C["border"])
        cam_card.grid(row=0, column=0, sticky="nsew")

        # Header de la cámara
        cam_header = ctk.CTkFrame(cam_card, fg_color="transparent", height=42)
        cam_header.pack(fill="x", padx=14, pady=(12, 0))

        ctk.CTkLabel(
            cam_header,
            text="Vista en vivo",
            font=ctk.CTkFont(family="Outfit", size=13, weight="bold"),
            text_color=C["muted"]
        ).pack(side="left")

        # Badges de modo
        badge_f = ctk.CTkFrame(cam_header, fg_color=C["accent_soft"], corner_radius=10, width=76, height=24)
        badge_f.pack(side="right", padx=(6, 0))
        badge_f.pack_propagate(False)
        ctk.CTkLabel(badge_f, text="● Facial", font=ctk.CTkFont(family="Outfit", size=11, weight="bold"),
                     text_color=C["indigo_text"]).place(relx=0.5, rely=0.5, anchor="center")

        badge_q = ctk.CTkFrame(cam_header, fg_color=C["qr_bg"], corner_radius=10, width=60, height=24)
        badge_q.pack(side="right", padx=(0, 4))
        badge_q.pack_propagate(False)
        ctk.CTkLabel(badge_q, text="▣ QR", font=ctk.CTkFont(family="Outfit", size=11, weight="bold"),
                     text_color=C["qr_text"]).place(relx=0.5, rely=0.5, anchor="center")

        # Área de video — Canvas como contenedor raíz
        import tkinter as tk
        self.cam_canvas = tk.Canvas(cam_card, bg="#12121e", bd=0, highlightthickness=0)
        self.cam_canvas.pack(fill="both", expand=True, padx=6, pady=6)

        # Label que muestra el feed de la cámara (encima del canvas)
        self.camera_label = ctk.CTkLabel(
            self.cam_canvas,
            text="",
            font=ctk.CTkFont(family="Outfit", size=15),
            text_color=C["muted"]
        )
        self.cam_canvas.create_window(0, 0, anchor="nw", window=self.camera_label,
                                       tags="cam_win")

        # Texto de espera centrado
        self.cam_idle_text = self.cam_canvas.create_text(
            0, 0, text="Inicie sesión para activar la cámara",
            fill="#4b5563", font=("Outfit", 14), tags="idle_text"
        )

        # Esquinas del scanner (8 líneas: 2 por esquina)
        corner_color = "#6d28d9"
        clen = 28  # largo de cada trazo
        self._scan_corners = []
        for tag in ["c_tl", "c_tr", "c_bl", "c_br"]:
            l1 = self.cam_canvas.create_line(0, 0, 0, 0, fill=corner_color, width=2, tags=tag)
            l2 = self.cam_canvas.create_line(0, 0, 0, 0, fill=corner_color, width=2, tags=tag)
            self._scan_corners.append((l1, l2))

        # Línea de barrido
        self._scan_line = self.cam_canvas.create_line(0, 0, 0, 0,
                                                       fill="#6d28d9", width=2,
                                                       tags="scan_line")
        self._scan_line_y = 0
        self._scan_dir = 1
        self._scan_alpha_step = 0

        # Redibujar esquinas al redimensionar
        self.cam_canvas.bind("<Configure>", self._on_cam_configure)

        # Arrancar animación del scanner idle
        self._animate_scanner()

        # Banner de estado
        self.status_banner = ctk.CTkFrame(
            left, fg_color=C["surface2"], corner_radius=12,
            border_width=1, border_color=C["red"], height=52
        )
        self.status_banner.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.status_banner.pack_propagate(False)

        self.status_banner_text = ctk.CTkLabel(
            self.status_banner,
            text="Scanner en espera · Inicie sesión de clase para comenzar la detección.",
            font=ctk.CTkFont(family="Outfit", size=13, weight="bold"),
            text_color=C["red"]
        )
        self.status_banner_text.pack(fill="both", expand=True, padx=14)

        # ── COLUMNA DERECHA ───────────────────────────────────────────────────
        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=0)  # config sesion
        right.rowconfigure(1, weight=0)  # metricas
        right.rowconfigure(2, weight=1)  # lista asistencia (expande)
        right.rowconfigure(3, weight=0)  # consola
        right.columnconfigure(0, weight=1)

        # ── 1. Config de sesión ───────────────────────────────────────────────
        cfg = ctk.CTkFrame(right, fg_color=C["surface"], corner_radius=16,
                           border_width=1, border_color=C["border"])
        cfg.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        ctk.CTkLabel(
            cfg, text="Configuración de clase",
            font=ctk.CTkFont(family="Outfit", size=13, weight="bold"),
            text_color=C["indigo_text"]
        ).grid(row=0, column=0, columnspan=4, padx=16, pady=(14, 8), sticky="w")

        # Materia
        ctk.CTkLabel(cfg, text="Materia", font=ctk.CTkFont(family="Outfit", size=11),
                     text_color=C["muted"]).grid(row=1, column=0, padx=(16, 4), pady=4, sticky="w")
        self.dropdown_materia = ctk.CTkOptionMenu(
            cfg, values=["Cargando..."],
            fg_color=C["surface2"], button_color=C["accent"],
            button_hover_color=C["accent_h"], text_color=C["text"],
            font=ctk.CTkFont(family="Outfit", size=12),
            width=220, corner_radius=8
        )
        self.dropdown_materia.grid(row=1, column=1, columnspan=3, padx=(0, 16), pady=4, sticky="ew")

        # Límites
        ctk.CTkLabel(cfg, text="Presente (min)", font=ctk.CTkFont(family="Outfit", size=11),
                     text_color=C["muted"]).grid(row=2, column=0, padx=(16, 4), pady=4, sticky="w")
        self.entry_presente = ctk.CTkEntry(
            cfg, width=60, height=32, corner_radius=8,
            fg_color=C["surface2"], border_color=C["border2"],
            text_color=C["text"], font=ctk.CTkFont(family="Outfit", size=12),
            justify="center"
        )
        self.entry_presente.insert(0, "5")
        self.entry_presente.grid(row=2, column=1, padx=4, pady=4, sticky="w")

        ctk.CTkLabel(cfg, text="Retardo (min)", font=ctk.CTkFont(family="Outfit", size=11),
                     text_color=C["muted"]).grid(row=2, column=2, padx=(8, 4), pady=4, sticky="w")
        self.entry_tarde = ctk.CTkEntry(
            cfg, width=60, height=32, corner_radius=8,
            fg_color=C["surface2"], border_color=C["border2"],
            text_color=C["text"], font=ctk.CTkFont(family="Outfit", size=12),
            justify="center"
        )
        self.entry_tarde.insert(0, "20")
        self.entry_tarde.grid(row=2, column=3, padx=(0, 16), pady=4, sticky="w")

        cfg.columnconfigure(1, weight=1)

        self.btn_session_control = ctk.CTkButton(
            cfg,
            text="▶  Iniciar clase",
            fg_color=C["green"], hover_color="#16a34a",
            text_color="#0a1a0a",
            font=ctk.CTkFont(family="Outfit", size=14, weight="bold"),
            height=40, corner_radius=10,
            command=self.toggle_session
        )
        self.btn_session_control.grid(row=3, column=0, columnspan=4, padx=16, pady=(8, 16), sticky="ew")

        # ── 2. Métricas ───────────────────────────────────────────────────────
        mf = ctk.CTkFrame(right, fg_color="transparent")
        mf.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        mf.columnconfigure((0, 1, 2), weight=1)

        for col, (val, label, color, bg) in enumerate([
            ("0", "Presentes", C["green"], C["green_bg"]),
            ("0", "Retardos",  C["yellow"], C["yellow_bg"]),
            ("0", "Ausentes",  C["red"], C["red_bg"]),
        ]):
            card = ctk.CTkFrame(mf, fg_color=C["surface2"], corner_radius=12,
                                border_width=1, border_color=C["border2"])
            pad_l = 0 if col == 0 else 4
            pad_r = 4 if col < 2 else 0
            card.grid(row=0, column=col, padx=(pad_l, pad_r), sticky="ew")
            card.columnconfigure(0, weight=1)

            attr_name = ["metric_present", "metric_late", "metric_absent"][col]
            lbl = ctk.CTkLabel(card, text=val,
                               font=ctk.CTkFont(family="Outfit", size=26, weight="bold"),
                               text_color=color)
            lbl.pack(pady=(10, 0))
            setattr(self, attr_name, lbl)

            ctk.CTkLabel(card, text=label,
                         font=ctk.CTkFont(family="Outfit", size=10),
                         text_color=C["muted"]).pack(pady=(0, 10))

        self.count_present = 0
        self.count_late    = 0
        self.count_absent  = 0

        # ── 3. Lista de asistencia ────────────────────────────────────────────
        list_frame = ctk.CTkFrame(right, fg_color=C["surface"], corner_radius=16,
                                  border_width=1, border_color=C["border"])
        list_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 8))

        ctk.CTkLabel(list_frame, text="Registro en tiempo real",
                     font=ctk.CTkFont(family="Outfit", size=13, weight="bold"),
                     text_color=C["indigo_text"]).pack(anchor="w", padx=14, pady=(14, 2))

        col_hdr = ctk.CTkFrame(list_frame, fg_color="transparent")
        col_hdr.pack(fill="x", padx=14, pady=(0, 4))
        for text, w in [("Estudiante", 160), ("Hora", 65), ("Canal", 70), ("Estado", 80)]:
            ctk.CTkLabel(col_hdr, text=text, width=w, anchor="w",
                         font=ctk.CTkFont(family="Outfit", size=10, weight="bold"),
                         text_color=C["muted"]).pack(side="left", padx=3)

        sep = ctk.CTkFrame(list_frame, fg_color=C["border"], height=1)
        sep.pack(fill="x", padx=14)

        self.table_scroll = ctk.CTkScrollableFrame(
            list_frame, fg_color="transparent", corner_radius=0
        )
        self.table_scroll.pack(fill="both", expand=True, padx=6, pady=6)

        # ── 4. Consola ────────────────────────────────────────────────────────
        console_card = ctk.CTkFrame(right, fg_color=C["surface"], corner_radius=16,
                                    border_width=1, border_color=C["border"], height=118)
        console_card.grid(row=3, column=0, sticky="ew")
        console_card.pack_propagate(False)

        ctk.CTkLabel(console_card, text="Consola de auditoría",
                     font=ctk.CTkFont(family="Outfit", size=10, weight="bold"),
                     text_color=C["muted"]).pack(anchor="w", padx=14, pady=(8, 0))

        self.console_text = ctk.CTkTextbox(
            console_card,
            fg_color="transparent",
            text_color="#4b5563",
            font=ctk.CTkFont(family="Consolas", size=11),
            activate_scrollbars=True
        )
        self.console_text.pack(fill="both", expand=True, padx=8, pady=(2, 6))
        self.console_text.configure(state="disabled")

    # ── Animación del scanner idle ───────────────────────────────────────────

    def _on_cam_configure(self, event):
        """Reposiciona esquinas y texto al cambiar tamaño del canvas."""
        self._draw_scan_corners()
        w, h = event.width, event.height
        self.cam_canvas.coords(self.cam_idle_text, w // 2, h // 2)
        # Reposicionar la ventana del camera_label para cubrir todo
        self.cam_canvas.coords("cam_win", 0, 0)
        self.camera_label.configure(width=w, height=h)

    def _draw_scan_corners(self):
        c = self.cam_canvas
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 10 or H < 10:
            return
        m = 18   # margen desde el borde
        L = 26   # largo del trazo

        coords = [
            # top-left
            ((m, m + L, m, m), (m, m, m + L, m)),
            # top-right
            ((W - m - L, W - m, W - m, W - m), (W - m - L, W - m, W - m, W - m - L)),
            # bottom-left
            ((m, m, m, H - m - L), (m, m + L, H - m, H - m)),
            # bottom-right
            ((W - m, W - m, W - m - L, W - m), (W - m, W - m, W - m, H - m - L)),
        ]
        # Reescribir con coordenadas correctas
        c.coords(self._scan_corners[0][0], m, m, m, m + L)          # TL vertical
        c.coords(self._scan_corners[0][1], m, m, m + L, m)           # TL horizontal
        c.coords(self._scan_corners[1][0], W - m, m, W - m, m + L)  # TR vertical
        c.coords(self._scan_corners[1][1], W - m - L, m, W - m, m)  # TR horizontal
        c.coords(self._scan_corners[2][0], m, H - m, m, H - m - L)  # BL vertical
        c.coords(self._scan_corners[2][1], m, H - m, m + L, H - m)  # BL horizontal
        c.coords(self._scan_corners[3][0], W - m, H - m, W - m, H - m - L)  # BR vertical
        c.coords(self._scan_corners[3][1], W - m - L, H - m, W - m, H - m)  # BR horizontal

    def _animate_scanner(self):
        """Anima la línea de barrido del scanner cuando no hay sesión activa."""
        if not self.winfo_exists():
            return
        try:
            c = self.cam_canvas
            W = c.winfo_width()
            H = c.winfo_height()

            if not self.session_active and W > 10 and H > 10:
                # Mover línea de barrido
                self._scan_line_y += self._scan_dir * 3
                if self._scan_line_y >= H - 20:
                    self._scan_dir = -1
                elif self._scan_line_y <= 20:
                    self._scan_dir = 1

                # Pulso de opacidad con colores alternos (simular fade)
                self._scan_alpha_step = (self._scan_alpha_step + 1) % 20
                pulse_colors = ["#6d28d9", "#7c3aed", "#8b5cf6", "#7c3aed"]
                col = pulse_colors[self._scan_alpha_step % len(pulse_colors)]

                c.coords(self._scan_line, 20, self._scan_line_y, W - 20, self._scan_line_y)
                c.itemconfig(self._scan_line, fill=col)
                c.itemconfig("idle_text", state="normal")

                # Dibujar esquinas
                self._draw_scan_corners()
                for l1, l2 in self._scan_corners:
                    c.itemconfig(l1, state="normal")
                    c.itemconfig(l2, state="normal")

            else:
                # Sesión activa: ocultar overlays del scanner
                c.itemconfig(self._scan_line, state="hidden")
                c.itemconfig("idle_text", state="hidden")
                for l1, l2 in self._scan_corners:
                    c.itemconfig(l1, state="hidden")
                    c.itemconfig(l2, state="hidden")

        except Exception:
            pass

        self.after(40, self._animate_scanner)  # ~25 fps animación

    # ── BD ───────────────────────────────────────────────────────────────────

    def load_classes_into_dropdown(self):
        session = get_session()
        try:
            clases = session.query(ClaseConfig).filter_by(activo=1).all()
            if clases:
                self.classes_map = {f"{c.nombre_materia} ({c.hora_inicio})": c.id for c in clases}
                self.dropdown_materia.configure(values=list(self.classes_map.keys()))
                self.dropdown_materia.set(list(self.classes_map.keys())[0])
            else:
                self.dropdown_materia.configure(values=["Sin materias configuradas"])
                self.dropdown_materia.set("Sin materias configuradas")
        except Exception as e:
            self.add_log(f"Error al cargar materias: {e}")
        finally:
            session.close()

    def add_log(self, message):
        t_str = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{t_str}]  {message}\n"
        self.console_text.configure(state="normal")
        self.console_text.insert("end", line)
        self.console_text.see("end")
        self.console_text.configure(state="disabled")
        try:
            print(line.strip())
        except Exception:
            try:
                encoding = sys.stdout.encoding or 'utf-8'
                print(line.encode(encoding, errors='replace').decode(encoding).strip())
            except Exception:
                pass

    # ── Control de sesión ────────────────────────────────────────────────────

    def toggle_session(self):
        if not self.session_active:
            self.start_session()
        else:
            self.stop_session()

    def start_session(self):
        selected_materia = self.dropdown_materia.get()
        if selected_materia not in getattr(self, "classes_map", {}):
            self.add_log("Error: Seleccione una materia válida.")
            return
        try:
            lim_pres = int(self.entry_presente.get().strip())
            lim_tarde = int(self.entry_tarde.get().strip())
        except ValueError:
            self.add_log("Error: Los límites deben ser números enteros.")
            return

        class_id = self.classes_map[selected_materia]
        session = get_session()
        try:
            nueva_sesion = Sesion(clase_config_id=class_id, estado="ABIERTA")
            session.add(nueva_sesion)
            session.commit()

            self.active_session_id = nueva_sesion.id
            self.active_class_name = selected_materia.split(" (")[0]
            self.hora_apertura = datetime.datetime.now()
            self.local_limite_presente = lim_pres
            self.local_limite_tarde = lim_tarde
            self.session_active = True

            # Reset métricas
            self.count_present = self.count_late = self.count_absent = 0
            self._refresh_metrics()

            # UI
            self.indicator_canvas.configure(text="● CLASE EN CURSO", text_color=C["green"])
            self.status_banner.configure(border_color=C["green"])
            self.status_banner_text.configure(
                text="Sistema híbrido activo · Acerque su rostro o código QR al lente",
                text_color=C["green"]
            )
            self.btn_session_control.configure(
                text="■  Finalizar clase",
                fg_color=C["red"], hover_color="#b91c1c", text_color="#fff0f0"
            )
            self.dropdown_materia.configure(state="disabled")
            self.entry_presente.configure(state="disabled")
            self.entry_tarde.configure(state="disabled")

            self.add_log(
                f"Sesión #{self.active_session_id} '{self.active_class_name}' "
                f"iniciada a las {self.hora_apertura.strftime('%H:%M:%S')}"
            )

            for widget in self.table_scroll.winfo_children():
                widget.destroy()

            self.scanner_thread = CameraScannerThread(
                camera_index=CAMERA_INDEX,
                face_engine=self.face_engine,
                qr_engine=self.qr_engine,
                result_queue=self.result_queue
            )
            self.scanner_thread.start()
            self.after(50, self.process_camera_queue)

        except Exception as e:
            session.rollback()
            self.add_log(f"Error al abrir sesión en BD: {e}")
        finally:
            session.close()

    def stop_session(self):
        if not self.session_active:
            return

        self.add_log("Finalizando clase y cerrando cámara...")

        if self.scanner_thread:
            self.scanner_thread.stop()
            self.scanner_thread.join(timeout=2.0)
            self.scanner_thread = None

        session = get_session()
        try:
            sesion_db = session.query(Sesion).filter_by(id=self.active_session_id).first()
            if sesion_db:
                sesion_db.hora_cierre = datetime.datetime.now()
                sesion_db.estado = "CERRADA"
                session.commit()
            self.add_log(f"Sesión #{self.active_session_id} cerrada en base de datos.")
            self.generate_and_send_report()
        except Exception as e:
            session.rollback()
            self.add_log(f"Error al cerrar sesión: {e}")
        finally:
            session.close()

        self.session_active = False
        self.active_session_id = None

        self.indicator_canvas.configure(text="● SESIÓN CERRADA", text_color=C["red"])
        self.status_banner.configure(border_color=C["red"])
        self.status_banner_text.configure(
            text="Scanner en espera · Inicie sesión de clase para comenzar la detección.",
            text_color=C["red"]
        )
        self.btn_session_control.configure(
            text="▶  Iniciar clase",
            fg_color=C["green"], hover_color="#16a34a", text_color="#0a1a0a"
        )

        try:
            self.camera_label_image = None
            self.camera_label._label.configure(image="", text="")
        except Exception:
            pass

        self.dropdown_materia.configure(state="normal")
        self.entry_presente.configure(state="normal")
        self.entry_tarde.configure(state="normal")

    # ── Procesamiento de frames ───────────────────────────────────────────────

    def process_camera_queue(self):
        if not self.session_active:
            return

        frame_data = None
        try:
            while True:
                msg_type, data = self.result_queue.get_nowait()
                if msg_type == "error":
                    self.add_log(f"[CÁMARA] {data}")
                    self.stop_session()
                    return
                elif msg_type == "frame":
                    frame_data = data
                self.result_queue.task_done()
        except queue.Empty:
            pass

        if frame_data:
            frame_rgb, detection = frame_data
            # Escalar al tamaño actual del canvas
            cw = self.cam_canvas.winfo_width()
            ch = self.cam_canvas.winfo_height()
            if cw > 10 and ch > 10:
                img_pil = Image.fromarray(frame_rgb).resize((cw, ch), Image.BILINEAR)
            else:
                img_pil = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.camera_label_image = img_tk
            try:
                self.camera_label._label.configure(image=img_tk, text="")
                self.camera_label.configure(width=cw, height=ch)
            except Exception:
                pass

            if detection:
                self.handle_detection(detection)

        self.after(15, self.process_camera_queue)

    def handle_detection(self, detection):
        if not self.session_active or not self.active_session_id:
            return

        metodo = detection["metodo"]
        exito  = detection["exito"]

        if not exito:
            if metodo == "FACIAL":
                score = detection.get("score", 0.0)
                self.status_banner_text.configure(
                    text=f"Rostro desconocido en escena — confianza: {score:.2f}",
                    text_color=C["yellow"]
                )
                self.status_banner.configure(border_color=C["yellow"])
            elif metodo == "QR":
                msg = detection.get("mensaje", "Token inválido")
                self.status_banner_text.configure(
                    text=f"Acceso QR denegado: {msg}",
                    text_color=C["red"]
                )
                self.status_banner.configure(border_color=C["red"])
            return

        alumno_id = detection["alumno_id"]
        nombre    = detection["nombre"]

        now = datetime.datetime.now()
        elapsed = (now - self.hora_apertura).total_seconds() / 60.0

        if elapsed <= self.local_limite_presente:
            estado = "PRESENTE"
            color  = C["green"]
            bg     = C["green_bg"]
        elif elapsed <= self.local_limite_tarde:
            estado = "RETARDO"
            color  = C["yellow"]
            bg     = C["yellow_bg"]
        else:
            estado = "FALTA"
            color  = C["red"]
            bg     = C["red_bg"]

        session = get_session()
        try:
            existente = session.query(Asistencia).filter_by(
                sesion_id=self.active_session_id,
                alumno_id=alumno_id
            ).first()

            if existente:
                self.status_banner_text.configure(
                    text=f"Hola {nombre} — ya tienes asistencia registrada ({existente.estado})",
                    text_color=C["indigo_text"]
                )
                self.status_banner.configure(border_color=C["accent"])
                return

            nueva = Asistencia(
                sesion_id=self.active_session_id,
                alumno_id=alumno_id,
                metodo=metodo,
                estado=estado
            )
            session.add(nueva)
            session.commit()

            self.add_log(f"REGISTRADO: {nombre} ({alumno_id}) · {estado} · {metodo}")

            self.status_banner_text.configure(
                text=f"Asistencia registrada: {nombre} → {estado}",
                text_color=color
            )
            self.status_banner.configure(border_color=color)

            # Actualizar métricas
            if estado == "PRESENTE":
                self.count_present += 1
            elif estado == "RETARDO":
                self.count_late += 1
            else:
                self.count_absent += 1
            self._refresh_metrics()

            self.add_student_row_ui(nombre, now.strftime("%H:%M:%S"), metodo, estado, color, bg)

        except Exception as e:
            session.rollback()
            self.add_log(f"Error al registrar asistencia: {e}")
        finally:
            session.close()

    def _refresh_metrics(self):
        self.metric_present.configure(text=str(self.count_present))
        self.metric_late.configure(text=str(self.count_late))
        self.metric_absent.configure(text=str(self.count_absent))

    def add_student_row_ui(self, nombre, hora, metodo, estado, color, bg):
        row = ctk.CTkFrame(self.table_scroll, fg_color=C["surface2"], corner_radius=10,
                           border_width=1, border_color=C["border2"], height=38)
        row.pack(fill="x", pady=3, padx=2)
        row.pack_propagate(False)

        # Iniciales / avatar
        initials = "".join(p[0].upper() for p in nombre.split()[:2])
        av = ctk.CTkFrame(row, fg_color=bg, corner_radius=8, width=28, height=28)
        av.place(x=6, rely=0.5, anchor="w")
        av.pack_propagate(False)
        ctk.CTkLabel(av, text=initials[:2], font=ctk.CTkFont(family="Outfit", size=9, weight="bold"),
                     text_color=color).place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(row, text=nombre, width=148, anchor="w",
                     font=ctk.CTkFont(family="Outfit", size=11),
                     text_color=C["text"]).place(x=40, rely=0.5, anchor="w")

        ctk.CTkLabel(row, text=hora, width=62, anchor="w",
                     font=ctk.CTkFont(family="Consolas", size=10),
                     text_color=C["muted"]).place(x=192, rely=0.5, anchor="w")

        # Badge método
        m_color = C["indigo_text"] if metodo == "FACIAL" else C["qr_text"]
        m_bg    = C["accent_soft"]  if metodo == "FACIAL" else C["qr_bg"]
        mb = ctk.CTkFrame(row, fg_color=m_bg, corner_radius=8, width=58, height=20)
        mb.place(x=258, rely=0.5, anchor="w")
        mb.pack_propagate(False)
        ctk.CTkLabel(mb, text=metodo[:6], font=ctk.CTkFont(family="Outfit", size=9, weight="bold"),
                     text_color=m_color).place(relx=0.5, rely=0.5, anchor="center")

        # Badge estado
        sb = ctk.CTkFrame(row, fg_color=bg, corner_radius=8, width=68, height=20)
        sb.place(x=322, rely=0.5, anchor="w")
        sb.pack_propagate(False)
        ctk.CTkLabel(sb, text=estado, font=ctk.CTkFont(family="Outfit", size=9, weight="bold"),
                     text_color=color).place(relx=0.5, rely=0.5, anchor="center")

    # ── Reportes y Telegram ───────────────────────────────────────────────────

    def generate_and_send_report(self):
        session = get_session()
        try:
            sesion_db = session.query(Sesion).filter_by(id=self.active_session_id).first()
            if not sesion_db:
                return

            # ── PASO 1: Registrar automáticamente los AUSENTES en la BD ────────
            todos_alumnos = session.query(Alumno).all()
            asistentes_ids = {
                a.alumno_id for a, _ in (
                    session.query(Asistencia, Alumno)
                    .join(Alumno, Asistencia.alumno_id == Alumno.id)
                    .filter(Asistencia.sesion_id == self.active_session_id)
                    .all()
                )
            }
            faltas_registradas = 0
            for alumno in todos_alumnos:
                if alumno.id not in asistentes_ids:
                    falta = Asistencia(
                        sesion_id=self.active_session_id,
                        alumno_id=alumno.id,
                        metodo="AUTO",
                        estado="FALTA"
                    )
                    session.add(falta)
                    faltas_registradas += 1
            if faltas_registradas > 0:
                session.commit()
                self.add_log(f"{faltas_registradas} ausente(s) registrados automáticamente en BD.")

            # ── PASO 2: Re-consultar lista completa (incluye FALTAs automáticas) ─
            asistentes = (
                session.query(Asistencia, Alumno)
                .join(Alumno, Asistencia.alumno_id == Alumno.id)
                .filter(Asistencia.sesion_id == self.active_session_id)
                .order_by(Asistencia.timestamp)
                .all()
            )

            total_alumnos    = len(asistentes)
            presentes        = [a for a, _ in asistentes if a.estado == "PRESENTE"]
            tardes           = [a for a, _ in asistentes if a.estado in ("TARDE", "RETARDO")]
            ausentes         = [a for a, _ in asistentes if a.estado == "FALTA"]

            fecha_str = sesion_db.fecha.strftime("%d/%m/%Y") if sesion_db.fecha else datetime.date.today().strftime("%d/%m/%Y")
            h_apert  = self.hora_apertura.strftime("%H:%M:%S") if self.hora_apertura else "—"
            h_cierr  = datetime.datetime.now().strftime("%H:%M:%S")

            # ── PASO 3: Construir reporte Telegram ───────────────────────────────
            report = (
                f"📝 *REPORTE DE ASISTENCIA — REC*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📖 *Materia:* {self.active_class_name}\n"
                f"📅 *Fecha:* {fecha_str}\n"
                f"⏰ *Apertura:* {h_apert} | *Cierre:* {h_cierr}\n"
                f"📊 *Sesión ID:* #{self.active_session_id}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"📈 *Métricas:*\n"
                f"✅ Presentes: {len(presentes)}\n"
                f"⚠️ Retardos: {len(tardes)}\n"
                f"❌ Ausentes: {len(ausentes)}\n"
                f"👥 Total enrolados: {total_alumnos}\n\n"
                f"📋 *Detalle completo:*\n"
            )

            tabla = "```\n"
            tabla += "N°  | Código | Estudiante        | Hora     | Estado  \n"
            tabla += "----|--------|-------------------|----------|----------\n"
            for idx, (asis, alum) in enumerate(asistentes, start=1):
                nombre_limpio = alum.nombre.strip()
                nombre_fmt = (nombre_limpio[:14] + "...") if len(nombre_limpio) > 17 else nombre_limpio.ljust(17)
                t_str = asis.timestamp.strftime("%H:%M:%S") if (asis.timestamp and asis.estado != "FALTA") else "  —   "
                estado_fmt = asis.estado.ljust(8)
                tabla += f"{str(idx).zfill(2)}  | {str(asis.alumno_id).ljust(6)} | {nombre_fmt} | {t_str} | {estado_fmt}\n"
            tabla += "```\n"
            report += tabla
            report += f"━━━━━━━━━━━━━━━━━━━━━━\n🤖 _Sistema REC_"

            # ── PASO 4: Guardar reporte .txt ─────────────────────────────────────
            report_dir = BASE_DIR / "reportes"
            report_dir.mkdir(exist_ok=True)
            report_path = report_dir / f"reporte_sesion_{self.active_session_id}.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report.replace("*", "").replace("```", ""))
            self.add_log(f"Reporte .txt guardado → reportes/{report_path.name}")

            # ── PASO 5: Exportar a Excel ─────────────────────────────────────────
            if EXCEL_DISPONIBLE:
                self._export_excel(report_dir, asistentes, fecha_str, h_apert, h_cierr)
            else:
                self.add_log("[Excel] openpyxl no instalado — omitiendo exportación.")

            # ── PASO 6: Enviar reporte a Telegram ────────────────────────────────
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                self.add_log("Enviando reporte a Telegram...")
                _report_copy = report
                def _send():
                    try:
                        bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
                        bot.send_message(TELEGRAM_CHAT_ID, _report_copy, parse_mode="Markdown")
                        self.add_log("[Telegram] Reporte enviado con éxito.")
                    except Exception as tx:
                        self.add_log(f"[Telegram ERROR] {tx}")
                threading.Thread(target=_send, daemon=True).start()
            else:
                self.add_log("[Telegram] Omitido — token/chat_id no configurados.")

        except Exception as e:
            self.add_log(f"Error al generar reporte: {e}")
        finally:
            session.close()
    # ── Ventana de enrolamiento ───────────────────────────────────────────────

    def open_enrollment_window(self):
        if self.session_active:
            from tkinter import messagebox
            messagebox.showwarning(
                "Clase en curso",
                "Finalice la sesión activa antes de enrolar nuevos alumnos."
            )
            return

        self.enroll_win = ctk.CTkToplevel(self)
        self.enroll_win.title("Enrolamiento Biométrico")
        self.enroll_win.geometry("860x500")
        self.enroll_win.resizable(False, False)
        self.enroll_win.configure(fg_color=C["bg"])
        self.enroll_win.transient(self)
        self.enroll_win.after(100, lambda: self.enroll_win.grab_set()
                               if (self.enroll_win and self.enroll_win.winfo_exists()) else None)
        self.enroll_win.protocol("WM_DELETE_WINDOW", self.close_enrollment_window)

        self.enroll_win.grid_columnconfigure(0, weight=1)
        self.enroll_win.grid_columnconfigure(1, weight=1)
        self.enroll_win.grid_rowconfigure(0, weight=1)

        # Cámara
        cam_f = ctk.CTkFrame(self.enroll_win, fg_color=C["surface"], corner_radius=16,
                              border_width=1, border_color=C["border"])
        cam_f.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="nsew")

        ctk.CTkLabel(cam_f, text="Vista de enrolamiento",
                     font=ctk.CTkFont(family="Outfit", size=12, weight="bold"),
                     text_color=C["muted"]).pack(anchor="w", padx=16, pady=(14, 6))

        self.enroll_cam_label = ctk.CTkLabel(cam_f, text="Inicializando cámara...",
                                              font=ctk.CTkFont(family="Outfit", size=13),
                                              text_color=C["muted"])
        self.enroll_cam_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Formulario
        form_f = ctk.CTkFrame(self.enroll_win, fg_color=C["surface"], corner_radius=16,
                               border_width=1, border_color=C["border"])
        form_f.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="nsew")

        ctk.CTkLabel(form_f, text="Registro biométrico",
                     font=ctk.CTkFont(family="Outfit", size=16, weight="bold"),
                     text_color=C["indigo_text"]).pack(anchor="w", padx=20, pady=(20, 2))
        ctk.CTkLabel(form_f, text="Rostro despejado, sin lentes ni gorra.",
                     font=ctk.CTkFont(family="Outfit", size=11),
                     text_color=C["muted"]).pack(anchor="w", padx=20, pady=(0, 14))

        for label, attr, ph in [
            ("Código / Matrícula", "entry_enroll_id", "Ej: 20230045"),
            ("Nombre completo",    "entry_enroll_name", "Ej: Juan Pérez"),
        ]:
            ctk.CTkLabel(form_f, text=label, font=ctk.CTkFont(family="Outfit", size=11),
                         text_color=C["muted"]).pack(anchor="w", padx=20, pady=(4, 2))
            e = ctk.CTkEntry(form_f, placeholder_text=ph, width=340, height=34,
                             fg_color=C["surface2"], border_color=C["border2"],
                             text_color=C["text"], font=ctk.CTkFont(family="Outfit", size=12),
                             corner_radius=8)
            e.pack(padx=20, pady=(0, 6))
            setattr(self, attr, e)

        ctk.CTkLabel(form_f, text="Semilla TOTP (autogenerada)",
                     font=ctk.CTkFont(family="Outfit", size=11),
                     text_color=C["muted"]).pack(anchor="w", padx=20, pady=(4, 2))
        self.temp_totp_secret = pyotp.random_base32()
        self.entry_enroll_seed = ctk.CTkEntry(
            form_f, width=340, height=34,
            fg_color=C["surface2"], border_color=C["border2"],
            text_color="#4b5563", font=ctk.CTkFont(family="Consolas", size=11),
            corner_radius=8
        )
        self.entry_enroll_seed.insert(0, self.temp_totp_secret)
        self.entry_enroll_seed.configure(state="readonly")
        self.entry_enroll_seed.pack(padx=20, pady=(0, 12))

        self.enroll_status_label = ctk.CTkLabel(
            form_f,
            text="Alinee su rostro frente al scanner...",
            font=ctk.CTkFont(family="Outfit", size=12, weight="bold"),
            text_color=C["yellow"]
        )
        self.enroll_status_label.pack(fill="x", padx=20, pady=4)

        ctk.CTkButton(
            form_f, text="Capturar firma y registrar",
            fg_color=C["green"], hover_color="#16a34a",
            text_color="#0a1a0a",
            font=ctk.CTkFont(family="Outfit", size=13, weight="bold"),
            height=38, corner_radius=10,
            command=self.capture_and_save_student
        ).pack(fill="x", padx=20, pady=(12, 4))

        ctk.CTkButton(
            form_f, text="Cancelar",
            fg_color=C["surface2"], hover_color=C["red_bg"],
            text_color=C["red"],
            font=ctk.CTkFont(family="Outfit", size=12),
            height=34, corner_radius=10,
            border_width=1, border_color=C["red"],
            command=self.close_enrollment_window
        ).pack(fill="x", padx=20, pady=(0, 12))

        self.enroll_cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.enroll_cap.isOpened():
            self.enroll_status_label.configure(text="ERROR: No se pudo abrir la cámara.", text_color=C["red"])
            return

        self.enroll_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.enroll_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.current_enroll_frame = None
        self.current_enroll_face_data = None
        self.enroll_active = True
        self.update_enroll_frame()

    def update_enroll_frame(self):
        if not getattr(self, "enroll_active", False):
            return
        ret, frame = self.enroll_cap.read()
        if not ret:
            self.enroll_cam_label.configure(text="Error de transmisión")
            self.enroll_win.after(30, self.update_enroll_frame)
            return

        frame = cv2.flip(frame, 1)
        self.current_enroll_frame = frame.copy()
        h, w, _ = frame.shape
        self.face_engine.detector.setInputSize((w, h))
        _, faces = self.face_engine.detector.detect(frame)

        display = frame.copy()
        if faces is not None and len(faces) > 0:
            self.current_enroll_face_data = faces[0]
            box = list(map(int, faces[0][0:4]))
            x, y, wb, hb = box
            cv2.rectangle(display, (x, y), (x + wb, y + hb), (52, 211, 153), 2)
            lm = list(map(int, faces[0][4:14]))
            for i in range(5):
                cv2.circle(display, (lm[i*2], lm[i*2+1]), 3, (129, 140, 248), -1)
            self.enroll_status_label.configure(text="● Rostro detectado — listo para registrar", text_color=C["green"])
        else:
            self.current_enroll_face_data = None
            self.enroll_status_label.configure(text="▲ Alinee su rostro (sin gorras ni lentes)", text_color=C["yellow"])

        display = cv2.resize(display, (380, 280))
        img_pil = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        img_tk  = ImageTk.PhotoImage(image=img_pil)
        self.enroll_cam_label_image = img_tk
        try:
            self.enroll_cam_label._label.configure(image=img_tk, text="")
        except Exception:
            pass
        self.enroll_win.after(20, self.update_enroll_frame)

    def capture_and_save_student(self):
        alumno_id   = self.entry_enroll_id.get().strip()
        nombre      = self.entry_enroll_name.get().strip()
        totp_secret = self.temp_totp_secret

        if not alumno_id or not nombre:
            self.enroll_status_label.configure(text="Error: ID y Nombre son obligatorios.", text_color=C["red"])
            return
        if self.current_enroll_frame is None or self.current_enroll_face_data is None:
            self.enroll_status_label.configure(text="Error: No se detecta rostro en pantalla.", text_color=C["red"])
            return

        try:
            emb = self.face_engine.extract_embedding(self.current_enroll_frame, self.current_enroll_face_data)
            m_id, m_name, _ = self.face_engine.match_face(emb)
            if m_id:
                self.enroll_status_label.configure(
                    text=f"Error: Este rostro pertenece a '{m_name}' ({m_id}).",
                    text_color=C["red"]
                )
                return

            session = get_session()
            try:
                if session.query(Alumno).filter_by(id=alumno_id).first():
                    self.enroll_status_label.configure(
                        text=f"Error: ID {alumno_id} ya está registrado.",
                        text_color=C["red"]
                    )
                    return

                a = Alumno(id=alumno_id, nombre=nombre)
                a.set_embedding(emb)
                a.totp_secret = totp_secret
                session.add(a)
                session.commit()
                self.add_log(f"Alumno '{nombre}' ({alumno_id}) registrado.")
                self.face_engine.reload_students_cache()

                from tkinter import messagebox
                messagebox.showinfo(
                    "Registro exitoso",
                    f"Estudiante registrado.\n\nNombre: {nombre}\nID: {alumno_id}\nTOTP: {totp_secret}"
                )
                self.close_enrollment_window()

            except Exception as e:
                session.rollback()
                self.enroll_status_label.configure(text=f"Error BD: {e}", text_color=C["red"])
            finally:
                session.close()

        except Exception as ex:
            self.enroll_status_label.configure(text=f"Error de extracción: {ex}", text_color=C["red"])

    def close_enrollment_window(self):
        self.enroll_active = False
        if getattr(self, "enroll_cap", None):
            self.enroll_cap.release()
            self.enroll_cap = None
        if getattr(self, "enroll_win", None):
            self.enroll_win.grab_release()
            self.enroll_win.destroy()
            self.enroll_win = None

    def on_closing(self):
        try:
            self.add_log("Cerrando aplicación...")
            if self.session_active:
                self.stop_session()
        except Exception:
            pass
        self.destroy()
        sys.exit(0)


if __name__ == "__main__":
    init_db()
    app = RecDashboard()
    app.mainloop()