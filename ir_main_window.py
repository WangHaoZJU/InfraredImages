"""
Main application window for RealSense D435i dual-IR visualization.

Provides a live side-by-side preview of both infrared cameras (IR Left
and IR Right) with OSD overlay, profile selection, and IR emitter control.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ir_camera_worker import DEFAULT_PROFILE, PREDEFINED_IR_PROFILES, IRCameraWorker

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
APP_TITLE = "RealSense D435i IR Viewer"
WIN_MIN_WIDTH = 1100
WIN_MIN_HEIGHT = 520
FPS_EMA_ALPHA = 0.12
OSD_MARGIN = 10
OSD_FONT_SIZE = 11

# Dark-theme palette (GitHub-dark inspired)
PALETTE = {
    "bg": "#0d1117",
    "surface": "#161b22",
    "border": "#30363d",
    "border_hover": "#58a6ff",
    "accent": "#1f6feb",
    "text_primary": "#e6edf3",
    "text_secondary": "#8b949e",
    "green": "#3fb950",
    "red": "#f85149",
    "red_bg": "#da3633",
}

# ── QSS Stylesheet ────────────────────────────────────────────────────────
STYLESHEET = f"""
QMainWindow {{
    background-color: {PALETTE["bg"]};
}}
QWidget {{
    color: {PALETTE["text_primary"]};
    font-family: "Segoe UI", "Noto Sans", "Ubuntu", sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    background-color: {PALETTE["surface"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 8px;
    margin-top: 20px;
    padding: 20px 14px 14px 14px;
    font-weight: 600;
    font-size: 13px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    color: {PALETTE["border_hover"]};
}}
QComboBox {{
    background-color: {PALETTE["bg"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 5px;
    padding: 6px 10px;
    min-height: 18px;
    selection-background-color: {PALETTE["accent"]};
}}
QComboBox:hover {{
    border-color: {PALETTE["border_hover"]};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background-color: {PALETTE["surface"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 4px;
    selection-background-color: {PALETTE["accent"]};
    outline: none;
}}
QCheckBox {{
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {PALETTE["border"]};
    border-radius: 4px;
    background-color: {PALETTE["bg"]};
}}
QCheckBox::indicator:checked {{
    background-color: #238636;
    border-color: #238636;
}}
QLabel#irLabelLeft {{
    background-color: #000;
    border-radius: 8px;
    border: 2px solid {PALETTE["border"]};
}}
QLabel#irLabelRight {{
    background-color: #000;
    border-radius: 8px;
    border: 2px solid {PALETTE["border"]};
}}
QStatusBar {{
    background-color: {PALETTE["surface"]};
    border-top: 1px solid {PALETTE["border"]};
    color: {PALETTE["text_secondary"]};
    font-size: 12px;
}}
QFrame#separator {{
    background-color: {PALETTE["border"]};
    max-height: 1px;
}}
QFrame#irDivider {{
    background-color: {PALETTE["border"]};
    max-width: 2px;
}}
"""


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _profile_label(w: int, h: int, fps: int) -> str:
    return f"{w}×{h} @ {fps} fps"


def _timestamp_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ═══════════════════════════════════════════════════════════════════════════
# IRMainWindow
# ═══════════════════════════════════════════════════════════════════════════

class IRMainWindow(QMainWindow):
    """Primary application window for dual-IR visualization."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(WIN_MIN_WIDTH, WIN_MIN_HEIGHT)
        self.setStyleSheet(STYLESHEET)

        # -- State ----------------------------------------------------------
        self._current_profile: tuple[int, int, int] = DEFAULT_PROFILE
        self._ir_width: int = DEFAULT_PROFILE[0]
        self._ir_height: int = DEFAULT_PROFILE[1]
        self._smooth_fps: float = 0.0

        # -- Camera worker --------------------------------------------------
        self._worker = IRCameraWorker()
        self._worker.frames_ready.connect(self._on_frames_ready)
        self._worker.fps_measured.connect(self._on_fps_measured)
        self._worker.profiles_ready.connect(self._on_profiles_ready)
        self._worker.camera_connected.connect(self._on_camera_connected)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.profile_changed.connect(self._on_profile_changed)

        # -- Build UI -------------------------------------------------------
        self._setup_ui()
        self._setup_status_bar()

        # -- Start camera ---------------------------------------------------
        self._worker.start()

    # ── UI construction ───────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.addWidget(self._create_ir_panel("Left"))
        splitter.addWidget(self._create_ir_panel("Right"))
        splitter.addWidget(self._create_control_panel())
        splitter.setStretchFactor(0, 4)  # IR Left 40 %
        splitter.setStretchFactor(1, 4)  # IR Right 40 %
        splitter.setStretchFactor(2, 2)  # Controls 20 %
        splitter.setSizes([420, 420, 220])
        root.addWidget(splitter)

    def _create_ir_panel(self, side: str) -> QWidget:
        """Create a panel with a QLabel for displaying one IR stream."""
        label_name = f"irLabel{side}"
        object_name = f"irLabel{side}"

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 4, 0)

        label = QLabel(f"IR {side}\nNo Signal")
        label.setObjectName(object_name)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(320, 240)
        label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        # Placeholder while camera is not connected
        placeholder = QPixmap(424, 240)
        placeholder.fill(QColor("#0a0a0a"))
        label.setPixmap(placeholder)

        layout.addWidget(label)

        # Store references
        if side == "Left":
            self._ir_left_label = label
        else:
            self._ir_right_label = label

        return container

    def _create_control_panel(self) -> QWidget:
        """Control panel with profile selector and IR emitter toggle."""
        wrapper = QWidget()
        wrapper.setFixedWidth(220)
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(8, 4, 0, 4)
        layout.setSpacing(10)

        layout.addWidget(self._create_stream_card())
        layout.addWidget(self._create_emitter_card())
        layout.addStretch()

        return wrapper

    # ── Cards ─────────────────────────────────────────────────────────────

    def _create_stream_card(self) -> QGroupBox:
        gb = QGroupBox("\U0001f4f7  IR Profile")
        layout = QVBoxLayout(gb)
        layout.setSpacing(8)

        # Profile selector
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Profile:"))
        self._profile_combo = QComboBox()
        self._profile_combo.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        # Pre-populate with predefined profiles
        for w, h, fps in PREDEFINED_IR_PROFILES:
            self._profile_combo.addItem(
                _profile_label(w, h, fps), (w, h, fps)
            )
        idx = self._profile_combo.findData(DEFAULT_PROFILE)
        if idx >= 0:
            self._profile_combo.setCurrentIndex(idx)
        self._profile_combo.currentIndexChanged.connect(
            self._on_profile_selected
        )
        row1.addWidget(self._profile_combo)
        layout.addLayout(row1)

        # Divider
        sep = QFrame()
        sep.setObjectName("separator")
        layout.addWidget(sep)

        # Profile info
        self._profile_info = QLabel()
        self._profile_info.setStyleSheet(
            f"color: {PALETTE['text_secondary']}; font-size: 12px;"
        )
        w, h, fps = DEFAULT_PROFILE
        self._profile_info.setText(
            f"Stream: {w}×{h}\nFormat: Y8 grayscale\nTarget: {fps} fps"
        )
        layout.addWidget(self._profile_info)

        return gb

    def _create_emitter_card(self) -> QGroupBox:
        gb = QGroupBox("\U0001f4a1  IR Emitter")
        layout = QVBoxLayout(gb)
        layout.setSpacing(8)

        self._emitter_cb = QCheckBox("IR Projector")
        self._emitter_cb.setChecked(True)
        self._emitter_cb.setToolTip(
            "Toggle the IR pattern projector.\n"
            "When ON, a dot pattern is projected to aid stereo depth.\n"
            "The pattern is visible in the IR camera views."
        )
        self._emitter_cb.toggled.connect(self._on_emitter_toggled)
        layout.addWidget(self._emitter_cb)

        info = QLabel(
            "The IR projector emits a dot pattern\n"
            "visible in both IR camera views.\n"
            "Turn OFF for ambient-only IR."
        )
        info.setStyleSheet(
            f"color: {PALETTE['text_secondary']}; font-size: 11px;"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        return gb

    # ── Status bar ────────────────────────────────────────────────────────

    def _setup_status_bar(self) -> None:
        self._status = QStatusBar()
        self.setStatusBar(self._status)

        self._status_camera = QLabel("● Disconnected")
        self._status.addWidget(self._status_camera)

        self._status_fps = QLabel("FPS: --")
        self._status_fps.setStyleSheet(
            f"color: {PALETTE['text_secondary']}; font-size: 12px;"
        )
        self._status.addPermanentWidget(self._status_fps)

        self._status_res = QLabel("")
        self._status_res.setStyleSheet(
            f"color: {PALETTE['text_secondary']}; font-size: 12px;"
        )
        self._status.addPermanentWidget(self._status_res)

    # ── Slots — camera worker signals ─────────────────────────────────────

    @Slot(np.ndarray, np.ndarray, float)
    def _on_frames_ready(
        self, ir_left: np.ndarray, ir_right: np.ndarray, _ts: float
    ) -> None:
        """Display both IR frames side-by-side with OSD overlay.

        IR frames arrive as single-channel Y8 (h, w) uint8.  We expand
        each to a 3-channel RGB array so that coloured OSD text renders
        correctly (Format_Grayscale8 would desaturate it).
        """
        hl, wl = ir_left.shape[:2]
        hr, wr = ir_right.shape[:2]
        self._ir_width = wl
        self._ir_height = hl

        # Expand single-channel → 3-channel RGB for display
        qimg_left = self._ir_to_qimage(ir_left)
        qimg_right = self._ir_to_qimage(ir_right)

        # One-shot diagnostic
        if not hasattr(self, "_diag_printed"):
            self._diag_printed = True
            import sys
            print(
                f"[ui] IR Left  {wl}×{hl}  "
                f"qimg.isNull={qimg_left.isNull()}  "
                f"pixel(0,0)={qimg_left.pixelColor(0, 0).value()}",
                file=sys.stderr,
            )
            print(
                f"[ui] IR Right {wr}×{hr}  "
                f"qimg.isNull={qimg_right.isNull()}  "
                f"pixel(0,0)={qimg_right.pixelColor(0, 0).value()}",
                file=sys.stderr,
            )

        # Draw OSD on each
        self._draw_osd(qimg_left, "IR Left")
        self._draw_osd(qimg_right, "IR Right")

        # Display left
        pix_left = QPixmap.fromImage(qimg_left)
        if not self._ir_left_label.size().isEmpty():
            pix_left = pix_left.scaled(
                self._ir_left_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        self._ir_left_label.setPixmap(pix_left)

        # Display right
        pix_right = QPixmap.fromImage(qimg_right)
        if not self._ir_right_label.size().isEmpty():
            pix_right = pix_right.scaled(
                self._ir_right_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        self._ir_right_label.setPixmap(pix_right)

    @Slot(float)
    def _on_fps_measured(self, fps: float) -> None:
        """Smooth FPS with EMA."""
        if self._smooth_fps == 0.0:
            self._smooth_fps = fps
        else:
            self._smooth_fps = (
                FPS_EMA_ALPHA * fps
                + (1.0 - FPS_EMA_ALPHA) * self._smooth_fps
            )
        self._status_fps.setText(f"FPS: {self._smooth_fps:.1f}")

    @Slot(list)
    def _on_profiles_ready(
        self, profiles: list[tuple[int, int, int]]
    ) -> None:
        """Populate profile dropdown with camera-reported profiles."""
        self._profile_combo.blockSignals(True)
        self._profile_combo.clear()
        for w, h, fps in profiles:
            self._profile_combo.addItem(
                _profile_label(w, h, fps), (w, h, fps)
            )
        idx = self._profile_combo.findData(self._current_profile)
        if idx >= 0:
            self._profile_combo.setCurrentIndex(idx)
        self._profile_combo.blockSignals(False)

    @Slot(bool)
    def _on_camera_connected(self, ok: bool) -> None:
        if ok:
            self._status_camera.setText(
                f"● Connected — {self._worker.device_name}"
            )
            self._status_camera.setStyleSheet(
                f"color: {PALETTE['green']}; font-weight: 600;"
            )
            # Apply initial emitter state
            QTimer.singleShot(500, self._sync_emitter_state)
        else:
            self._status_camera.setText("● Disconnected")
            self._status_camera.setStyleSheet(
                f"color: {PALETTE['red']}; font-weight: 600;"
            )

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._status.showMessage(f"⚠ {msg}", 8000)

    @Slot(int, int, int)
    def _on_profile_changed(self, w: int, h: int, fps: int) -> None:
        self._current_profile = (w, h, fps)
        self._status_res.setText(f"{w}×{h} @ {fps} fps")
        self._profile_info.setText(
            f"Stream: {w}×{h}\nFormat: Y8 grayscale\nTarget: {fps} fps"
        )

    # ── Slots — UI controls ───────────────────────────────────────────────

    def _on_profile_selected(self, _idx: int) -> None:
        data = self._profile_combo.currentData()
        if data is None or data == self._current_profile:
            return
        w, h, fps = data
        self._current_profile = (w, h, fps)
        self._worker.request_change_profile(w, h, fps)
        self._status_res.setText(f"{w}×{h} @ {fps} fps")
        self._profile_info.setText(
            f"Stream: {w}×{h}\nFormat: Y8 grayscale\nTarget: {fps} fps"
        )

    def _on_emitter_toggled(self, checked: bool) -> None:
        self._worker.set_emitter_enabled(checked)

    # ── OSD rendering ─────────────────────────────────────────────────────

    def _draw_osd(self, qimg: QImage, label: str) -> None:
        """Paint IR stream label, resolution badge, and timestamp on *qimg*."""
        painter = QPainter(qimg)
        painter.setRenderHint(QPainter.Antialiasing)

        font = QFont("monospace", OSD_FONT_SIZE)
        font.setStyleHint(QFont.Monospace)
        painter.setFont(font)

        w = qimg.width()
        h = qimg.height()

        def badge(
            x: int,
            y: int,
            text: str,
            fg: QColor = QColor("#00ff88"),
            bg: QColor = QColor(0, 0, 0, 160),
        ) -> None:
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(text)
            th = fm.height()
            pad_x, pad_y = 9, 4
            painter.setPen(Qt.NoPen)
            painter.setBrush(bg)
            painter.drawRoundedRect(
                x - pad_x, y - th + 1,
                tw + pad_x * 2, th + pad_y * 2,
                5, 5,
            )
            painter.setPen(fg)
            painter.drawText(x, y, text)

        # Top-left: stream label
        badge(OSD_MARGIN, OSD_MARGIN + OSD_FONT_SIZE + 4,
              label,
              fg=QColor(PALETTE["green"]))

        # Top-right: resolution
        res_text = f"{w}×{h}"
        fm = painter.fontMetrics()
        rw = fm.horizontalAdvance(res_text)
        badge(w - rw - OSD_MARGIN, OSD_MARGIN + OSD_FONT_SIZE + 4,
              res_text,
              fg=QColor("#c9d1d9"))

        # Bottom-left: timestamp
        ts = _timestamp_str()
        badge(OSD_MARGIN, h - OSD_MARGIN - 3,
              ts,
              fg=QColor(PALETTE["text_secondary"]))

        # Bottom-right: FPS (if available)
        if self._smooth_fps > 0:
            fps_text = f"FPS {self._smooth_fps:.1f}"
            fw = fm.horizontalAdvance(fps_text)
            badge(w - fw - OSD_MARGIN, h - OSD_MARGIN - 3,
                  fps_text,
                  fg=QColor(PALETTE["green"]))

        painter.end()

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _ir_to_qimage(ir_data: np.ndarray) -> QImage:
        """Convert single-channel Y8 IR frame to an RGB QImage for display.

        Stacks the grayscale data into 3 identical channels so that
        coloured OSD overlays (green/red text) render with correct colour
        rather than being desaturated to grayscale.
        """
        if not ir_data.flags["C_CONTIGUOUS"]:
            ir_data = np.ascontiguousarray(ir_data)
        h, w = ir_data.shape
        rgb = np.stack([ir_data, ir_data, ir_data], axis=-1)  # (h, w, 3)
        rgb = np.ascontiguousarray(rgb)
        return QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)

    def _sync_emitter_state(self) -> None:
        """Apply the current emitter checkbox state to the camera."""
        self._worker.set_emitter_enabled(self._emitter_cb.isChecked())

    # ── Window lifecycle ──────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._worker.stop_camera()
        self._worker.wait(3000)
        event.accept()
