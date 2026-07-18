"""
RealSense IR camera worker thread.

Manages the RealSense pipeline in a background thread for dual infrared
streams (IR Left = stream 1, IR Right = stream 2) from a D435i camera.
Communicates with the UI via Qt signals.
"""

import time
import numpy as np
import pyrealsense2 as rs
from PySide6.QtCore import QThread, Signal, QMutex


# ---------------------------------------------------------------------------
# Predefined IR stream profiles (fallback when camera is not connected)
# ---------------------------------------------------------------------------
PREDEFINED_IR_PROFILES: list[tuple[int, int, int]] = [
    (1280, 800, 30),
    (848, 480, 90),
    (848, 480, 60),
    (848, 480, 30),
    (640, 480, 60),
    (640, 480, 30),
    (424, 240, 60),
    (424, 240, 30),
]

DEFAULT_PROFILE = (848, 480, 30)


class IRCameraWorker(QThread):
    """Background thread that drives the RealSense dual-IR streams."""

    # -- Signals (emitted → main thread) ------------------------------------
    frames_ready = Signal(np.ndarray, np.ndarray, float)
    # (ir_left, ir_right, timestamp_sec)
    fps_measured = Signal(float)              # measured fps
    profiles_ready = Signal(list)              # list[(w, h, fps), …]
    camera_connected = Signal(bool)            # True when pipeline starts
    error_occurred = Signal(str)               # human-readable error
    profile_changed = Signal(int, int, int)    # (w, h, fps) after switch

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pipeline: rs.pipeline | None = None
        self._config: rs.config | None = None
        self._stereo_sensor: rs.sensor | None = None
        self._device: rs.device | None = None

        # Target profile (set by UI)
        self._width = DEFAULT_PROFILE[0]
        self._height = DEFAULT_PROFILE[1]
        self._fps = DEFAULT_PROFILE[2]

        # Thread control
        self._running = False
        self._mutex = QMutex()
        self._pending_profile: tuple[int, int, int] | None = None

        # IR emitter state
        self._emitter_enabled: bool | None = None

    # -- Public API (callable from main thread) -----------------------------

    def request_change_profile(self, width: int, height: int, fps: int) -> None:
        """Request a resolution / fps change. Applied on the worker thread."""
        self._mutex.lock()
        self._pending_profile = (width, height, fps)
        self._mutex.unlock()

    def set_emitter_enabled(self, enabled: bool) -> None:
        """Toggle the IR pattern projector on/off."""
        self._emitter_enabled = enabled
        if self._stereo_sensor is not None:
            try:
                self._stereo_sensor.set_option(
                    rs.option.emitter_enabled, 1.0 if enabled else 0.0
                )
            except Exception as exc:
                self.error_occurred.emit(f"Failed to set emitter: {exc}")

    @property
    def device_name(self) -> str:
        """Human-readable device name, or 'Unknown' if not connected."""
        if self._device is not None:
            try:
                return self._device.get_info(rs.camera_info.name)
            except Exception:
                pass
        return "Unknown"

    def stop_camera(self) -> None:
        """Signal the worker loop to exit and stop the pipeline."""
        self._running = False
        # Wake up wait_for_frames by stopping the pipeline
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass

    # -- QThread run --------------------------------------------------------

    def run(self) -> None:
        self._running = True
        ctx = rs.context()

        # Attempt to find a device
        try:
            devices = ctx.query_devices()
            if len(devices) == 0:
                self.error_occurred.emit(
                    "No RealSense camera found. "
                    "Please connect a D435i / D455i."
                )
                self.camera_connected.emit(False)
                return
            self._device = devices[0]
        except Exception as exc:
            self.error_occurred.emit(f"Device query failed: {exc}")
            self.camera_connected.emit(False)
            return

        # Start pipeline with dual IR streams
        try:
            self._pipeline = rs.pipeline(ctx)
            self._config = rs.config()
            self._config.enable_stream(
                rs.stream.infrared, 1,
                self._width, self._height, rs.format.y8, self._fps,
            )
            self._config.enable_stream(
                rs.stream.infrared, 2,
                self._width, self._height, rs.format.y8, self._fps,
            )
            profile = self._pipeline.start(self._config)
            self._stereo_sensor = self._find_stereo_sensor(profile)
            self.camera_connected.emit(True)

            # Apply pending emitter state
            if self._emitter_enabled is not None and self._stereo_sensor is not None:
                try:
                    self._stereo_sensor.set_option(
                        rs.option.emitter_enabled,
                        1.0 if self._emitter_enabled else 0.0,
                    )
                except Exception:
                    pass

            # Publish supported IR profiles
            supported = self._enumerate_ir_profiles(profile)
            self.profiles_ready.emit(
                supported if supported else PREDEFINED_IR_PROFILES
            )

        except Exception as exc:
            self.error_occurred.emit(f"Pipeline start failed: {exc}")
            self.camera_connected.emit(False)
            return

        # -- Frame loop -----------------------------------------------------
        frame_count = 0
        last_fps_time = time.perf_counter()
        _diag_done = False  # one-shot diagnostic
        consecutive_timeouts = 0

        while self._running:
            # Process pending profile change
            self._process_pending_profile()

            # Fetch a frame set
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                consecutive_timeouts = 0
            except RuntimeError as exc:
                if not self._running:
                    break
                if "Frame didn't arrive" in str(exc):
                    consecutive_timeouts += 1
                    if consecutive_timeouts == 5:
                        self.error_occurred.emit(
                            "No IR frames received for 5 seconds. "
                            "Check the USB 3 cable/port."
                        )
                    continue
                self.error_occurred.emit(f"Camera pipeline error: {exc}")
                break

            # Extract both IR frames
            ir_left_frame = frames.get_infrared_frame(1)
            ir_right_frame = frames.get_infrared_frame(2)
            if not ir_left_frame or not ir_right_frame:
                continue

            # Convert to contiguous numpy arrays (own the buffer)
            ir_left = np.asanyarray(ir_left_frame.get_data()).copy()
            ir_right = np.asanyarray(ir_right_frame.get_data()).copy()
            timestamp = ir_left_frame.get_timestamp() / 1000.0  # ms → s

            # One-shot diagnostic
            if not _diag_done:
                _diag_done = True
                import sys
                print(
                    f"[ir-camera] IR Left  shape={ir_left.shape} "
                    f"dtype={ir_left.dtype} "
                    f"min={ir_left.min():.0f} max={ir_left.max():.0f} "
                    f"mean={ir_left.mean():.1f}",
                    file=sys.stderr,
                )
                print(
                    f"[ir-camera] IR Right shape={ir_right.shape} "
                    f"dtype={ir_right.dtype} "
                    f"min={ir_right.min():.0f} max={ir_right.max():.0f} "
                    f"mean={ir_right.mean():.1f}",
                    file=sys.stderr,
                )

            self.frames_ready.emit(ir_left, ir_right, timestamp)

            # FPS measurement (every ~1 s)
            frame_count += 1
            now = time.perf_counter()
            elapsed = now - last_fps_time
            if elapsed >= 1.0:
                self.fps_measured.emit(frame_count / elapsed)
                frame_count = 0
                last_fps_time = now

        # Cleanup
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._stereo_sensor = None
        self._pipeline = None

    # -- Internals ----------------------------------------------------------

    def _process_pending_profile(self) -> None:
        self._mutex.lock()
        pending = self._pending_profile
        self._pending_profile = None
        self._mutex.unlock()

        if pending is None:
            return

        w, h, fps = pending
        if (w, h, fps) == (self._width, self._height, self._fps):
            return  # no-op

        self._width, self._height, self._fps = w, h, fps

        # Restart pipeline with new config
        try:
            self._pipeline.stop()
            self._config = rs.config()
            self._config.enable_stream(
                rs.stream.infrared, 1, w, h, rs.format.y8, fps
            )
            self._config.enable_stream(
                rs.stream.infrared, 2, w, h, rs.format.y8, fps
            )
            profile = self._pipeline.start(self._config)
            self._stereo_sensor = self._find_stereo_sensor(profile)
            self.profile_changed.emit(w, h, fps)

            # Re-apply emitter state after pipeline restart
            if self._emitter_enabled is not None and self._stereo_sensor is not None:
                try:
                    self._stereo_sensor.set_option(
                        rs.option.emitter_enabled,
                        1.0 if self._emitter_enabled else 0.0,
                    )
                except Exception:
                    pass

        except Exception as exc:
            self.error_occurred.emit(f"Profile change failed: {exc}")

    @staticmethod
    def _find_stereo_sensor(profile: rs.pipeline_profile) -> rs.sensor | None:
        """Find the stereo module sensor (owns IR streams on D400 series)."""
        device = profile.get_device()
        for s in device.query_sensors():
            try:
                name = s.get_info(rs.camera_info.name)
                if "Stereo Module" in name or "Stereo" in name:
                    return s
            except Exception:
                pass
        # Fallback: stereo module is typically sensor 0 on D400 series
        sensors = device.query_sensors()
        return sensors[0] if len(sensors) > 0 else None

    @staticmethod
    def _enumerate_ir_profiles(
        profile: rs.pipeline_profile,
    ) -> list[tuple[int, int, int]]:
        """Query all Y8 infrared stream profiles from the device."""
        result: set[tuple[int, int, int]] = set()
        device = profile.get_device()
        for sensor in device.query_sensors():
            for sp in sensor.get_stream_profiles():
                if (
                    sp.stream_type() == rs.stream.infrared
                    and sp.format() == rs.format.y8
                ):
                    vi = sp.as_video_stream_profile()
                    result.add((vi.width(), vi.height(), vi.fps()))
        return sorted(result, key=lambda x: (-x[2], -x[0], -x[1]))
