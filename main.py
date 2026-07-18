#!/usr/bin/env python3
"""
RealSense D435i IR Viewer — graphical tool for visualizing the two infrared
cameras (IR Left / IR Right) of the Intel RealSense D435i depth camera.

Usage:
    python main.py
"""

import sys
from PySide6.QtWidgets import QApplication
from ir_main_window import IRMainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("RealSense D435i IR Viewer")

    app.setStyle("Fusion")

    window = IRMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
