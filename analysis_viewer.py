
import sys
import argparse
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QAction,
    QMenu,
    QMenuBar,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


PALETTE = {
    "background": "#0f172a",
    "surface": "#1e293b",
    "card": "#16213c",
    "text": "#f8fafc",
    "muted": "#cbd5f5",
    "accent": "#38bdf8",
    "accent2": "#34d399",
    "warning": "#f97316",
}


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(PALETTE["background"]))
    palette.setColor(QPalette.WindowText, QColor(PALETTE["text"]))
    palette.setColor(QPalette.Base, QColor(PALETTE["card"]))
    palette.setColor(QPalette.AlternateBase, QColor(PALETTE["surface"]))
    palette.setColor(QPalette.ToolTipBase, QColor(PALETTE["surface"]))
    palette.setColor(QPalette.ToolTipText, QColor(PALETTE["text"]))
    palette.setColor(QPalette.Text, QColor(PALETTE["text"]))
    palette.setColor(QPalette.Button, QColor(PALETTE["surface"]))
    palette.setColor(QPalette.ButtonText, QColor(PALETTE["text"]))
    palette.setColor(QPalette.Highlight, QColor(PALETTE["accent"]))
    palette.setColor(QPalette.HighlightedText, QColor(PALETTE["background"]))
    app.setPalette(palette)
    app.setStyleSheet(
        """
        QWidget { background-color: #0f172a; color: #f8fafc; }
        QTableWidget { background-color: #16213c; gridline-color: #334155; }
        QHeaderView::section { background-color: #1e293b; color: #f8fafc; padding: 4px; }
        QTabBar::tab { background: #1e293b; padding: 6px 12px; }
        QTabBar::tab:selected { background: #38bdf8; color: #0f172a; }
        QComboBox { background-color: #16213c; color: #f8fafc; selection-background-color: #38bdf8; selection-color: #0f172a; border: 1px solid #334155; padding: 4px; }
        QLabel { color: #cbd5f5; }
        """
    )


class HeatmapCanvas(FigureCanvasQTAgg):
    def __init__(self, title: str):
        self.fig = Figure(figsize=(5, 4), facecolor=PALETTE["card"])
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.title = title
        self.colorbar = None
        self.fig.tight_layout()

    def set_data(self, matrix: Optional[np.ndarray], row_labels: List[str], col_labels: List[str], cmap: str = "viridis") -> None:
        self.ax.clear()
        if matrix is None or matrix.size == 0:
            self.ax.text(0.5, 0.5, "No data", ha="center", va="center", color=PALETTE["muted"])
            self.ax.set_title(self.title, color=PALETTE["text"])
            self.draw()
            return
        mat = np.asarray(matrix, dtype=float)
        im = self.ax.imshow(mat, aspect="auto", cmap=cmap)
        self.ax.set_title(self.title, color=PALETTE["text"])
        self.ax.set_xticks(range(len(col_labels)))
        self.ax.set_xticklabels(col_labels, rotation=90, color=PALETTE["text"])
        self.ax.set_yticks(range(len(row_labels)))
        self.ax.set_yticklabels(row_labels, color=PALETTE["text"])
        self.ax.tick_params(colors=PALETTE["text"])
        if self.colorbar is not None:
            self.colorbar.remove()
        self.colorbar = self.fig.colorbar(im, ax=self.ax)
        self.colorbar.ax.tick_params(colors=PALETTE["text"])
        self.fig.tight_layout()
        self.draw()


class SliceCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(5, 3), facecolor=PALETTE["card"])
        super().__init__(self.fig)
        self.ax_row = self.fig.add_subplot(2, 1, 1)
        self.ax_col = self.fig.add_subplot(2, 1, 2)
        self.fig.tight_layout()

    def set_slices(self,
        row_values: np.ndarray,
        row_labels: List[str],
        row_name: str,
        col_values: np.ndarray,
        col_labels: List[str],
        col_name: str,
        row_errors: Optional[np.ndarray] = None,
        col_errors: Optional[np.ndarray] = None,
    ) -> None:
        self.ax_row.clear()
        self.ax_col.clear()
        row_values = np.asarray(row_values if row_values is not None else [], dtype=float)
        col_values = np.asarray(col_values if col_values is not None else [], dtype=float)
        row_errors = np.asarray(row_errors if row_errors is not None else [], dtype=float)
        col_errors = np.asarray(col_errors if col_errors is not None else [], dtype=float)

        if row_values.size:
            x_indices = np.arange(len(row_values))
            # Line + symbol plot with error bars for correctors (row slice)
            self.ax_row.errorbar(x_indices, row_values,
                               yerr=row_errors if row_errors.size == row_values.size else None,
                               fmt='o-', color=PALETTE["accent"],
                               markersize=6, linewidth=2, capsize=4,
                               ecolor=PALETTE["muted"], elinewidth=1.5)
            self.ax_row.set_xticks(range(len(row_labels)))
            self.ax_row.set_xticklabels(row_labels, rotation=90, color=PALETTE["text"])
            self.ax_row.set_ylabel("Response", color=PALETTE["text"])
            self.ax_row.set_title(f"BPM Response: {row_name} (to all correctors)", color=PALETTE["text"])
            self.ax_row.grid(True, alpha=0.2, color=PALETTE["muted"])
        else:
            self.ax_row.text(0.5, 0.5, "Select a BPM", ha="center", va="center", color=PALETTE["muted"])

        if col_values.size:
            x_indices = np.arange(len(col_values))
            # Line + symbol plot with error bars for BPMs (column slice)
            self.ax_col.errorbar(x_indices, col_values,
                               yerr=col_errors if col_errors.size == col_values.size else None,
                               fmt='s-', color=PALETTE["accent2"],
                               markersize=6, linewidth=2, capsize=4,
                               ecolor=PALETTE["muted"], elinewidth=1.5)
            self.ax_col.set_xticks(range(len(col_labels)))
            self.ax_col.set_xticklabels(col_labels, rotation=90, color=PALETTE["text"])
            self.ax_col.set_ylabel("Response", color=PALETTE["text"])
            self.ax_col.set_title(f"Corrector Response: {col_name} (at all BPMs)", color=PALETTE["text"])
            self.ax_col.grid(True, alpha=0.2, color=PALETTE["muted"])
        else:
            self.ax_col.text(0.5, 0.5, "Select a corrector", ha="center", va="center", color=PALETTE["muted"])

        self.ax_row.tick_params(colors=PALETTE["text"])
        self.ax_col.tick_params(colors=PALETTE["text"])
        self.fig.tight_layout()
        self.draw()


class BetaCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(5, 3), facecolor=PALETTE["card"])
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()

    def set_beta(self, labels: List[str], values: Optional[np.ndarray]) -> None:
        self.ax.clear()
        values = np.asarray(values if values is not None else [], dtype=float)
        if values.size:
            # Sort by position (keep original order, not by value)
            x_indices = np.arange(len(values))
            # Line + symbol plot for beta function
            self.ax.plot(x_indices, values, 'o-', color=PALETTE["warning"],
                        markersize=6, linewidth=2)
            self.ax.set_xticks(range(len(labels)))
            self.ax.set_xticklabels(labels, rotation=90, color=PALETTE["text"])
            self.ax.set_ylabel("β (m)", color=PALETTE["text"])
            self.ax.set_title("Beta Function at BPMs (from Response Matrix)", color=PALETTE["text"])
            self.ax.grid(True, alpha=0.2, color=PALETTE["muted"])
        else:
            self.ax.text(0.5, 0.5, "No BPMs", ha="center", va="center", color=PALETTE["muted"])
        self.ax.tick_params(colors=PALETTE["text"])
        self.fig.tight_layout()
        self.draw()


@dataclass
class PlaneDescriptor:
    name: str
    indices: List[int]



class PlaneTab(QWidget):
    def __init__(self, plane: PlaneDescriptor, matrix: np.ndarray, reading_labels: List[str], setting_labels: List[str], error_matrix: Optional[np.ndarray]):
        super().__init__()
        self.plane = plane
        self.matrix = matrix
        self.reading_labels = reading_labels
        self.setting_labels = setting_labels
        self.error_matrix = error_matrix
        self.similarity = self._compute_similarity(matrix)
        self.beta_values = self._compute_beta(matrix)

        main_layout = QVBoxLayout(self)

        # Controls

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("Select BPM:"))
        combo_style = """
        QComboBox { background-color: #1e293b; color: #f8fafc; border: 1px solid #38bdf8; padding: 4px; }
        QComboBox QAbstractItemView { background-color: #0f172a; color: #f8fafc; selection-background-color: #38bdf8; selection-color: #0f172a; }
        """
        self.bpm_combo = QComboBox()
        self.bpm_combo.setMinimumWidth(240)
        self.bpm_combo.setStyleSheet(combo_style)
        self.bpm_combo.addItems(reading_labels)
        self.bpm_combo.currentIndexChanged.connect(self._update_slice)
        control_row.addWidget(self.bpm_combo)

        control_row.addWidget(QLabel("Select Corrector:"))
        self.corr_combo = QComboBox()
        self.corr_combo.setMinimumWidth(240)
        self.corr_combo.setStyleSheet(combo_style)
        self.corr_combo.addItems(setting_labels)
        self.corr_combo.currentIndexChanged.connect(self._update_slice)
        control_row.addWidget(self.corr_combo)

        control_row.addStretch(1)
        main_layout.addLayout(control_row)

        # Response / Similarity / Error tabs
        self.secondary_tabs = QTabWidget()

        # Response tab
        response_widget = QWidget()
        response_layout = QVBoxLayout(response_widget)
        response_splitter = QSplitter(Qt.Horizontal)
        self.response_canvas = HeatmapCanvas(f"{plane.name} Response Matrix")
        response_splitter.addWidget(self.response_canvas)
        self.response_table = QTableWidget()
        self.response_table.setEditTriggers(QTableWidget.NoEditTriggers)
        response_splitter.addWidget(self.response_table)
        response_splitter.setSizes([600, 400])
        response_layout.addWidget(response_splitter)
        self.secondary_tabs.addTab(response_widget, "Response Matrix")
        self._tab_index_response = self.secondary_tabs.indexOf(response_widget)

        # Similarity tab
        similarity_widget = QWidget()
        similarity_layout = QVBoxLayout(similarity_widget)
        similarity_splitter = QSplitter(Qt.Horizontal)
        self.similarity_canvas = HeatmapCanvas(f"{plane.name} Cosine Similarity")
        similarity_splitter.addWidget(self.similarity_canvas)
        self.similarity_table = QTableWidget()
        self.similarity_table.setEditTriggers(QTableWidget.NoEditTriggers)
        similarity_splitter.addWidget(self.similarity_table)
        similarity_splitter.setSizes([600, 400])
        similarity_layout.addWidget(similarity_splitter)
        self.secondary_tabs.addTab(similarity_widget, "Cosine Similarity")
        self._tab_index_similarity = self.secondary_tabs.indexOf(similarity_widget)

        # Error tab
        error_widget = QWidget()
        error_layout = QVBoxLayout(error_widget)
        self.error_canvas = HeatmapCanvas(f"{plane.name} Error Matrix")
        error_layout.addWidget(self.error_canvas)
        self.secondary_tabs.addTab(error_widget, "Error Matrix")
        self._tab_index_error = self.secondary_tabs.indexOf(error_widget)

        main_layout.addWidget(self.secondary_tabs)

        # Slice + beta area
        lower_splitter = QSplitter(Qt.Horizontal)
        self.slice_canvas = SliceCanvas()
        lower_splitter.addWidget(self.slice_canvas)
        beta_container = QWidget()
        beta_layout = QVBoxLayout(beta_container)
        self.beta_canvas = BetaCanvas()
        beta_layout.addWidget(self.beta_canvas)
        self.beta_table = QTableWidget()
        self.beta_table.setColumnCount(2)
        self.beta_table.setHorizontalHeaderLabels(["BPM", "Beta (m)"])
        self.beta_table.setEditTriggers(QTableWidget.NoEditTriggers)
        beta_layout.addWidget(self.beta_table)
        lower_splitter.addWidget(beta_container)
        lower_splitter.setSizes([600, 400])
        main_layout.addWidget(lower_splitter)

        self._populate()

    def _populate(self):
        self.response_canvas.set_data(self.matrix, self.reading_labels, self.setting_labels)
        rows, cols = self.matrix.shape
        self.response_table.setRowCount(rows)
        self.response_table.setColumnCount(cols)
        self.response_table.setVerticalHeaderLabels(self.reading_labels)
        self.response_table.setHorizontalHeaderLabels(self.setting_labels)
        for r in range(rows):
            for c in range(cols):
                val = self.matrix[r, c]
                text = "N/A" if not np.isfinite(val) else f"{val:.4f}"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.response_table.setItem(r, c, item)
        self.response_table.resizeColumnsToContents()

        if self.beta_values is not None:
            self.beta_canvas.set_beta(self.reading_labels, self.beta_values)
            self.beta_table.setRowCount(len(self.reading_labels))
            for idx, label in enumerate(self.reading_labels):
                self.beta_table.setItem(idx, 0, QTableWidgetItem(label))
                value = self.beta_values[idx] if idx < len(self.beta_values) else np.nan
                text = "N/A" if not np.isfinite(value) else f"{value:.6f}"
                self.beta_table.setItem(idx, 1, QTableWidgetItem(text))
            self.beta_table.resizeColumnsToContents()
        else:
            self.beta_canvas.set_beta([], None)
            self.beta_table.setRowCount(0)

        if self.similarity is not None and self.similarity.size:
            row_labels = self.setting_labels[: self.similarity.shape[0]]
            col_labels = self.setting_labels[: self.similarity.shape[1]]
            self.similarity_canvas.set_data(self.similarity, row_labels, col_labels)
            self.similarity_table.setRowCount(self.similarity.shape[0])
            self.similarity_table.setColumnCount(self.similarity.shape[1])
            self.similarity_table.setVerticalHeaderLabels(row_labels)
            self.similarity_table.setHorizontalHeaderLabels(col_labels)
            for r in range(self.similarity.shape[0]):
                for c in range(self.similarity.shape[1]):
                    val = self.similarity[r, c]
                    text = "N/A" if not np.isfinite(val) else f"{val:.4f}"
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignCenter)
                    self.similarity_table.setItem(r, c, item)
            self.similarity_table.resizeColumnsToContents()
            self.secondary_tabs.setTabEnabled(self._tab_index_similarity, True)
        else:
            self.similarity_canvas.set_data(None, [], [])
            self.similarity_table.setRowCount(0)
            self.similarity_table.setColumnCount(0)
            self.secondary_tabs.setTabEnabled(self._tab_index_similarity, False)

        if self.error_matrix is not None and self.error_matrix.size:
            self.error_canvas.set_data(self.error_matrix, self.reading_labels, self.setting_labels, cmap="magma")
            self.secondary_tabs.setTabEnabled(self._tab_index_error, True)
        else:
            self.error_canvas.set_data(None, [], [])
            self.secondary_tabs.setTabEnabled(self._tab_index_error, False)

        self._update_slice()

    def _update_slice(self):
        row_idx = self.bpm_combo.currentIndex()
        col_idx = self.corr_combo.currentIndex()
        if row_idx < 0 or col_idx < 0:
            self.slice_canvas.set_slices([], [], "", [], [], "")
            return
        row_values = self.matrix[row_idx, :]
        col_values = self.matrix[:, col_idx]
        row_name = self.reading_labels[row_idx]
        col_name = self.setting_labels[col_idx]

        # Extract error bars if available
        row_errors = None
        col_errors = None
        if self.error_matrix is not None and self.error_matrix.size:
            row_errors = self.error_matrix[row_idx, :]
            col_errors = self.error_matrix[:, col_idx]

        self.slice_canvas.set_slices(row_values, self.setting_labels, row_name,
                                     col_values, self.reading_labels, col_name,
                                     row_errors, col_errors)

    @staticmethod
    def _compute_similarity(matrix: np.ndarray) -> Optional[np.ndarray]:
        if matrix.size == 0 or matrix.shape[1] == 0:
            return None
        corrector_matrix = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0).T
        if corrector_matrix.shape[0] == 0:
            return None
        norms = np.linalg.norm(corrector_matrix, axis=1)
        if not norms.size:
            return None
        norms = np.where(norms < 1e-9, 1.0, norms)
        normalized = corrector_matrix / norms[:, np.newaxis]
        similarity = np.matmul(normalized, normalized.T)
        return similarity

    @staticmethod
    def _compute_beta(matrix: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute the beta function at BPMs from the response matrix using SVD.

        In accelerator physics, the response matrix element is:
        R_ij ∝ √(β_i · β_j) · cos(Δψ_ij) / (2·sin(πQ))

        Using Singular Value Decomposition (SVD), we can separate the BPM
        and corrector beta functions:
        R = U · Σ · V^T

        The first singular value and vector contain the dominant beta structure.
        For BPMs: β_i ∝ (U[:,0] · σ_0)²

        This is the standard method used in accelerator physics (CERN, Fermilab, etc.)
        """
        if matrix.size == 0:
            return None

        # Remove NaN values by replacing with zeros (SVD requires complete matrix)
        clean_matrix = np.nan_to_num(matrix, nan=0.0)

        # Check if matrix has sufficient rank
        if clean_matrix.shape[0] == 0 or clean_matrix.shape[1] == 0:
            return None

        try:
            # SVD decomposition: R = U * Σ * V^T
            # U contains BPM information, V contains corrector information
            # Σ contains singular values (strengths of each mode)
            U, s, Vt = np.linalg.svd(clean_matrix, full_matrices=False)

            if U.shape[0] == 0 or U.shape[1] == 0 or len(s) == 0:
                return None

            # The first singular vector scaled by first singular value
            # gives √β_i (up to a constant factor)
            # We square it to get β_i
            beta_estimate = (U[:, 0] * s[0]) ** 2

            # Take absolute value to ensure positive beta
            beta_estimate = np.abs(beta_estimate)

            # Normalize to get reasonable units (optional - makes largest beta = max response)
            # This helps with visualization
            max_beta = np.max(beta_estimate)
            if max_beta > 0:
                # Scale to physical units: normalize by maximum
                # The actual scale factor would need lattice information
                beta_estimate = beta_estimate / max_beta * np.max(np.abs(clean_matrix))

            return beta_estimate

        except np.linalg.LinAlgError:
            # Fallback to simple method if SVD fails
            print("[WARNING] SVD failed for beta calculation, using fallback method")
            beta_estimate = np.sqrt(np.nansum(clean_matrix ** 2, axis=1))
            return beta_estimate
class AnalysisWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HLA Advanced Analysis")
        self.resize(1200, 820)
        self.tabs = QTabWidget()
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.addWidget(self.tabs)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self._plane_tabs: List[PlaneTab] = []
        self._build_menu()

    def _build_menu(self) -> None:
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)
        file_menu = QMenu("File", self)
        menubar.addMenu(file_menu)
        open_action = QAction("Open Response CSV...", self)
        open_action.triggered.connect(self._open_csv_dialog)
        file_menu.addAction(open_action)
        export_action = QAction("Export Current Plane...", self)
        export_action.triggered.connect(self._export_current)
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def set_planes(
        self,
        matrix: np.ndarray,
        reading_labels: List[str],
        setting_labels: List[str],
        error_matrix: Optional[np.ndarray],
        plane_map: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        self.tabs.clear()
        self._plane_tabs.clear()
        matrix = np.asarray(matrix, dtype=float)
        error_matrix = np.asarray(error_matrix, dtype=float) if error_matrix is not None else None

        plane_defs: List[PlaneDescriptor] = []
        if plane_map:
            for name, indices in plane_map.items():
                clean_idx = sorted(set(i for i in indices if 0 <= i < matrix.shape[0]))
                if clean_idx:
                    plane_defs.append(PlaneDescriptor(name=name, indices=clean_idx))
        if not plane_defs:
            plane_defs.append(PlaneDescriptor(name="All", indices=list(range(matrix.shape[0]))))
        else:
            used = sorted(set(i for plane in plane_defs for i in plane.indices))
            if len(used) != matrix.shape[0]:
                plane_defs.insert(0, PlaneDescriptor(name="All", indices=list(range(matrix.shape[0]))))

        for plane in plane_defs:
            sub_matrix = matrix[plane.indices, :]
            sub_error = error_matrix[plane.indices, :] if error_matrix is not None else None
            sub_labels = [reading_labels[i] for i in plane.indices]
            tab = PlaneTab(plane, sub_matrix, sub_labels, setting_labels, sub_error)
            self.tabs.addTab(tab, plane.name)
            self._plane_tabs.append(tab)
        self.statusBar().showMessage("Response matrix loaded", 4000)

    def _current_tab(self) -> Optional[PlaneTab]:
        idx = self.tabs.currentIndex()
        if 0 <= idx < len(self._plane_tabs):
            return self._plane_tabs[idx]
        return None

    def _open_csv_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Response CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path, index_col=0)
            self.set_planes(df.to_numpy(dtype=float), df.index.astype(str).tolist(), df.columns.astype(str).tolist(), None, None)
            self.statusBar().showMessage(f"Loaded {path}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {e}")

    def _export_current(self) -> None:
        tab = self._current_tab()
        if not tab:
            QMessageBox.information(self, "Export", "No plane selected")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Response Matrix", f"response_{tab.plane.name.lower()}.csv", "CSV Files (*.csv)")
        if not path:
            return
        df = pd.DataFrame(tab.matrix, index=tab.reading_labels, columns=tab.setting_labels)
        df.to_csv(path)
        self.statusBar().showMessage(f"Plane saved to {path}", 4000)


def _run_window(payload: Dict[str, any]) -> None:
    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication(sys.argv)
        created = True
    apply_dark_theme(app)
    window = AnalysisWindow()
    mode = payload.get("mode")
    if mode == "matrix":
        window.set_planes(payload["matrix"], payload["reading_labels"], payload["setting_labels"], payload.get("error_matrix"), payload.get("plane_map"))
    elif mode == "csv":
        df = pd.read_csv(payload["csv_file"], index_col=0)
        window.set_planes(df.to_numpy(dtype=float), df.index.astype(str).tolist(), df.columns.astype(str).tolist(), None, None)
    else:
        QMessageBox.critical(window, "Launch error", "Unsupported mode")
    window.show()
    if created:
        sys.exit(app.exec_())


def launch_from_arrays(matrix: np.ndarray, reading_labels: List[str], setting_labels: List[str], error_matrix: Optional[np.ndarray] = None, plane_map: Optional[Dict[str, List[int]]] = None) -> multiprocessing.Process:
    payload = {
        "mode": "matrix",
        "matrix": np.asarray(matrix, dtype=float),
        "reading_labels": list(reading_labels),
        "setting_labels": list(setting_labels),
        "error_matrix": None if error_matrix is None else np.asarray(error_matrix, dtype=float),
        "plane_map": plane_map,
    }
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_run_window, args=(payload,))
    proc.start()
    return proc


def launch_from_csv(csv_path: Path) -> multiprocessing.Process:
    payload = {
        "mode": "csv",
        "csv_file": str(Path(csv_path).resolve()),
    }
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_run_window, args=(payload,))
    proc.start()
    return proc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Response matrix analysis viewer")
    parser.add_argument("--file", "-f", dest="csv_file", help="CSV file to load on startup")
    args = parser.parse_args(argv)
    payload = None
    if args.csv_file:
        payload = {
            "mode": "csv",
            "csv_file": args.csv_file,
        }
    else:
        parser.print_help()
        return 0
    _run_window(payload)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
