import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import (
    QMainWindow, QFileDialog, QMessageBox,
    QGridLayout, QVBoxLayout, QHBoxLayout, QTabWidget, QSplitter,
    QWidget, QLabel, QLineEdit, QTableWidget, QTableWidgetItem,
    QPushButton, QAction, QListWidget, QSpinBox, QListWidgetItem
)

from numpy.fft import fft
from mpl_canvas import MplCanvas  # Relative import of our MplCanvas class

###############################################################################
# Main Application
###############################################################################
class ResponseAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ORM Application - Dark Theme")
        self.setGeometry(100, 100, 1300, 800)

        # Data & Analysis containers
        self.df = None
        self.corrector_names_txt = []
        self.bpm_names_txt = []
        self.actual_correctors = []
        self.actual_bpm_h = []
        self.actual_bpm_v = []
        self.excluded_bpm = []

        # Corrector + BPM errors
        self.corrector_errors = {}
        self.bpm_errors = {}

        # Orbit response matrix
        self.R_measured_H = None
        self.R_measured_V = None
        # Error in the matrix
        self.ERR_measured_H = None
        self.ERR_measured_V = None
        # Actual BPM/corrector amplitudes for each element
        self.bpm_amplitudes_H = None
        self.bpm_amplitudes_V = None
        self.corr_amplitudes_H = None
        self.corr_amplitudes_V = None

        # Default plot font size
        self.plot_font_size = 12

        # We'll store references to MplCanvas so we can easily re-apply font
        self.all_canvases = []

        self._initUI()
        self._setDarkTheme()

    def _initUI(self):
        """Build the UI: menu bar, main tabs, status row, etc."""
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("File")

        loadTxtAction = QAction("Load Device Lists (Txt)", self)
        loadTxtAction.triggered.connect(self.loadDeviceLists)
        fileMenu.addAction(loadTxtAction)

        openCSVAction = QAction("Open CSV", self)
        openCSVAction.triggered.connect(self.openCSVFile)
        fileMenu.addAction(openCSVAction)

        exitAction = QAction("Exit", self)
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

        helpMenu = menubar.addMenu("Help")
        docAction = QAction("Documentation", self)
        docAction.triggered.connect(self.openDocumentation)
        helpMenu.addAction(docAction)

        # Central layout
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        mainLayout = QGridLayout(centralWidget)

        # Main tab widget
        self.mainTabs = QTabWidget()
        mainLayout.addWidget(self.mainTabs, 0, 0, 1, 1)

        # Create each major tab
        self._createCorrectorsTab()
        self._createBPMTab()
        self._createResponseMatrixTab()
        self._createErrorsTab()
        self._createExcludedBPMsTab()

        # Status panel at bottom
        self.statusPanel = QWidget()
        statusLayout = QHBoxLayout(self.statusPanel)
        # Show loaded file
        self.importedFileLabel = QLabel("Loaded CSV:")
        self.importedFileEdit = QLineEdit()
        self.importedFileEdit.setReadOnly(True)
        statusLayout.addWidget(self.importedFileLabel)
        statusLayout.addWidget(self.importedFileEdit)

        # Spinbox for font size
        lblFont = QLabel("Plot Font Size:")
        statusLayout.addWidget(lblFont)
        self.spinFontSize = QSpinBox()
        self.spinFontSize.setRange(8, 40)
        self.spinFontSize.setValue(self.plot_font_size)
        self.spinFontSize.valueChanged.connect(self.onFontSizeChanged)
        statusLayout.addWidget(self.spinFontSize)

        mainLayout.addWidget(self.statusPanel, 1, 0, 1, 1)

    ###########################################################################
    # Dark Theme
    ###########################################################################
    def _setDarkTheme(self):
        from PyQt5.QtWidgets import QApplication
        QApplication.setStyle("Fusion")
        palette = QPalette()

        base_bg = QColor(45, 45, 45)
        alt_bg = QColor(60, 60, 60)
        text_fg = QColor(220, 220, 220)
        highlight_bg = QColor(100, 100, 150)
        highlight_fg = QColor(255, 255, 255)

        palette.setColor(QPalette.Window, base_bg)
        palette.setColor(QPalette.WindowText, text_fg)
        palette.setColor(QPalette.Base, alt_bg)
        palette.setColor(QPalette.AlternateBase, base_bg)
        palette.setColor(QPalette.ToolTipBase, text_fg)
        palette.setColor(QPalette.ToolTipText, text_fg)
        palette.setColor(QPalette.Text, text_fg)
        palette.setColor(QPalette.Button, alt_bg)
        palette.setColor(QPalette.ButtonText, text_fg)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, highlight_bg)
        palette.setColor(QPalette.HighlightedText, highlight_fg)
        self.setPalette(palette)

        style_sheet = """
            QWidget {
                background-color: #2D2D2D;
                color: #DDDDDD;
            }
            QLineEdit {
                background-color: #3C3C3C;
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #5A5A5A;
                color: #FFFFFF;
                border: none;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #707070;
            }
            QTabBar::tab {
                background-color: #3C3C3C;
                color: #DDDDDD;
                padding: 6px;
            }
            QTabBar::tab:selected {
                background-color: #2C2C2C;
                color: #FFFFFF;
            }
            QTableWidget {
                background-color: #3C3C3C;
                color: #FFFFFF;
                gridline-color: #AAAAAA;
            }
            QHeaderView::section {
                background-color: #4C4C4C;
                color: #FFFFFF;
            }
            QLabel {
                color: #FFFFFF;
            }
        """
        self.setStyleSheet(style_sheet)

    def onFontSizeChanged(self, val):
        """User changed the spinbox for font size."""
        self.plot_font_size = val
        # Re-apply to each MplCanvas
        for c in self.all_canvases:
            self._applyPlotFont(c)
            c.draw()

    ###########################################################################
    # Helper Methods for Plotting
    ###########################################################################
    def _clearPlot(self, canvas: MplCanvas):
        """Clear the canvas axes and redraw."""
        canvas.axes.clear()
        canvas.draw()

    def _plotTimeDomainData(self, canvas: MplCanvas, selectedItems, dataFrame, title):
        """Plot time-domain signals on the given canvas."""
        if dataFrame is None:
            return
        for it in selectedItems:
            dev_name = it.text()
            if dev_name in dataFrame.columns:
                data = dataFrame[dev_name].values
                x = np.arange(len(data))
                canvas.axes.plot(x, data, label=dev_name)
        canvas.axes.set_title(title)
        canvas.axes.legend(loc="best")
        self._applyPlotFont(canvas)
        canvas.draw()

    def _plotFrequencyDomainData(self, canvas: MplCanvas, selectedItems, dataFrame, title):
        """Plot frequency-domain (FFT) signals on the given canvas."""
        if dataFrame is None:
            return
        for it in selectedItems:
            dev_name = it.text()
            if dev_name in dataFrame.columns:
                data = dataFrame[dev_name].values
                N = len(data)
                fft_vals = fft(data)
                freq = np.fft.fftfreq(N, d=1.0)[1:N//2]
                amp = np.abs(fft_vals)[1:N//2]
                amp_scaled = (2.0 / N) * amp
                canvas.axes.plot(freq, amp_scaled, label=dev_name)
        canvas.axes.set_title(title)
        canvas.axes.legend(loc="best")
        self._applyPlotFont(canvas)
        canvas.draw()

    ###########################################################################
    # Excluded BPMs Tab
    ###########################################################################
    def _createExcludedBPMsTab(self):
        self.tabExcludedBPMs = QWidget()
        self.mainTabs.addTab(self.tabExcludedBPMs, "Excluded BPMs")
        vbox_excl = QVBoxLayout(self.tabExcludedBPMs)
        self.tableExcludedBPMs = QTableWidget()
        self.tableExcludedBPMs.setColumnCount(1)
        self.tableExcludedBPMs.setHorizontalHeaderLabels(["BPM Name"])
        vbox_excl.addWidget(self.tableExcludedBPMs)

        lbl_info = QLabel("BPMs that read 0 for all samples are excluded from the ORM.")
        vbox_excl.addWidget(lbl_info)

    def populateExcludedBPMsTable(self):
        self.tableExcludedBPMs.setRowCount(len(self.excluded_bpm))
        for i, bpm_name in enumerate(self.excluded_bpm):
            self.tableExcludedBPMs.setItem(i, 0, QTableWidgetItem(bpm_name))

    ###########################################################################
    # 1) Correctors Tab
    ###########################################################################
    def _createCorrectorsTab(self):
        self.tabCorrectors = QWidget()
        self.mainTabs.addTab(self.tabCorrectors, "Correctors")

        vbox = QVBoxLayout(self.tabCorrectors)
        self.tabGroupCorr = QTabWidget()
        vbox.addWidget(self.tabGroupCorr)

        # (A) Parameters
        self.tabCorrParams = QWidget()
        self.tabGroupCorr.addTab(self.tabCorrParams, "Parameters")
        vbox_params = QVBoxLayout(self.tabCorrParams)
        self.tableCorrParams = QTableWidget()
        vbox_params.addWidget(self.tableCorrParams)
        btnSaveCorrParamsTable = QPushButton("Save Corrector Parameters (CSV)")
        btnSaveCorrParamsTable.clicked.connect(self.onSaveCorrParamsTable)
        vbox_params.addWidget(btnSaveCorrParamsTable)

        # (B) Time Domain
        self.tabCorrTime = QWidget()
        self.tabGroupCorr.addTab(self.tabCorrTime, "Time Domain")
        vbox_ctime = QVBoxLayout(self.tabCorrTime)

        self.listCorrTime = QListWidget()
        self.listCorrTime.setSelectionMode(QListWidget.MultiSelection)
        vbox_ctime.addWidget(QLabel("Select Correctors for Time Plot:"))
        vbox_ctime.addWidget(self.listCorrTime)

        hbox_corr_time_btn = QHBoxLayout()
        btnClearCorrTimePlot = QPushButton("Clear Plot")
        btnClearCorrTimePlot.clicked.connect(self.onClearCorrTimePlot)
        hbox_corr_time_btn.addWidget(btnClearCorrTimePlot)
        btnPlotCorrTime = QPushButton("Re-Plot Selected")
        btnPlotCorrTime.clicked.connect(self.onPlotCorrTimeSelected)
        hbox_corr_time_btn.addWidget(btnPlotCorrTime)
        vbox_ctime.addLayout(hbox_corr_time_btn)

        self.canvasCorrTime = MplCanvas(self, width=5, height=4)
        self.toolbarCorrTime = None
        self.toolbarCorrTime = NavigationToolbar2QT(self.canvasCorrTime, self)
        vbox_ctime.addWidget(self.toolbarCorrTime)
        vbox_ctime.addWidget(self.canvasCorrTime)
        self.all_canvases.append(self.canvasCorrTime)

        # (C) Frequency Domain
        self.tabCorrFreq = QWidget()
        self.tabGroupCorr.addTab(self.tabCorrFreq, "Frequency Domain")
        vbox_cfreq = QVBoxLayout(self.tabCorrFreq)

        self.listCorrFreq = QListWidget()
        self.listCorrFreq.setSelectionMode(QListWidget.MultiSelection)
        vbox_cfreq.addWidget(QLabel("Select Correctors for Freq Plot:"))
        vbox_cfreq.addWidget(self.listCorrFreq)

        hbox_corr_freq_btn = QHBoxLayout()
        btnClearCorrFreqPlot = QPushButton("Clear Plot")
        btnClearCorrFreqPlot.clicked.connect(self.onClearCorrFreqPlot)
        hbox_corr_freq_btn.addWidget(btnClearCorrFreqPlot)
        btnPlotCorrFreq = QPushButton("Re-Plot Selected")
        btnPlotCorrFreq.clicked.connect(self.onPlotCorrFreqSelected)
        hbox_corr_freq_btn.addWidget(btnPlotCorrFreq)
        vbox_cfreq.addLayout(hbox_corr_freq_btn)

        self.canvasCorrFreq = MplCanvas(self, width=5, height=4)
        self.toolbarCorrFreq = NavigationToolbar2QT(self.canvasCorrFreq, self)
        vbox_cfreq.addWidget(self.toolbarCorrFreq)
        vbox_cfreq.addWidget(self.canvasCorrFreq)
        self.all_canvases.append(self.canvasCorrFreq)

    def onClearCorrTimePlot(self):
        self._clearPlot(self.canvasCorrTime)

    def onPlotCorrTimeSelected(self):
        self.onClearCorrTimePlot()
        selectedItems = self.listCorrTime.selectedItems()
        self._plotTimeDomainData(self.canvasCorrTime, selectedItems, self.df, "Selected Correctors - Time Domain")

    def onClearCorrFreqPlot(self):
        self._clearPlot(self.canvasCorrFreq)

    def onPlotCorrFreqSelected(self):
        self.onClearCorrFreqPlot()
        selectedItems = self.listCorrFreq.selectedItems()
        self._plotFrequencyDomainData(self.canvasCorrFreq, selectedItems, self.df, "Selected Correctors - Frequency Domain")

    def onSaveCorrParamsTable(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Corrector Params Table", "", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            self._exportQTableWidgetToCSV(self.tableCorrParams, file_path)
            QMessageBox.information(self, "Export Successful", f"Table exported to {file_path}")
        except Exception as ex:
            QMessageBox.critical(self, "Export Error", str(ex))

    ###########################################################################
    # 2) BPMs Tab
    ###########################################################################
    def _createBPMTab(self):
        self.tabBPMs = QWidget()
        self.mainTabs.addTab(self.tabBPMs, "BPMs")

        self.tabGroupBPM = QTabWidget()
        vbox = QVBoxLayout(self.tabBPMs)
        vbox.addWidget(self.tabGroupBPM)

        # Time Domain sub-tab
        self.tabBPMTime = QWidget()
        self.tabGroupBPM.addTab(self.tabBPMTime, "Time Domain")
        vbox_bpm_time = QVBoxLayout(self.tabBPMTime)

        # Horizontal
        labelH = QLabel("Select Horizontal BPM(s) for Time Plot:")
        vbox_bpm_time.addWidget(labelH)
        self.listBPMTimeH = QListWidget()
        self.listBPMTimeH.setSelectionMode(QListWidget.MultiSelection)
        vbox_bpm_time.addWidget(self.listBPMTimeH)

        hbox_bpm_time_h = QHBoxLayout()
        btnClearBPMTimeH = QPushButton("Clear Plot")
        btnClearBPMTimeH.clicked.connect(self.onClearBPMTimeH)
        hbox_bpm_time_h.addWidget(btnClearBPMTimeH)
        btnPlotBPMTimeH = QPushButton("Re-Plot")
        btnPlotBPMTimeH.clicked.connect(self.onPlotBPMTimeHSelected)
        hbox_bpm_time_h.addWidget(btnPlotBPMTimeH)
        vbox_bpm_time.addLayout(hbox_bpm_time_h)

        self.canvasBPMTimeH = MplCanvas(self, width=5, height=3)
        self.toolbarBPMTimeH = NavigationToolbar2QT(self.canvasBPMTimeH, self)
        vbox_bpm_time.addWidget(self.toolbarBPMTimeH)
        vbox_bpm_time.addWidget(self.canvasBPMTimeH)
        self.all_canvases.append(self.canvasBPMTimeH)

        # Vertical
        labelV = QLabel("Select Vertical BPM(s) for Time Plot:")
        vbox_bpm_time.addWidget(labelV)
        self.listBPMTimeV = QListWidget()
        self.listBPMTimeV.setSelectionMode(QListWidget.MultiSelection)
        vbox_bpm_time.addWidget(self.listBPMTimeV)

        hbox_bpm_time_v = QHBoxLayout()
        btnClearBPMTimeV = QPushButton("Clear Plot")
        btnClearBPMTimeV.clicked.connect(self.onClearBPMTimeV)
        hbox_bpm_time_v.addWidget(btnClearBPMTimeV)
        btnPlotBPMTimeV = QPushButton("Re-Plot")
        btnPlotBPMTimeV.clicked.connect(self.onPlotBPMTimeVSelected)
        hbox_bpm_time_v.addWidget(btnPlotBPMTimeV)
        vbox_bpm_time.addLayout(hbox_bpm_time_v)

        self.canvasBPMTimeV = MplCanvas(self, width=5, height=3)
        self.toolbarBPMTimeV = NavigationToolbar2QT(self.canvasBPMTimeV, self)
        vbox_bpm_time.addWidget(self.toolbarBPMTimeV)
        vbox_bpm_time.addWidget(self.canvasBPMTimeV)
        self.all_canvases.append(self.canvasBPMTimeV)

        # Frequency Domain sub-tab
        self.tabBPMFreq = QWidget()
        self.tabGroupBPM.addTab(self.tabBPMFreq, "Frequency Domain")
        vbox_bpm_freq = QVBoxLayout(self.tabBPMFreq)

        # Horizontal freq
        labelHf = QLabel("Select Horizontal BPM(s) for Freq Plot:")
        vbox_bpm_freq.addWidget(labelHf)
        self.listBPMFreqH = QListWidget()
        self.listBPMFreqH.setSelectionMode(QListWidget.MultiSelection)
        vbox_bpm_freq.addWidget(self.listBPMFreqH)

        hbox_bpm_freq_h = QHBoxLayout()
        btnClearBPMFreqH = QPushButton("Clear Plot")
        btnClearBPMFreqH.clicked.connect(self.onClearBPMFreqH)
        hbox_bpm_freq_h.addWidget(btnClearBPMFreqH)
        btnPlotBPMFreqH = QPushButton("Re-Plot")
        btnPlotBPMFreqH.clicked.connect(self.onPlotBPMFreqHSelected)
        hbox_bpm_freq_h.addWidget(btnPlotBPMFreqH)
        vbox_bpm_freq.addLayout(hbox_bpm_freq_h)

        self.canvasBPMFreqH = MplCanvas(self, width=5, height=3)
        self.toolbarBPMFreqH = NavigationToolbar2QT(self.canvasBPMFreqH, self)
        vbox_bpm_freq.addWidget(self.toolbarBPMFreqH)
        vbox_bpm_freq.addWidget(self.canvasBPMFreqH)
        self.all_canvases.append(self.canvasBPMFreqH)

        # Vertical freq
        labelVf = QLabel("Select Vertical BPM(s) for Freq Plot:")
        vbox_bpm_freq.addWidget(labelVf)
        self.listBPMFreqV = QListWidget()
        self.listBPMFreqV.setSelectionMode(QListWidget.MultiSelection)
        vbox_bpm_freq.addWidget(self.listBPMFreqV)

        hbox_bpm_freq_v = QHBoxLayout()
        btnClearBPMFreqV = QPushButton("Clear Plot")
        btnClearBPMFreqV.clicked.connect(self.onClearBPMFreqV)
        hbox_bpm_freq_v.addWidget(btnClearBPMFreqV)
        btnPlotBPMFreqV = QPushButton("Re-Plot")
        btnPlotBPMFreqV.clicked.connect(self.onPlotBPMFreqVSelected)
        hbox_bpm_freq_v.addWidget(btnPlotBPMFreqV)
        vbox_bpm_freq.addLayout(hbox_bpm_freq_v)

        self.canvasBPMFreqV = MplCanvas(self, width=5, height=3)
        self.toolbarBPMFreqV = NavigationToolbar2QT(self.canvasBPMFreqV, self)
        vbox_bpm_freq.addWidget(self.toolbarBPMFreqV)
        vbox_bpm_freq.addWidget(self.canvasBPMFreqV)
        self.all_canvases.append(self.canvasBPMFreqV)

    # BPM time/freq methods
    def onClearBPMTimeH(self):
        self._clearPlot(self.canvasBPMTimeH)

    def onPlotBPMTimeHSelected(self):
        self.onClearBPMTimeH()
        items = self.listBPMTimeH.selectedItems()
        self._plotTimeDomainData(self.canvasBPMTimeH, items, self.df, "Selected Horizontal BPM(s) - Time")

    def onClearBPMTimeV(self):
        self._clearPlot(self.canvasBPMTimeV)

    def onPlotBPMTimeVSelected(self):
        self.onClearBPMTimeV()
        items = self.listBPMTimeV.selectedItems()
        self._plotTimeDomainData(self.canvasBPMTimeV, items, self.df, "Selected Vertical BPM(s) - Time")

    def onClearBPMFreqH(self):
        self._clearPlot(self.canvasBPMFreqH)

    def onPlotBPMFreqHSelected(self):
        self.onClearBPMFreqH()
        items = self.listBPMFreqH.selectedItems()
        self._plotFrequencyDomainData(self.canvasBPMFreqH, items, self.df, "Selected Horizontal BPM(s) - Freq")

    def onClearBPMFreqV(self):
        self._clearPlot(self.canvasBPMFreqV)

    def onPlotBPMFreqVSelected(self):
        self.onClearBPMFreqV()
        items = self.listBPMFreqV.selectedItems()
        self._plotFrequencyDomainData(self.canvasBPMFreqV, items, self.df, "Selected Vertical BPM(s) - Freq")

    ###########################################################################
    # 3) Response Matrix Tab
    ###########################################################################
    def _createResponseMatrixTab(self):
        self.tabRespMatrix = QWidget()
        self.mainTabs.addTab(self.tabRespMatrix, "Response Matrix")
        vbox_rm = QVBoxLayout(self.tabRespMatrix)

        self.tabGroupResp = QTabWidget()
        vbox_rm.addWidget(self.tabGroupResp)

        # Horizontal
        self.tabRM_H = QWidget()
        self.tabGroupResp.addTab(self.tabRM_H, "Horizontal")
        splitter_h = QSplitter(Qt.Horizontal)

        self.tableRM_H = QTableWidget()
        splitter_h.addWidget(self.tableRM_H)

        right_container_h = QWidget()
        rch_layout = QVBoxLayout(right_container_h)
        self.canvasRM_H = MplCanvas(self, width=6, height=3)
        self.toolbarRM_H = NavigationToolbar2QT(self.canvasRM_H, self)
        rch_layout.addWidget(self.toolbarRM_H)
        rch_layout.addWidget(self.canvasRM_H)
        splitter_h.addWidget(right_container_h)
        self.all_canvases.append(self.canvasRM_H)

        layout_tab_rmh = QVBoxLayout(self.tabRM_H)
        layout_tab_rmh.addWidget(splitter_h)
        hbox_btn_rmh = QHBoxLayout()
        btnSaveRM_H_Table = QPushButton("Save Table (H)")
        btnSaveRM_H_Table.clicked.connect(self.onSaveResponseMatrixHTable)
        hbox_btn_rmh.addWidget(btnSaveRM_H_Table)
        btnSaveRM_H_Plot = QPushButton("Save Plot (H)")
        btnSaveRM_H_Plot.clicked.connect(self.onSaveResponseMatrixHPlot)
        hbox_btn_rmh.addWidget(btnSaveRM_H_Plot)
        layout_tab_rmh.addLayout(hbox_btn_rmh)

        # Vertical
        self.tabRM_V = QWidget()
        self.tabGroupResp.addTab(self.tabRM_V, "Vertical")
        splitter_v = QSplitter(Qt.Horizontal)

        self.tableRM_V = QTableWidget()
        splitter_v.addWidget(self.tableRM_V)

        right_container_v = QWidget()
        rcv_layout = QVBoxLayout(right_container_v)
        self.canvasRM_V = MplCanvas(self, width=6, height=3)
        self.toolbarRM_V = NavigationToolbar2QT(self.canvasRM_V, self)
        rcv_layout.addWidget(self.toolbarRM_V)
        rcv_layout.addWidget(self.canvasRM_V)
        splitter_v.addWidget(right_container_v)
        self.all_canvases.append(self.canvasRM_V)

        layout_tab_rmv = QVBoxLayout(self.tabRM_V)
        layout_tab_rmv.addWidget(splitter_v)
        hbox_btn_rmv = QHBoxLayout()
        btnSaveRM_V_Table = QPushButton("Save Table (V)")
        btnSaveRM_V_Table.clicked.connect(self.onSaveResponseMatrixVTable)
        hbox_btn_rmv.addWidget(btnSaveRM_V_Table)
        btnSaveRM_V_Plot = QPushButton("Save Plot (V)")
        btnSaveRM_V_Plot.clicked.connect(self.onSaveResponseMatrixVPlot)
        hbox_btn_rmv.addWidget(btnSaveRM_V_Plot)
        layout_tab_rmv.addLayout(hbox_btn_rmv)

    def onSaveResponseMatrixHTable(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Horizontal RM Table", "", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            self._exportQTableWidgetToCSV(self.tableRM_H, file_path)
        except Exception as ex:
            QMessageBox.critical(self, "Export Error", str(ex))

    def onSaveResponseMatrixHPlot(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Horizontal RM Plot", "", "PNG Files (*.png);;JPG Files (*.jpg)")
        if not file_path:
            return
        try:
            self.canvasRM_H.fig.tight_layout()
            self.canvasRM_H.fig.savefig(file_path, dpi=150, bbox_inches='tight')
        except Exception as ex:
            QMessageBox.critical(self, "Save Error", str(ex))

    def onSaveResponseMatrixVTable(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Vertical RM Table", "", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            self._exportQTableWidgetToCSV(self.tableRM_V, file_path)
        except Exception as ex:
            QMessageBox.critical(self, "Export Error", str(ex))

    def onSaveResponseMatrixVPlot(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Vertical RM Plot", "", "PNG Files (*.png);;JPG Files (*.jpg)")
        if not file_path:
            return
        try:
            self.canvasRM_V.fig.tight_layout()
            self.canvasRM_V.fig.savefig(file_path, dpi=150, bbox_inches='tight')
        except Exception as ex:
            QMessageBox.critical(self, "Save Error", str(ex))

    ###########################################################################
    # 4) Errors Tab
    ###########################################################################
    def _createErrorsTab(self):
        self.tabErrors = QWidget()
        self.mainTabs.addTab(self.tabErrors, "Errors")

        vbox_err = QVBoxLayout(self.tabErrors)
        self.tabGroupErrors = QTabWidget()
        vbox_err.addWidget(self.tabGroupErrors)

        # Horizontal
        self.tabErrH = QWidget()
        self.tabGroupErrors.addTab(self.tabErrH, "Horizontal")
        splitter_errh = QSplitter(Qt.Horizontal)
        self.tableErrH = QTableWidget()
        splitter_errh.addWidget(self.tableErrH)

        right_container_errh = QWidget()
        rceh_layout = QVBoxLayout(right_container_errh)
        self.canvasErrH = MplCanvas(self, width=5, height=3)
        self.toolbarErrH = NavigationToolbar2QT(self.canvasErrH, self)
        rceh_layout.addWidget(self.toolbarErrH)
        rceh_layout.addWidget(self.canvasErrH)
        splitter_errh.addWidget(right_container_errh)
        self.all_canvases.append(self.canvasErrH)

        layout_tab_errh = QVBoxLayout(self.tabErrH)
        layout_tab_errh.addWidget(splitter_errh)

        # Vertical
        self.tabErrV = QWidget()
        self.tabGroupErrors.addTab(self.tabErrV, "Vertical")
        splitter_errv = QSplitter(Qt.Horizontal)
        self.tableErrV = QTableWidget()
        splitter_errv.addWidget(self.tableErrV)

        right_container_errv = QWidget()
        rcev_layout = QVBoxLayout(right_container_errv)
        self.canvasErrV = MplCanvas(self, width=5, height=3)
        self.toolbarErrV = NavigationToolbar2QT(self.canvasErrV, self)
        rcev_layout.addWidget(self.toolbarErrV)
        rcev_layout.addWidget(self.canvasErrV)
        splitter_errv.addWidget(right_container_errv)
        self.all_canvases.append(self.canvasErrV)

        layout_tab_errv = QVBoxLayout(self.tabErrV)
        layout_tab_errv.addWidget(splitter_errv)

    ###########################################################################
    # CSV Export
    ###########################################################################
    def _exportQTableWidgetToCSV(self, tableWidget, csv_file_path):
        import csv
        row_count = tableWidget.rowCount()
        col_count = tableWidget.columnCount()

        col_headers = []
        for c in range(col_count):
            hdr_item = tableWidget.horizontalHeaderItem(c)
            if hdr_item:
                col_headers.append(hdr_item.text())
            else:
                col_headers.append(f"Column_{c}")

        row_labels = []
        for r in range(row_count):
            vh_item = tableWidget.verticalHeaderItem(r)
            if vh_item:
                row_labels.append(vh_item.text())
            else:
                row_labels.append(f"Row_{r}")

        top_row = [""] + col_headers
        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(top_row)
            for r in range(row_count):
                row_data = [row_labels[r]]
                for c in range(col_count):
                    item = tableWidget.item(r, c)
                    cell_text = item.text() if item else ""
                    row_data.append(cell_text)
                writer.writerow(row_data)

    ###########################################################################
    # Plot Font Helper
    ###########################################################################
    def _applyPlotFont(self, canvas: MplCanvas):
        """Adjust axis label/title/tick fonts to the chosen size."""
        fs = self.plot_font_size
        canvas.axes.set_xlabel(canvas.axes.get_xlabel(), fontsize=fs)
        canvas.axes.set_ylabel(canvas.axes.get_ylabel(), fontsize=fs)
        canvas.axes.set_title(canvas.axes.get_title(), fontsize=fs)
        canvas.axes.tick_params(axis='both', labelsize=fs)

    ###########################################################################
    # Analysis Pipeline
    ###########################################################################
    def performAnalysis(self):
        if self.df is None:
            return

        # 1) compute corrector & BPM errors
        self.computeCorrectorErrors()
        self.computeBPMErrors()

        # 2) fill corrector param table
        self.fillCorrectorParameters()

        # 3) user can selectively plot correctors & BPM

        # 4) build response matrix
        self.buildResponseMatrix()

        # 5) build error matrix
        self.buildORMErrorMatrix()

    def computeCorrectorErrors(self):
        """For each corrector: remove dominant freq → RMS remainder => error."""
        self.corrector_errors.clear()
        for cdev in self.actual_correctors:
            data = self.df[cdev].values
            fft_c = fft(data)
            amp_c = np.abs(fft_c)
            idx_max_c = np.argmax(amp_c[1:]) + 1
            amp_c[idx_max_c] = 0
            remainder = amp_c
            rms_err = np.sqrt(np.mean(remainder**2))
            self.corrector_errors[cdev] = rms_err

    def computeBPMErrors(self):
        """For each BPM: remove each corrector's freq → RMS remainder => error."""
        self.bpm_errors.clear()

        # gather correctors' freq
        corr_freqs = {}
        for cdev in self.actual_correctors:
            data_c = self.df[cdev].values
            fft_c = fft(data_c)
            amp_c = np.abs(fft_c)
            idx_max_c = np.argmax(amp_c[1:]) + 1
            freq_c = np.fft.fftfreq(len(data_c), d=1.0)[idx_max_c]
            corr_freqs[cdev] = freq_c
        all_corr_freqs = list(corr_freqs.values())

        for bdev in (self.actual_bpm_h + self.actual_bpm_v):
            data_b = self.df[bdev].values
            fft_b = fft(data_b)
            amp_b = np.abs(fft_b)
            N = len(data_b)
            freq_b = np.fft.fftfreq(N, d=1.0)
            for fc in all_corr_freqs:
                idx_near = np.argmin(np.abs(freq_b - fc))
                amp_b[idx_near] = 0
            remainder = amp_b
            rms_err = np.sqrt(np.mean(remainder**2))
            self.bpm_errors[bdev] = rms_err

    def fillCorrectorParameters(self):
        """Fill a table with corrector name, freq idx, etc."""
        corr = self.actual_correctors
        self.tableCorrParams.setRowCount(len(corr))
        self.tableCorrParams.setColumnCount(5)
        self.tableCorrParams.setHorizontalHeaderLabels([
            "Corrector", "Peak-to-Peak", "Dominant Freq Idx", "Dominant Freq (Hz)", "Max FFT Amp"
        ])

        for i, cdev in enumerate(corr):
            data = self.df[cdev].values
            p2p = data.max() - data.min()
            fft_data = fft(data)
            fft_amp = np.abs(fft_data)
            idx_max = np.argmax(fft_amp[1:]) + 1
            max_val = fft_amp[idx_max]
            freq_arr = np.fft.fftfreq(len(data), d=1.0)
            dom_freq = freq_arr[idx_max]

            self.tableCorrParams.setItem(i, 0, QTableWidgetItem(cdev))
            self.tableCorrParams.setItem(i, 1, QTableWidgetItem(f"{p2p:.3f}"))
            self.tableCorrParams.setItem(i, 2, QTableWidgetItem(str(idx_max)))
            self.tableCorrParams.setItem(i, 3, QTableWidgetItem(f"{dom_freq:.3f}"))
            self.tableCorrParams.setItem(i, 4, QTableWidgetItem(f"{max_val:.3f}"))

    def buildResponseMatrix(self):
        """Compute orbit response matrix with BPM_amp / Corr_amp at corrector freq."""
        corr = self.actual_correctors
        freq_c_list = []
        amp_c_list = []
        for cdev in corr:
            data_c = self.df[cdev].values
            fft_c = fft(data_c)
            amp_c = np.abs(fft_c)
            idx_max_c = np.argmax(amp_c[1:]) + 1
            freq_c = np.fft.fftfreq(len(data_c), d=1.0)[idx_max_c]
            freq_c_list.append(freq_c)
            amp_c_list.append(amp_c[idx_max_c])

        # Horizontal
        bpmh = self.actual_bpm_h
        nBPM_H = len(bpmh)
        nCorr = len(corr)
        self.R_measured_H = np.zeros((nBPM_H, nCorr), dtype=float)
        self.bpm_amplitudes_H = np.zeros((nBPM_H, nCorr), dtype=float)
        self.corr_amplitudes_H = np.zeros((nBPM_H, nCorr), dtype=float)

        self.tableRM_H.setRowCount(nBPM_H)
        self.tableRM_H.setColumnCount(nCorr)
        self.tableRM_H.setVerticalHeaderLabels(bpmh)
        self.tableRM_H.setHorizontalHeaderLabels(corr)

        for j, cdev in enumerate(corr):
            freq_c = freq_c_list[j]
            corr_amp = amp_c_list[j]
            for i, bdev in enumerate(bpmh):
                data_b = self.df[bdev].values
                fft_b = fft(data_b)
                amp_b = np.abs(fft_b)
                N = len(data_b)
                freq_b = np.fft.fftfreq(N, d=1.0)
                idx_near = np.argmin(np.abs(freq_b - freq_c))
                bpm_amp = amp_b[idx_near]
                self.bpm_amplitudes_H[i, j] = bpm_amp
                self.corr_amplitudes_H[i, j] = corr_amp
                Rij = bpm_amp / corr_amp if corr_amp != 0 else 0.0
                self.R_measured_H[i, j] = Rij
                self.tableRM_H.setItem(i, j, QTableWidgetItem(f"{Rij:.4f}"))

        # Plot heatmap
        self.canvasRM_H.axes.clear()
        if self.R_measured_H.size > 0:
            im = self.canvasRM_H.axes.imshow(self.R_measured_H, aspect='auto', cmap='jet')
            self.canvasRM_H.axes.set_title("Horizontal Orbit Response")
            self._applyPlotFont(self.canvasRM_H)
            self.canvasRM_H.axes.set_xticks(range(nCorr))
            self.canvasRM_H.axes.set_yticks(range(nBPM_H))
            self.canvasRM_H.axes.set_xticklabels(corr, rotation=90)
            self.canvasRM_H.axes.set_yticklabels(bpmh)
            self.canvasRM_H.fig.colorbar(im, ax=self.canvasRM_H.axes, orientation='vertical')
        self.canvasRM_H.fig.tight_layout()
        self.canvasRM_H.draw()

        # Vertical
        bpmv = self.actual_bpm_v
        nBPM_V = len(bpmv)
        self.R_measured_V = np.zeros((nBPM_V, nCorr), dtype=float)
        self.bpm_amplitudes_V = np.zeros((nBPM_V, nCorr), dtype=float)
        self.corr_amplitudes_V = np.zeros((nBPM_V, nCorr), dtype=float)

        self.tableRM_V.setRowCount(nBPM_V)
        self.tableRM_V.setColumnCount(nCorr)
        self.tableRM_V.setVerticalHeaderLabels(bpmv)
        self.tableRM_V.setHorizontalHeaderLabels(corr)

        for j, cdev in enumerate(corr):
            freq_c = freq_c_list[j]
            corr_amp = amp_c_list[j]
            for i, bdev in enumerate(bpmv):
                data_b = self.df[bdev].values
                fft_b = fft(data_b)
                amp_b = np.abs(fft_b)
                N = len(data_b)
                freq_b = np.fft.fftfreq(N, d=1.0)
                idx_near = np.argmin(np.abs(freq_b - freq_c))
                bpm_amp = amp_b[idx_near]
                self.bpm_amplitudes_V[i, j] = bpm_amp
                self.corr_amplitudes_V[i, j] = corr_amp
                Rij = bpm_amp / corr_amp if corr_amp != 0.0 else 0.0
                self.R_measured_V[i, j] = Rij
                self.tableRM_V.setItem(i, j, QTableWidgetItem(f"{Rij:.4f}"))

        self.canvasRM_V.axes.clear()
        if self.R_measured_V.size > 0:
            im2 = self.canvasRM_V.axes.imshow(self.R_measured_V, aspect='auto', cmap='jet')
            self.canvasRM_V.axes.set_title("Vertical Orbit Response")
            self._applyPlotFont(self.canvasRM_V)
            self.canvasRM_V.axes.set_xticks(range(nCorr))
            self.canvasRM_V.axes.set_yticks(range(nBPM_V))
            self.canvasRM_V.axes.set_xticklabels(corr, rotation=90)
            self.canvasRM_V.axes.set_yticklabels(bpmv)
            self.canvasRM_V.fig.colorbar(im2, ax=self.canvasRM_V.axes, orientation='vertical')
        self.canvasRM_V.fig.tight_layout()
        self.canvasRM_V.draw()

    def buildORMErrorMatrix(self):
        """Propagate errors for each R_ij = BPM_amp / Corr_amp."""
        if self.R_measured_H is None or self.R_measured_V is None:
            return

        # Horizontal
        shape_h = self.R_measured_H.shape
        self.ERR_measured_H = np.zeros(shape_h, dtype=float)

        row_labels_h = [self.tableRM_H.verticalHeaderItem(r).text() for r in range(shape_h[0])]
        col_labels_h = [self.tableRM_H.horizontalHeaderItem(c).text() for c in range(shape_h[1])]

        for i, bpm_dev in enumerate(row_labels_h):
            for j, cdev in enumerate(col_labels_h):
                BPM_amp = self.bpm_amplitudes_H[i, j]
                Corr_amp = self.corr_amplitudes_H[i, j]
                eb = self.bpm_errors.get(bpm_dev, 0)
                ec = self.corrector_errors.get(cdev, 0)

                if Corr_amp == 0:
                    self.ERR_measured_H[i, j] = 0.0
                else:
                    dR2 = (eb**2)/(Corr_amp**2) + ((BPM_amp**2)/(Corr_amp**4))*(ec**2)
                    self.ERR_measured_H[i, j] = np.sqrt(dR2)

        self.tableErrH.setRowCount(shape_h[0])
        self.tableErrH.setColumnCount(shape_h[1])
        self.tableErrH.setVerticalHeaderLabels(row_labels_h)
        self.tableErrH.setHorizontalHeaderLabels(col_labels_h)

        for r in range(shape_h[0]):
            for c in range(shape_h[1]):
                val = self.ERR_measured_H[r, c]
                self.tableErrH.setItem(r, c, QTableWidgetItem(f"{val:.4e}"))

        self.canvasErrH.axes.clear()
        if self.ERR_measured_H.size > 0:
            im_h = self.canvasErrH.axes.imshow(self.ERR_measured_H, aspect='auto', cmap='jet')
            self.canvasErrH.axes.set_title("Horizontal ORM Error")
            self._applyPlotFont(self.canvasErrH)
            self.canvasErrH.axes.set_xticks(range(shape_h[1]))
            self.canvasErrH.axes.set_yticks(range(shape_h[0]))
            self.canvasErrH.axes.set_xticklabels(col_labels_h, rotation=90)
            self.canvasErrH.axes.set_yticklabels(row_labels_h)
            self.canvasErrH.fig.colorbar(im_h, ax=self.canvasErrH.axes, orientation='vertical')
        self.canvasErrH.fig.tight_layout()
        self.canvasErrH.draw()

        # Vertical
        shape_v = self.R_measured_V.shape
        self.ERR_measured_V = np.zeros(shape_v, dtype=float)

        row_labels_v = [self.tableRM_V.verticalHeaderItem(r).text() for r in range(shape_v[0])]
        col_labels_v = [self.tableRM_V.horizontalHeaderItem(c).text() for c in range(shape_v[1])]

        for i, bpm_dev in enumerate(row_labels_v):
            for j, cdev in enumerate(col_labels_v):
                BPM_amp = self.bpm_amplitudes_V[i, j]
                Corr_amp = self.corr_amplitudes_V[i, j]
                eb = self.bpm_errors.get(bpm_dev, 0)
                ec = self.corrector_errors.get(cdev, 0)

                if Corr_amp == 0:
                    self.ERR_measured_V[i, j] = 0.0
                else:
                    dR2 = (eb**2)/(Corr_amp**2) + ((BPM_amp**2)/(Corr_amp**4))*(ec**2)
                    self.ERR_measured_V[i, j] = np.sqrt(dR2)

        self.tableErrV.setRowCount(shape_v[0])
        self.tableErrV.setColumnCount(shape_v[1])
        self.tableErrV.setVerticalHeaderLabels(row_labels_v)
        self.tableErrV.setHorizontalHeaderLabels(col_labels_v)

        for r in range(shape_v[0]):
            for c in range(shape_v[1]):
                val = self.ERR_measured_V[r, c]
                self.tableErrV.setItem(r, c, QTableWidgetItem(f"{val:.4e}"))

        self.canvasErrV.axes.clear()
        if self.ERR_measured_V.size > 0:
            im_v = self.canvasErrV.axes.imshow(self.ERR_measured_V, aspect='auto', cmap='jet')
            self.canvasErrV.axes.set_title("Vertical ORM Error")
            self._applyPlotFont(self.canvasErrV)
            self.canvasErrV.axes.set_xticks(range(shape_v[1]))
            self.canvasErrV.axes.set_yticks(range(shape_v[0]))
            self.canvasErrV.axes.set_xticklabels(col_labels_v, rotation=90)
            self.canvasErrV.axes.set_yticklabels(row_labels_v)
            self.canvasErrV.fig.colorbar(im_v, ax=self.canvasErrV.axes, orientation='vertical')
        self.canvasErrV.fig.tight_layout()
        self.canvasErrV.draw()

    ###########################################################################
    # File / Menu Actions
    ###########################################################################
    def loadDeviceLists(self):
        cfile, _ = QFileDialog.getOpenFileName(self, "Open Corrector Devices Text File", "", "Text Files (*.txt)")
        if cfile:
            with open(cfile, 'r') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
                self.corrector_names_txt = lines

        bfile, _ = QFileDialog.getOpenFileName(self, "Open BPM Devices Text File", "", "Text Files (*.txt)")
        if bfile:
            with open(bfile, 'r') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
                self.bpm_names_txt = lines

        QMessageBox.information(self, "Device Lists", "Device lists loaded successfully.")

    def openCSVFile(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if not fname:
            return
        try:
            df = pd.read_csv(fname)
        except Exception as ex:
            QMessageBox.critical(self, "File Error", f"Could not read CSV:\n{ex}")
            return

        if df.isnull().values.any():
            QMessageBox.warning(self, "Data Warning", "CSV contains missing data (NaN).")

        self.df = df
        self.importedFileEdit.setText(fname)

        # Identify correctors
        self.actual_correctors = []
        for cdev in self.corrector_names_txt:
            col_r = f"{cdev}(R)"
            if col_r in df.columns:
                self.actual_correctors.append(col_r)

        # Identify BPM columns, exclude all-zero
        self.actual_bpm_h = []
        self.actual_bpm_v = []
        self.excluded_bpm = []
        for bdev in self.bpm_names_txt:
            col_r = f"{bdev}(R)"
            if col_r in df.columns:
                data_b = df[col_r].values
                if np.all(data_b == 0):
                    self.excluded_bpm.append(col_r)
                    continue
                if "BPH" in bdev.upper():
                    self.actual_bpm_h.append(col_r)
                elif "BPV" in bdev.upper():
                    self.actual_bpm_v.append(col_r)
                else:
                    self.actual_bpm_h.append(col_r)

        self.populateExcludedBPMsTable()

        # Fill the corrector + BPM lists for selective plotting
        self.listCorrTime.clear()
        self.listCorrFreq.clear()
        for cdev in self.actual_correctors:
            self.listCorrTime.addItem(cdev)
            self.listCorrFreq.addItem(cdev)

        self.listBPMTimeH.clear()
        for bdev in self.actual_bpm_h:
            self.listBPMTimeH.addItem(bdev)

        self.listBPMTimeV.clear()
        for bdev in self.actual_bpm_v:
            self.listBPMTimeV.addItem(bdev)

        self.listBPMFreqH.clear()
        for bdev in self.actual_bpm_h:
            self.listBPMFreqH.addItem(bdev)

        self.listBPMFreqV.clear()
        for bdev in self.actual_bpm_v:
            self.listBPMFreqV.addItem(bdev)

        # Now run the full pipeline
        self.performAnalysis()

    def openDocumentation(self):
        QMessageBox.information(self, "Documentation", "Here you can link to your RA_Manual.pdf or relevant doc...")
