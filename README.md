# ORM Application (Orbit Response Matrix Analyzer)

This **PyQt5** application analyzes orbit response matrices (ORM) from accelerator data. It is designed to load time-domain signals of **BPMs** (Beam Position Monitors) and **corrector magnets**, compute FFT-based spectra, build orbit response matrices, and propagate measurement errors—all within a dark-themed GUI.

🌐 **Project Website:** [https://abhishek-pathak-90.github.io/ORM-Application/](https://abhishek-pathak-90.github.io/ORM-Application/)

---

## Table of Contents
1. [Features](#features)  
2. [Preview / Screenshots](#preview--screenshots)  
3. [Project Structure](#project-structure)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [How It Works](#how-it-works)  
7. [Customization](#customization)  
8. [Contributing](#contributing)  
9. [License](#license)

---

## Features
- **Dark-Themed PyQt5 GUI**  
  A modern dark UI with multiple tabs for **Correctors**, **BPMs**, **Response Matrices**, and **Error Matrices**.

- **Device Management**  
  Load lists of corrector and BPM devices from text files.

- **CSV Data Import**  
  Open a CSV file containing time-domain signal data; the app automatically identifies valid BPM & corrector columns.

- **Plots**  
  - Time-domain plots  
  - Frequency-domain (FFT) plots  
  - Heatmap of horizontal and vertical orbit response matrices  
  - Error matrix heatmaps  

- **Data Tables**  
  - Corrector parameters (peak-to-peak, dominant frequency, etc.)  
  - BPM measurement errors  
  - Response matrices and their error tables  

- **Save & Export**  
  - Export tables as CSV  
  - Save plots as PNG/JPG  

- **Excluded BPMs**  
  BPM channels that read zeros for all samples are excluded from analysis and listed in a special tab.

- **Adjustable Plot Fonts**  
  Easily control plot font size via a spin box at the bottom.


---

## Project Structure
.
├── main.py                    # Entry point to launch the application
├── mpl_canvas.py             # Contains the MplCanvas class for embedding Matplotlib
└── response_analyzer_app.py   # PyQt5 application with all tabs, UI, and logic

- **`main.py`**  
  Minimal entry script. It creates a `QApplication` instance and initializes the main window (`ResponseAnalyzerApp`).

- **`mpl_canvas.py`**  
  A small utility class (`MplCanvas`) that sets up a Matplotlib Figure and Axes within a PyQt widget.

- **`response_analyzer_app.py`**  
  Contains the main `ResponseAnalyzerApp` class, which extends `QMainWindow`.  
  Builds the tabs: **Correctors**, **BPMs**, **Response Matrix**, **Errors**, **Excluded BPMs**.  
  Handles file loading, data parsing, plotting, and analysis logic.

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your_username/orm-application.git
   cd orm-application
   ```

2. **Install Python Dependencies**  
   The app requires Python 3.7+, PyQt5, Matplotlib, NumPy, and Pandas:
   ```bash
   pip install pyqt5 matplotlib numpy pandas
   ```
   Or from a conda environment:
   ```bash
   conda install pyqt numpy pandas matplotlib
   ```

3. **Verify Setup**  
   Check that the packages are installed correctly by running `pip list` or `conda list`.

## Usage

### Run the Application
```bash
python main.py
```
This will launch the GUI window.

### Load Device Lists (optional)
1. Go to `File → Load Device Lists (Txt)` and select corrector and BPM text files.
2. Each file should have one device name per line.

### Open CSV Data
1. Go to `File → Open CSV` to load a time-series data file.
2. The application expects columns like `CorrectorName(R)` and `BPMName(R)` (though it's configurable in the code).

### Analyze
As soon as the CSV is loaded, the app automatically:
- Identifies valid BPM and corrector columns
- Populates the tables (corrector params, BPM lists)
- Builds the Response Matrices and Error Matrices

### Plot
- **Correctors Tab**: Time and Frequency sub-tabs. Select correctors in a list widget, then click Re-Plot Selected.
- **BPMs Tab**: Similarly for horizontal and vertical BPM signals.

### Save or Export
- Any table can be exported as CSV (e.g., `Save Corrector Parameters`, `Save Table (H)`)
- Plots can be saved as PNG or JPG

---

## How It Works

### Data Import
- The user loads a CSV file containing time-domain data
- Pandas reads the data into a DataFrame `self.df`

### Analysis
1. **Compute Errors**: For each corrector/BPM signal, the dominant frequency is located via `np.argmax(FFT)`. Residual amplitude is used to compute RMS error.
2. **Build Response Matrix**: For each BPM and corrector pair, find the BPM amplitude at the corrector's dominant frequency. `R_ij = (BPM amplitude) / (corrector amplitude)`.
3. **Error Propagation**: The error matrix is computed using the partial derivative approach for each `R_ij`.

### Plotting
- **Time Domain**: Plain line plots vs. sample index
- **Frequency Domain**: FFT amplitude vs. frequency bin
- **Heatmaps**: `imshow` used to display 2D response/error matrices

---

## Customization
- Modify `response_analyzer_app.py` to add new tabs or widgets, tweak error calculations, or adjust how BPM/corrector files are read
- The styling is based on a dark Fusion theme. You can adjust it in the `_setDarkTheme()` method
- Plot Font: A spinbox in the status bar controls plot text size. See `_applyPlotFont()`

## Contributing
We welcome bug reports, feature requests, and code contributions! To contribute:

1. Fork the repository
2. Create a new branch for your feature/bug fix
3. Make changes and write tests if possible
4. Submit a pull request with a clear description

## License
MIT License

Copyright (c) 2023 [Abhishek Pathak]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

## Contact / Support
If you have questions or issues:

- Open a GitHub issue at https://github.com/your_username/orm-application/issues
- Email: walkwithtime@gmail.com