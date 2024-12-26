import sys
from PyQt5.QtWidgets import QApplication
from response_analyzer_app import ResponseAnalyzerApp

def main():
    app = QApplication(sys.argv)
    win = ResponseAnalyzerApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
