# gui.py
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QPushButton,
                             QVBoxLayout, QWidget, QLabel, QMessageBox)
from PyQt5.QtGui import QFont
from summarizer import TextSummarizer


class SummarizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Text Summarizer")
        self.setGeometry(100, 100, 600, 700)

        # Create central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout()

        # Input Text Area
        input_label = QLabel("Enter Text to Summarize:")
        layout.addWidget(input_label)

        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Paste your text here...")
        layout.addWidget(self.input_text)

        # Summarize Button
        summarize_button = QPushButton("Generate Summary")
        summarize_button.clicked.connect(self.generate_summary)
        layout.addWidget(summarize_button)

        # Output Text Area
        output_label = QLabel("Summary:")
        layout.addWidget(output_label)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        # Set layout
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Initialize summarizer
        self.summarizer = TextSummarizer()

    def generate_summary(self):
        # Get input text
        input_text = self.input_text.toPlainText().strip()

        # Validate input
        if not input_text:
            QMessageBox.warning(self, "Error", "Please enter some text to summarize.")
            return

        try:
            # Generate summary using the new 'summarize' method
            summary = self.summarizer.summarize(input_text)

            # Display summary
            self.output_text.setPlainText(summary)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")