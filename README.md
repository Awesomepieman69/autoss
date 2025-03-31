# vmauto - Automatic Screen Capture & Text Analysis

This application automatically captures screenshots from your screen at regular intervals, extracts text using OCR, and analyzes the text using OpenAI's GPT-4o model.

## Features

- Capture screenshots every 5 seconds
- Select specific regions of interest (ROI) on your screen
- Extract text using Tesseract OCR
- Process text through OpenAI GPT-4o
- Display captured images and AI responses in a simple GUI

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Pytesseract (and Tesseract OCR installed on your system)
- OpenAI API
- Python-dotenv
- Pillow
- Tkinter

## Installation

1. Clone this repository
2. Install Tesseract OCR:
   - macOS: `brew install tesseract`
   - Ubuntu: `sudo apt install tesseract-ocr`
   - Windows: Download and install from [GitHub Releases](https://github.com/UB-Mannheim/tesseract/wiki)

3. Install Python dependencies:
   ```
   pip install numpy opencv-python pytesseract openai python-dotenv pillow
   ```

4. Create a `.env` file in the project directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the application:
   ```
   python 4o.py
   ```

2. Select the monitor number to capture from
3. Click "Select ROI" to define the region of interest
4. Click "Start Capture" to begin automatic monitoring

The application will capture the selected region every 5 seconds, extract any text, and send it to OpenAI for analysis. Results are displayed in the GUI.

## License

Private repository - All rights reserved