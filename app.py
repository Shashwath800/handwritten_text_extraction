from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import cv2
import numpy as np
import easyocr
import pytesseract
from PIL import Image
import base64
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.models import load_model
import re
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Check if Tesseract is available
def check_tesseract():
    try:
        # Try to find tesseract executable
        tesseract_path = shutil.which("tesseract")
        if tesseract_path:
            return True

        # Try common Windows installation paths
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', ''))
        ]

        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                return True

        return False
    except:
        return False


TESSERACT_AVAILABLE = check_tesseract()

# Initialize OCR readers
try:
    easyocr_reader = easyocr.Reader(['en'])
    EASYOCR_AVAILABLE = True
except:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not available")

# Try to load your trained model
try:
    custom_model = load_model('emnist_cnn_model.keras')
    CUSTOM_MODEL_AVAILABLE = True
    # EMNIST character mapping (you'll need to adjust this based on your model)
    emnist_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
except:
    CUSTOM_MODEL_AVAILABLE = False
    print("Custom EMNIST model not available")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}


def preprocess_image_for_ocr(image_path):
    """Enhanced image preprocessing for better OCR results"""
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Morphological operations to clean up the image
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Resize image if too small
    height, width = cleaned.shape
    if height < 300 or width < 300:
        scale_factor = max(300 / height, 300 / width)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        cleaned = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    return cleaned


def extract_text_pytesseract(image_path):
    """Extract text using Tesseract with enhanced preprocessing"""
    try:
        if not TESSERACT_AVAILABLE:
            return "Tesseract is not installed. Please install Tesseract OCR:\n1. Download from: https://github.com/UB-Mannheim/tesseract/wiki\n2. Install and add to PATH\n3. Restart your application"

        processed_img = preprocess_image_for_ocr(image_path)

        # Custom Tesseract configuration
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '

        text = pytesseract.image_to_string(processed_img, config=custom_config)
        return text.strip()
    except Exception as e:
        return f"Tesseract Error: {str(e)}\n\nTo fix this:\n1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki\n2. Install it\n3. Add to your system PATH\n4. Restart the application"


def extract_text_easyocr(image_path):
    """Extract text using EasyOCR"""
    try:
        if not EASYOCR_AVAILABLE:
            return "EasyOCR not available. Install with: pip install easyocr"

        results = easyocr_reader.readtext(image_path)
        text_list = [result[1] for result in results if result[2] > 0.5]  # Confidence > 0.5
        return ' '.join(text_list)
    except Exception as e:
        return f"EasyOCR Error: {str(e)}"


def extract_text_custom_model(image_path):
    """Extract text using your custom EMNIST model"""
    try:
        if not CUSTOM_MODEL_AVAILABLE:
            return "Custom model not available. Make sure 'emnist_cnn_model.keras' exists in the project directory."

        # Load and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Segment characters (this is a simplified approach)
        # In practice, you'd need more sophisticated character segmentation
        chars = segment_characters(img)

        predictions = []
        for char_img in chars:
            # Resize to 28x28 for EMNIST
            char_img = cv2.resize(char_img, (28, 28))
            char_img = char_img.reshape(1, 28, 28, 1) / 255.0

            pred = custom_model.predict(char_img, verbose=0)
            pred_idx = np.argmax(pred)
            if pred_idx < len(emnist_chars):
                predictions.append(emnist_chars[pred_idx])

        return ''.join(predictions)
    except Exception as e:
        return f"Custom Model Error: {str(e)}"


def segment_characters(img):
    """Simple character segmentation (you might want to improve this)"""
    # Apply thresholding
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # Filter small noise
            char_img = img[y:y + h, x:x + w]
            chars.append(char_img)

    return chars


def image_to_base64(image_path):
    """Convert image to base64 for display in HTML"""
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        return img_base64


@app.route('/')
def index():
    return render_template('front.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract text using multiple methods
        results = {}

        # Tesseract OCR
        tesseract_text = extract_text_pytesseract(filepath)
        results['tesseract'] = tesseract_text

        # EasyOCR
        easyocr_text = extract_text_easyocr(filepath)
        results['easyocr'] = easyocr_text

        # Custom Model
        custom_text = extract_text_custom_model(filepath)
        results['custom'] = custom_text

        # Convert image to base64 for display
        img_base64 = image_to_base64(filepath)

        # Clean up uploaded file
        os.remove(filepath)

        return jsonify({
            'success': True,
            'results': results,
            'image': img_base64,
            'filename': filename
        })

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/status')
def status():
    """Check the status of all OCR engines"""
    return jsonify({
        'tesseract': TESSERACT_AVAILABLE,
        'easyocr': EASYOCR_AVAILABLE,
        'custom_model': CUSTOM_MODEL_AVAILABLE
    })


if __name__ == '__main__':
    print("=" * 50)
    print("OCR Status:")
    print(f"Tesseract: {'✓ Available' if TESSERACT_AVAILABLE else '✗ Not Available'}")
    print(f"EasyOCR: {'✓ Available' if EASYOCR_AVAILABLE else '✗ Not Available'}")
    print(f"Custom Model: {'✓ Available' if CUSTOM_MODEL_AVAILABLE else '✗ Not Available'}")
    print("=" * 50)

    port = int(os.environ.get("PORT", 5000))  # Render provides PORT automatically
    app.run(host="0.0.0.0", port=port)
