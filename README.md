# 🚀 Advanced OCR Flask Web Application

A beautiful, modern web application for extracting text from images using multiple OCR technologies including Tesseract, EasyOCR, and custom deep learning models.

## ✨ Features

- **Multiple OCR Engines**: Tesseract, EasyOCR, and custom CRNN model
- **Beautiful UI**: Modern, responsive design with drag-and-drop functionality
- **Advanced Preprocessing**: Image enhancement for better OCR accuracy
- **Real-time Processing**: Fast text extraction with loading animations
- **Copy to Clipboard**: Easy text copying functionality
- **Multiple Format Support**: JPG, PNG, GIF, BMP, TIFF

## 📁 Project Structure

```
ocr_flask_app/
├── app.py                 # Main Flask application
├── advanced_ocr_model.py  # Custom OCR model implementation
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main web interface
├── uploads/              # Temporary upload directory (auto-created)
├── models/               # Trained model storage
│   ├── emnist_cnn_model.keras
│   └── advanced_ocr_model.h5
└── static/               # Static files (if needed)
```

## 🛠️ Installation

### 1. System Dependencies

#### Windows
```bash
# Install Tesseract OCR
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
```

#### macOS
```bash
brew install tesseract
```

### 2. Python Environment

```bash
# Create virtual environment
python -m venv ocr_env

# Activate virtual environment
# Windows:
ocr_env\Scripts\activate
# Linux/macOS:
source ocr_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Create Project Files

Create the following directory structure and save the provided code files:

1. **app.py** - Main Flask application (provided above)
2. **templates/index.html** - Web interface (provided above)
3. **advanced_ocr_model.py** - Advanced OCR model (provided above)
4. **requirements.txt** - Dependencies (provided above)

## 🚀 Running the Application

### Basic Setup
```bash
# Navigate to project directory
cd ocr_flask_app

# Activate virtual environment
source ocr_env/bin/activate  # or ocr_env\Scripts\activate on Windows

# Run the Flask app
python app.py
```

### Advanced Setup with Custom Model Training

```bash
# First, train the advanced OCR model
python advanced_ocr_model.py

# Then run the Flask app
python app.py
```

The application will be available at `http://localhost:5000`

## 🎯 Usage

1. **Open your browser** and navigate to `http://localhost:5000`
2. **Upload an image** by clicking "Choose File" or dragging and dropping
3. **Wait for processing** - the app will use multiple OCR engines
4. **View results** from different OCR methods
5. **Copy text** using the copy buttons
6. **Process another image** using the reset button

## 🔧 Configuration

### Tesseract Configuration
You can modify Tesseract settings in `app.py`:

```python
# Custom Tesseract configuration
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
```

### Image Processing Parameters
Adjust preprocessing in the `preprocess_image_for_ocr` function:

```python
# Gaussian blur kernel size
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Adaptive threshold parameters
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY, 11, 2)
```

## 🧠 Advanced OCR Model

The custom CRNN (Convolutional Recurrent Neural Network) model provides:

- **CNN Feature Extraction**: Extracts visual features from text images
- **RNN Sequence Modeling**: Models sequential nature of text
- **CTC Loss**: Handles variable-length sequences without alignment
- **Attention Mechanism**: Focuses on relevant image regions

### Training Your Own Model

```python
from advanced_ocr_model import AdvancedOCRModel

# Initialize model
ocr_model = AdvancedOCRModel()

# Train with your data
history = ocr_model.train_model(train_images, train_labels, 
                               val_images, val_labels, epochs=50)

# Save trained model
ocr_model.model.save('custom_ocr_model.h5')
```

## 📊 Performance Tips

### For Better Accuracy:
1. **Image Quality**: Use high-resolution, clear images
2. **Preprocessing**: Adjust threshold and noise reduction parameters
3. **Multiple Engines**: Compare results from different OCR engines
4. **Custom Training**: Train on domain-specific text data

### For Better Speed:
1. **Image Resize**: Limit maximum image dimensions
2. **Model Optimization**: Use TensorFlow Lite for mobile deployment
3. **Caching**: Cache processed results for repeated requests

## 🎨 Customization

### UI Customization
Modify the CSS in `templates/index.html`:

```css
/* Change color scheme */
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --accent-color: #f093fb;
}
```

### Adding New OCR Engines
Extend the Flask app with additional OCR methods:

```python
def extract_text_new_engine(image_path):
    # Implement your OCR engine here
    return extracted_text

# Add to upload route
results['new_engine'] = extract_text_new_engine(filepath)
```

## 🚨 Troubleshooting

### Common Issues:

1. **Tesseract not found**
   ```bash
   # Add Tesseract to PATH or specify location
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

2. **EasyOCR model download fails**
   ```bash
   # Check internet connection and disk space
   # Models are ~500MB and downloaded on first use
   ```

3. **TensorFlow/Keras errors**
   ```bash
   # Ensure compatible versions
   pip install tensorflow==2.13.0 keras==2.13.1
   ```

4. **Memory issues with large images**
   ```python
   # Add image size limits in Flask config
   app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
   ```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

If you encounter any issues or have questions, please create an issue in the project repository.

---

**Happy Text Extracting! 🎉**