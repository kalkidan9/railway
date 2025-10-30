import os
import io
import fitz  # PyMuPDF
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import base64
import numpy as np
from werkzeug.utils import secure_filename
import pytesseract
import re

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = '/tmp/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Tesseract for Railway
try:
    pytesseract.get_tesseract_version()
except:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# --- Health Check ---
@app.route('/')
def home():
    return jsonify({
        'message': 'Ethiopian ID Processor API', 
        'status': 'running',
        'version': '1.0'
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

# --- PDF Processing Class ---
class PDFImageExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_fin_number(self, ocr_text):
        """Extract FIN number from OCR text"""
        try:
            # Multiple patterns to try
            patterns = [
                r'(FIN|FAN)[\s:\-]*([0-9]{4}\s?[0-9]{4}\s?[0-9]{4})',
                r'\b([0-9]{4}\s?[0-9]{4}\s?[0-9]{4})\b',
                r'([0-9]{12})'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, ocr_text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        fin_candidate = match[1]
                    else:
                        fin_candidate = match
                    
                    cleaned_fin = re.sub(r'\D', '', fin_candidate)
                    if len(cleaned_fin) == 12:
                        return f"{cleaned_fin[:4]} {cleaned_fin[4:8]} {cleaned_fin[8:12]}"
            
            return "Not found"
        except Exception as e:
            return f"Error: {str(e)}"

    def extract_expiry_date(self, text):
        """Extract expiry dates"""
        pattern = r'(\d{4}/\d{2}/\d{2})\s*[|]\s*(\d{4}/[a-zA-Z]{3}/\d{2})'
        match = re.search(pattern, text)
        
        if match:
            return {
                "ethiopian": match.group(1),
                "gregorian": match.group(2)
            }
        
        return {
            "ethiopian": "Not found",
            "gregorian": "Not found"
        }

    def extract_images(self):
        """Main extraction method"""
        pdf = None
        try:
            pdf = fitz.open(self.pdf_path)
            image_data = []
            fin_number = "Not found"
            expiry_dates = {"ethiopian": "Not found", "gregorian": "Not found"}

            for page_num in range(len(pdf)):
                page = pdf.load_page(page_num)
                images = page.get_images(full=True)

                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    
                    if base_image and base_image.get("image"):
                        image_bytes = base_image["image"]
                        
                        # Convert to base64
                        try:
                            with Image.open(io.BytesIO(image_bytes)) as img_pil:
                                # Determine format
                                ext = base_image.get('ext', 'png').lower()
                                if ext in ['jpg', 'jpeg']:
                                    img_pil = img_pil.convert('RGB')
                                    img_format = 'jpeg'
                                else:
                                    img_format = 'png'
                                
                                img_byte_arr = io.BytesIO()
                                img_pil.save(img_byte_arr, format=img_format)
                                
                                image_data.append({
                                    "label": f"Image {len(image_data) + 1}",
                                    "data": f"data:image/{img_format};base64,{base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')}"
                                })

                                # OCR on first image for data extraction
                                if len(image_data) == 1:
                                    ocr_text = pytesseract.image_to_string(img_pil, lang='eng')
                                    fin_number = self.extract_fin_number(ocr_text)
                                    expiry_dates = self.extract_expiry_date(ocr_text)
                                    
                        except Exception as e:
                            print(f"Image processing error: {e}")
                            continue

            return image_data, fin_number, expiry_dates

        except Exception as e:
            print(f"PDF processing error: {e}")
            return [], "Error processing PDF", {"ethiopian": "Error", "gregorian": "Error"}
        finally:
            if pdf:
                pdf.close()

# --- API Endpoint ---
@app.route('/api/extract_images', methods=['POST'])
def extract_images_from_pdf():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file provided'}), 400
    
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400

    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process PDF
        extractor = PDFImageExtractor(file_path)
        images, fin_number, expiry_dates = extractor.extract_images()
        
        return jsonify({
            'images': images,
            'fin_number': fin_number,
            'expiry_date': expiry_dates,
            'message': 'Processing complete'
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    finally:
        # Clean up
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

if __name__ == '__main__':
    # Railway provides PORT environment variable - use it!
    port = int(os.environ.get('PORT', 8080))  # ⬅️ CHANGED to 8080 default
    print(f"🚀 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
