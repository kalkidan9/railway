import os
import io
import gc
import fitz  # PyMuPDF
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import base64
import numpy as np
from werkzeug.utils import secure_filename
import pytesseract
import re
import traceback

# Try to import rembg with fallback
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
    print("✅ rembg successfully imported")
except ImportError as e:
    print(f"❌ rembg import failed: {e}")
    REMBG_AVAILABLE = False
    def remove(image, session=None, **kwargs):
        return image
    def new_session(*args, **kwargs):
        return None

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = '/tmp/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # Reduced to 8MB for Railway

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Tesseract for Railway
try:
    pytesseract.get_tesseract_version()
except:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

print(f"🔧 rembg available: {REMBG_AVAILABLE}")

# --- Health Check ---
@app.route('/')
def home():
    return jsonify({
        'message': 'Ethiopian ID Processor API', 
        'status': 'running',
        'version': '1.0',
        'rembg_available': REMBG_AVAILABLE
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'rembg_available': REMBG_AVAILABLE})

# --- Lightweight PDF Processing Class ---
class SimplePDFExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_fin_number_fast(self, ocr_text):
        """Fast FIN extraction without complex regex"""
        try:
            # Simple pattern matching
            if 'FIN' in ocr_text.upper():
                # Look for numbers near FIN
                lines = ocr_text.split('\n')
                for line in lines:
                    if 'FIN' in line.upper():
                        numbers = re.sub(r'\D', '', line)
                        if len(numbers) >= 12:
                            return numbers[:12]
            
            # Fallback: look for any 12-digit number
            numbers = re.findall(r'\d{12}', ocr_text)
            if numbers:
                return numbers[0]
                
            return "Not found"
        except:
            return "Not found"

    def extract_expiry_date_fast(self, text):
        """Fast expiry date extraction"""
        try:
            # Simple pattern
            dates = re.findall(r'\d{4}/\d{2}/\d{2}', text)
            if len(dates) >= 2:
                return {
                    "ethiopian": dates[0],
                    "gregorian": dates[1] if len(dates) > 1 else "Not found"
                }
        except:
            pass
        return {"ethiopian": "Not found", "gregorian": "Not found"}

    def extract_images_safe(self):
        """Safe extraction with memory management"""
        pdf = None
        try:
            print(f"📖 Opening PDF: {os.path.basename(self.pdf_path)}")
            pdf = fitz.open(self.pdf_path)
            
            image_data = []
            fin_number = "Not found"
            expiry_dates = {"ethiopian": "Not found", "gregorian": "Not found"}
            
            # Process only first page to save memory
            for page_num in range(min(1, len(pdf))):
                page = pdf.load_page(page_num)
                images = page.get_images(full=True)

                for img_index, img in enumerate(images[:3]):  # Limit to 3 images
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    
                    if base_image and base_image.get("image"):
                        image_bytes = base_image["image"]
                        
                        try:
                            with Image.open(io.BytesIO(image_bytes)) as img_pil:
                                # Convert to base64
                                ext = base_image.get('ext', 'png').lower()
                                if ext in ['jpg', 'jpeg']:
                                    img_pil = img_pil.convert('RGB')
                                    img_format = 'jpeg'
                                else:
                                    img_format = 'png'
                                
                                img_byte_arr = io.BytesIO()
                                img_pil.save(img_byte_arr, format=img_format, optimize=True)
                                
                                image_data.append({
                                    "label": f"Image {len(image_data) + 1}",
                                    "data": f"data:image/{img_format};base64,{base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')}"
                                })

                                # Quick OCR on first image only
                                if len(image_data) == 1:
                                    ocr_text = pytesseract.image_to_string(
                                        img_pil, 
                                        lang='eng',
                                        config='--psm 6 --oem 1'
                                    )
                                    fin_number = self.extract_fin_number_fast(ocr_text)
                                    expiry_dates = self.extract_expiry_date_fast(ocr_text)
                                    
                        except Exception as e:
                            print(f"⚠️ Image error: {e}")
                            continue

            print(f"✅ Extracted {len(image_data)} images")
            return image_data, fin_number, expiry_dates

        except Exception as e:
            print(f"❌ PDF error: {e}")
            return [], f"Error: {str(e)}", {"ethiopian": "Error", "gregorian": "Error"}
        finally:
            if pdf:
                pdf.close()
            # Force garbage collection
            gc.collect()

# --- Simple API Endpoint ---
@app.route('/api/extract_images', methods=['POST'])
def extract_images_from_pdf():
    print("🔍 API called")
    
    if 'pdf' not in request.files:
        print("❌ No pdf in files")
        return jsonify({'error': 'No PDF file provided'}), 400
    
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400

    file_path = None
    try:
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file_data = file.read()
        if len(file_data) == 0:
            return jsonify({'error': 'Uploaded file is empty'}), 400
            
        with open(file_path, 'wb') as f:
            f.write(file_data)
        
        print(f"✅ File saved: {len(file_data)} bytes")
        
        # Process with simple extractor
        extractor = SimplePDFExtractor(file_path)
        images, fin_number, expiry_dates = extractor.extract_images_safe()
        
        return jsonify({
            'images': images,
            'fin_number': fin_number,
            'expiry_date': expiry_dates,
            'message': 'Processing complete'
        })
        
    except Exception as e:
        print(f"❌ Processing failed: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    finally:
        # Clean up
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        gc.collect()

# No app.run() here - Railway uses Gunicorn
