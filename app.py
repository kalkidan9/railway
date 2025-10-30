import os
import io
import fitz  # PyMuPDF
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageFilter
import base64
import numpy as np
from werkzeug.utils import secure_filename
import pytesseract
import re

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Configure for Railway
UPLOAD_FOLDER = '/tmp/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure Tesseract for Railway
try:
    pytesseract.get_tesseract_version()
except:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# --- Health Check Endpoints ---
@app.route('/')
def home():
    return jsonify({
        'message': 'Ethiopian ID Processor API', 
        'status': 'running'
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

# --- PDF Image Extractor Class (Without rembg) ---
class PDFImageExtractor:
    def __init__(self, pdf_path, output_folder="/tmp/output"):
        self.pdf_path = pdf_path
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def preprocess_image_for_ocr(self, image):
        """Enhance image quality for better OCR results"""
        img = image.convert('L')  # Grayscale
        img = img.point(lambda x: 0 if x < 140 else 255, '1')  # Threshold
        return img

    def extract_expiry_date(self, text):
        """Extract expiry dates from OCR text"""
        pattern = r'(\d{4}/\d{2}/\d{2}\s*[|]\s*\d{4}/[a-zA-Z]{3}/\d{2})'
        match = re.search(pattern, text)
        
        default_expiry = {
            "ethiopian": "2025/11/02",
            "gregorian": "2033/Jul/09"
        }

        if match:
            date_str = match.group(1).replace(' ', '')
            dates = date_str.split('|')
            if len(dates) == 2:
                ethiopian_date = dates[0]
                english_date_parts = dates[1].split('/')
                if len(english_date_parts) == 3:
                    english_date_parts[1] = english_date_parts[1][:3].title()
                    english_date = '/'.join(english_date_parts)
                    
                    return {
                        "ethiopian": ethiopian_date,
                        "gregorian": english_date
                    }
        
        return default_expiry

    def extract_fin_number(self, ocr_text):
        """Extract FIN number from OCR text"""
        try:
            fin_match = re.search(r'(FIN|FAN)\s*[:\s-]*\s*([\d\s]{10,20})', ocr_text, re.IGNORECASE)

            if fin_match:
                raw_fin_digits_string = fin_match.group(2)
                cleaned_fin_value = re.sub(r'\D', '', raw_fin_digits_string)

                if len(cleaned_fin_value) >= 12:
                    fin_number = cleaned_fin_value[:12]
                    return fin_number
            
            # Fallback: look for any 12-digit number
            all_12_digit = re.findall(r'\b\d{12}\b', ocr_text)
            if all_12_digit and len(all_12_digit) == 1:
                return all_12_digit[0]
                
            return "Not found"
            
        except Exception as e:
            print(f"FIN extraction error: {str(e)}")
            return "Error during FIN extraction"

    def extract_images(self):
        """Extract images and data from PDF"""
        pdf = None
        image_data = []
        fin_number = "Not found"
        expiry_dates = {
            "ethiopian": "2025/11/02",
            "gregorian": "2033/Jul/09"
        }

        try:
            pdf = fitz.open(self.pdf_path)
            print(f"Processing PDF: {os.path.basename(self.pdf_path)}")

            # Extract images and perform OCR
            for page_num in range(len(pdf)):
                page = pdf.load_page(page_num)
                images = page.get_images(full=True)

                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    if not base_image or not base_image.get("image"):
                        continue

                    image_bytes = base_image["image"]
                    ext = base_image.get('ext', 'png')

                    # Convert to base64 for response
                    try:
                        with Image.open(io.BytesIO(image_bytes)) as img_pil:
                            img_format = 'jpeg' if ext.lower() in ['jpg', 'jpeg'] else 'png'
                            if img_format == 'jpeg':
                                img_pil = img_pil.convert('RGB')
                            
                            img_byte_arr = io.BytesIO()
                            img_pil.save(img_byte_arr, format=img_format)
                            
                            image_data.append({
                                "label": f"Image {len(image_data) + 1}",
                                "data": f"data:image/{img_format};base64,{base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')}"
                            })

                        # Try OCR on first few images for FIN and expiry
                        if len(image_data) <= 3:
                            with Image.open(io.BytesIO(image_bytes)) as img_ocr:
                                processed_img = self.preprocess_image_for_ocr(img_ocr)
                                ocr_text = pytesseract.image_to_string(
                                    processed_img, 
                                    lang='eng',
                                    config='--psm 6 --oem 3'
                                )
                                
                                if fin_number == "Not found":
                                    fin_number = self.extract_fin_number(ocr_text)
                                
                                if expiry_dates["ethiopian"] == "2025/11/02":
                                    expiry_dates = self.extract_expiry_date(ocr_text)

                    except Exception as e:
                        print(f"Image processing error: {str(e)}")
                        continue

            return image_data, fin_number, expiry_dates

        except Exception as e:
            print(f"PDF processing error: {str(e)}")
            return [], "Error", {"ethiopian": "Error", "gregorian": "Error"}
        finally:
            if pdf:
                pdf.close()

# --- API Endpoint ---
@app.route('/api/extract_images', methods=['POST'])
def extract_images_from_pdf():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(file_path)
            extractor = PDFImageExtractor(file_path)
            image_data, fin_number, expiry_dates = extractor.extract_images()

            return jsonify({
                'images': image_data,
                'fin_number': fin_number,
                'expiry_date': expiry_dates
            })

        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
        finally:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
    else:
        return jsonify({'error': 'Only PDF files are supported'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
