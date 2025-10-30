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

# Try to import rembg with fallback
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
    print("✅ rembg successfully imported")
except ImportError as e:
    print(f"❌ rembg import failed: {e}")
    REMBG_AVAILABLE = False
    # Create dummy functions if rembg is not available
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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

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

# --- PDF Processing Class ---
class PDFImageExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        # Initialize rembg session only if available
        self.session = new_session("u2net") if REMBG_AVAILABLE else None

    def remove_background(self, image):
        """Remove background from image if rembg is available"""
        if not REMBG_AVAILABLE:
            return image
            
        try:
            print("🔄 Attempting background removal...")
            result = remove(
                image,
                session=self.session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10
            )
            print("✅ Background removal successful")
            return result
        except Exception as e:
            print(f"❌ Background removal failed: {e}")
            return image

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

    def process_profile_image(self, image_bytes):
        """Process profile image with background removal"""
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                # Only process images that look like profile pictures
                width, height = img.size
                
                # Check if image is portrait-oriented and reasonably sized
                if height > 200 and height > width:
                    print(f"🖼️ Processing profile image: {width}x{height}")
                    
                    # Remove background
                    img_no_bg = self.remove_background(img)
                    
                    # Convert to base64
                    img_byte_arr = io.BytesIO()
                    if img_no_bg.mode == 'RGBA':
                        img_no_bg.save(img_byte_arr, format='PNG')
                        img_format = 'png'
                    else:
                        img_no_bg.save(img_byte_arr, format='JPEG')
                        img_format = 'jpeg'
                    
                    return {
                        "label": "Profile Picture",
                        "data": f"data:image/{img_format};base64,{base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')}",
                        "processed": True
                    }
                
                # If not a profile picture, return original
                img_byte_arr = io.BytesIO()
                img_format = 'jpeg' if img.mode == 'RGB' else 'png'
                img.save(img_byte_arr, format=img_format)
                
                return {
                    "label": "Image",
                    "data": f"data:image/{img_format};base64,{base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')}",
                    "processed": False
                }
                    
        except Exception as e:
            print(f"❌ Profile processing error: {e}")
            # Fallback: return original image
            img_byte_arr = io.BytesIO()
            with Image.open(io.BytesIO(image_bytes)) as img:
                img_format = 'jpeg' if img.mode == 'RGB' else 'png'
                img.save(img_byte_arr, format=img_format)
            
            return {
                "label": "Image (Fallback)",
                "data": f"data:image/{img_format};base64,{base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')}",
                "processed": False
            }

    def extract_images(self):
        """Main extraction method"""
        pdf = None
        try:
            print(f"📖 Opening PDF: {self.pdf_path}")
            
            # Check if file exists and has content
            if not os.path.exists(self.pdf_path):
                raise Exception(f"PDF file not found: {self.pdf_path}")
            
            file_size = os.path.getsize(self.pdf_path)
            if file_size == 0:
                raise Exception("PDF file is empty")
                
            print(f"📄 PDF file size: {file_size} bytes")
            
            pdf = fitz.open(self.pdf_path)
            image_data = []
            fin_number = "Not found"
            expiry_dates = {"ethiopian": "Not found", "gregorian": "Not found"}
            profile_image_processed = False

            print(f"📑 PDF has {len(pdf)} pages")
            
            for page_num in range(len(pdf)):
                page = pdf.load_page(page_num)
                images = page.get_images(full=True)
                print(f"🖼️ Page {page_num + 1}: {len(images)} images found")

                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    
                    if base_image and base_image.get("image"):
                        image_bytes = base_image["image"]
                        print(f"  📸 Image {img_index + 1}: {len(image_bytes)} bytes")
                        
                        # Check if this might be a profile picture (first large image)
                        if not profile_image_processed and len(image_bytes) > 10000:  # > 10KB
                            print(f"🎯 Processing potential profile image {len(image_data) + 1}")
                            processed_image = self.process_profile_image(image_bytes)
                            image_data.append(processed_image)
                            profile_image_processed = processed_image.get("processed", False)
                        else:
                            # Regular image processing
                            try:
                                with Image.open(io.BytesIO(image_bytes)) as img_pil:
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
                                        "data": f"data:image/{img_format};base64,{base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')}",
                                        "processed": False
                                    })

                            except Exception as e:
                                print(f"⚠️ Image processing error: {e}")
                                continue

                        # OCR on first image for data extraction
                        if len(image_data) == 1:
                            try:
                                with Image.open(io.BytesIO(image_bytes)) as img_pil:
                                    print("🔍 Performing OCR on first image...")
                                    ocr_text = pytesseract.image_to_string(img_pil, lang='eng')
                                    print(f"📝 OCR text length: {len(ocr_text)} characters")
                                    fin_number = self.extract_fin_number(ocr_text)
                                    expiry_dates = self.extract_expiry_date(ocr_text)
                                    print(f"📊 Extracted FIN: {fin_number}")
                            except Exception as e:
                                print(f"⚠️ OCR error: {e}")

            print(f"✅ Extraction complete: {len(image_data)} images processed")
            return image_data, fin_number, expiry_dates

        except Exception as e:
            print(f"❌ PDF processing error: {e}")
            return [], f"Error: {str(e)}", {"ethiopian": "Error", "gregorian": "Error"}
        finally:
            if pdf:
                pdf.close()

# --- API Endpoint ---
@app.route('/api/extract_images', methods=['POST'])
def extract_images_from_pdf():
    print("🔍 API endpoint called - checking request...")
    print(f"📨 Request method: {request.method}")
    print(f"📦 Request files: {list(request.files.keys())}")
    
    if 'pdf' not in request.files:
        print("❌ No 'pdf' in request.files")
        return jsonify({'error': 'No PDF file provided'}), 400
    
    file = request.files['pdf']
    print(f"📄 File received: {file.filename}")
    print(f"📏 File content length: {file.content_length}")
    print(f"📋 File content type: {file.content_type}")
    
    if file.filename == '':
        print("❌ Empty filename")
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.lower().endswith('.pdf'):
        print(f"❌ Not a PDF file: {file.filename}")
        return jsonify({'error': 'Only PDF files are supported'}), 400

    file_path = None
    try:
        # Save file temporarily with better error handling
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"💾 Saving file to: {file_path}")
        
        # Read file content first to check if it's empty
        file_data = file.read()
        if len(file_data) == 0:
            raise Exception("Uploaded file is empty")
            
        # Write to file
        with open(file_path, 'wb') as f:
            f.write(file_data)
        
        print(f"✅ File saved successfully: {os.path.getsize(file_path)} bytes")
        
        # Process PDF
        print("🔄 Starting PDF processing...")
        extractor = PDFImageExtractor(file_path)
        images, fin_number, expiry_dates = extractor.extract_images()
        
        print(f"🎉 Processing complete: {len(images)} images, FIN: {fin_number}")
        
        return jsonify({
            'images': images,
            'fin_number': fin_number,
            'expiry_date': expiry_dates,
            'message': 'Processing complete',
            'rembg_available': REMBG_AVAILABLE,
            'images_count': len(images)
        })
        
    except Exception as e:
        print(f"❌ Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    finally:
        # Clean up
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"🧹 Cleaned up temporary file: {file_path}")
            except Exception as e:
                print(f"⚠️ Cleanup error: {e}")


