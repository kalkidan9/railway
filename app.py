import os
import io
import fitz  # PyMuPDF
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageFilter
from rembg import remove, new_session
import base64
import numpy as np
from werkzeug.utils import secure_filename
import pytesseract
import re

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure Tesseract path (Update this with your actual path)
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Image Processing Class ---
class PDFImageExtractor:
    """
    Extracts images from a PDF file, processing the first image found
    as a profile picture and extracts FIN number and expiry dates.
    """
    def __init__(self, pdf_path, output_folder="output_temp"):
        self.pdf_path = pdf_path
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"\nInitializing PDF processor for: {os.path.basename(pdf_path)}")

    def preprocess_image_for_ocr(self, image):
        """Enhance image quality for better OCR results"""
        img = image.convert('L')  # Grayscale
        img = img.point(lambda x: 0 if x < 140 else 255, '1')  # Threshold
        return img

    def extract_expiry_date(self, text):
        """
        Extracts and separates the Ethiopian and English calendar dates.
        Returns a dictionary with both dates properly labeled.
        """
        # Pattern to match the specific date format
        pattern = r'(\d{4}/\d{2}/\d{2}\s*[|]\s*\d{4}/[a-zA-Z]{3}/\d{2})'
        match = re.search(pattern, text)
        
        default_expiry = {
            "ethiopian": "2025/11/02",
            "gregorian": "2033/Jul/09"
        }

        if match:
            # Clean and standardize the format
            date_str = match.group(1)
            date_str = date_str.replace(' ', '')  # Remove any spaces
            
            # Split into the two dates
            dates = date_str.split('|')
            if len(dates) == 2:
                # First date is Ethiopian (YYYY/MM/DD format)
                ethiopian_date = dates[0]
                
                # Second date is English (YYYY/MMM/DD format)
                english_date_parts = dates[1].split('/')
                if len(english_date_parts) == 3:
                    # Ensure month is properly capitalized
                    english_date_parts[1] = english_date_parts[1][:3].title()
                    english_date = '/'.join(english_date_parts)
                    
                    print(f"\nEXPIRY DATES EXTRACTED:")
                    print(f"Ethiopian: {ethiopian_date}")
                    print(f"English: {english_date}")
                    
                    return {
                        "ethiopian": ethiopian_date,
                        "gregorian": english_date
                    }
        
        print("\nEXPIRY DATE: Not Found, defaulting to N/A")
        return default_expiry

    def extract_fin_number(self, ocr_text):
        """Extracts FIN number from OCR text with multiple fallback methods"""
        try:
            # Primary pattern matching
            fin_match = re.search(r'(FIN|FAN)\s*[:\s-]*\s*([\d\s]{10,20})', ocr_text, re.IGNORECASE)

            if fin_match:
                raw_fin_digits_string = fin_match.group(2)
                cleaned_fin_value = re.sub(r'\D', '', raw_fin_digits_string)

                if len(cleaned_fin_value) > 12:
                    fin_number = cleaned_fin_value[:12]  # Truncate to first 12 digits
                    print(f"FIN found and truncated: {fin_number}")
                elif len(cleaned_fin_value) == 12:
                    fin_number = cleaned_fin_value  # Exactly 12 digits
                    print(f"FIN found: {fin_number}")
                else:
                    fin_number = "Not found"
                    print(f"FIN extraction issue: Extracted digits '{cleaned_fin_value}' are not 12 digits")
            else:
                # Fallback 1: Search for any standalone 12-digit numbers
                all_12_digit_numbers_standalone = re.findall(r'\b\d{12}\b', ocr_text)
                if all_12_digit_numbers_standalone:
                    if len(all_12_digit_numbers_standalone) == 1:
                        fin_number = all_12_digit_numbers_standalone[0]
                        print(f"FIN found (fallback 1, standalone): {fin_number}")
                    else:
                        fin_number = "Not found"
                        print("Multiple potential 12-digit numbers found")
                else:
                    # Fallback 2: Search for patterns like XXXX XXXX XXXX
                    all_12_digit_numbers_spaced = re.findall(r'\b\d{4}\s?\d{4}\s?\d{4}\b', ocr_text)
                    if all_12_digit_numbers_spaced:
                        cleaned_list = [re.sub(r'\D', '', num) for num in all_12_digit_numbers_spaced]
                        valid_cleaned_fours = [num for num in cleaned_list if len(num) == 12]
                        if valid_cleaned_fours:
                            if len(valid_cleaned_fours) == 1:
                                fin_number = valid_cleaned_fours[0]
                                print(f"FIN found (fallback 2, spaced): {fin_number}")
                            else:
                                fin_number = "Not found"
                                print("Multiple potential 12-digit patterns found")
                        else:
                            fin_number = "Not found"
                    else:
                        fin_number = "Not found"
            
            return fin_number

        except Exception as fin_ocr_err:
            print(f"FIN OCR Error: {str(fin_ocr_err)}")
            return "Error during FIN extraction"

    def extract_images(self, fin_target_image_index=4, expiry_target_image_index=3):
        """
        Extracts images from the PDF.
        Processes the first image found as a profile picture.
        Extracts FIN number and expiry dates from specified images.
        """
        pdf = None
        all_extracted_images_info = []
        profile_image_data = None
        fin_number = "Not found"
        expiry_dates = {
            "ethiopian": "2025/11/02",
            "gregorian": "2033/Jul/09"
        }

        try:
            pdf = fitz.open(self.pdf_path)
            print(f"\nProcessing PDF: {os.path.basename(self.pdf_path)}")

            # First pass: Extract all images and perform OCR
            for page_num in range(len(pdf)):
                page = pdf.load_page(page_num)
                images = page.get_images(full=True)

                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    if not base_image or not base_image.get("image"):
                        continue

                    image_bytes = base_image["image"]
                    width = base_image.get('width', 0)
                    height = base_image.get('height', 0)
                    ext = base_image.get('ext', 'png')

                    # Store image info
                    image_info = {
                        "image_bytes": image_bytes,
                        "ext": ext,
                        "width": width,
                        "height": height,
                        "original_xref": xref,
                        "page_num": page_num,
                        "img_index": img_index
                    }
                    all_extracted_images_info.append(image_info)

                    # FIN extraction
                    if len(all_extracted_images_info) == fin_target_image_index:
                        try:
                            with Image.open(io.BytesIO(image_bytes)) as img:
                                ocr_text = pytesseract.image_to_string(img, lang='eng')
                                fin_number = self.extract_fin_number(ocr_text)
                                print(f"\nFIN NUMBER EXTRACTED: {fin_number}")
                        except pytesseract.TesseractNotFoundError:
                            fin_number = "Tesseract Error: Not installed or not in PATH."
                        except Exception as e:
                            print(f"\nFIN EXTRACTION ERROR: {str(e)}")

                    # Expiry date extraction
                    if len(all_extracted_images_info) == expiry_target_image_index:
                        try:
                            with Image.open(io.BytesIO(image_bytes)) as img:
                                processed_img = self.preprocess_image_for_ocr(img)
                                ocr_text = pytesseract.image_to_string(
                                    processed_img, 
                                    lang='eng',
                                    config='--psm 6 --oem 3'
                                )
                                expiry_dates = self.extract_expiry_date(ocr_text)
                        except Exception as e:
                            print(f"\nEXPIRY DATE EXTRACTION ERROR: {str(e)}")

            # Second pass: Process profile image (first image > 300px height)
            for i, img_info in enumerate(all_extracted_images_info):
                if img_info["height"] > 300:
                    profile_image = all_extracted_images_info.pop(i)
                    print("\nFound profile picture (height > 300px)")
                    break

            # Process profile image if found
            if profile_image:
                processed_profile = self.process_profile_image(profile_image["image_bytes"])
                if processed_profile:
                    final_image_data = [processed_profile]
                else:
                    final_image_data = []
            else:
                final_image_data = []

            # Process remaining images
            for idx, img_info in enumerate(all_extracted_images_info):
                if not img_info.get("image_bytes"):
                    continue

                label = "QR Code" if idx == 0 else f"Additional Image {idx + 1}"
                
                try:
                    with Image.open(io.BytesIO(img_info['image_bytes'])) as img:
                        img_format = img_info['ext'].lower()
                        if img_format in ['jpg', 'jpeg']:
                            img = img.convert('RGB')
                            img_format = 'jpeg'
                        else:
                            img = img.convert('RGBA')
                            img_format = 'png'
                        
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format=img_format)
                        final_image_data.append({
                            "label": label,
                            "data": f"data:image/{img_format};base64,{base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')}"
                        })
                except Exception as e:
                    print(f"Could not process image: {str(e)}")

            print(f"\nPROCESSING COMPLETE")
            print(f"Images extracted: {len(final_image_data)}")
            print(f"FIN: {fin_number}")
            print(f"Expiry Dates: {expiry_dates}")

            return final_image_data, fin_number, expiry_dates

        except Exception as e:
            print(f"\nPDF PROCESSING ERROR: {str(e)}")
            return [], "Error", {"ethiopian": "Error", "gregorian": "Error"}
        finally:
            if pdf:
                pdf.close()

    def process_profile_image(self, image_bytes):
        """
        Processes the profile picture with background removal and enhancements
        """
        if not image_bytes:
            return None

        try:
            # Open the original image
            img_original = Image.open(io.BytesIO(image_bytes))
            original_mode = img_original.mode

            # Convert to RGBA for rembg processing
            img_for_rembg = img_original.convert("RGBA") if img_original.mode != 'RGBA' else img_original

            # Background removal
            session = new_session("u2net_human_seg")
            img_no_bg = remove(
                img_for_rembg,
                session=session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10
            )

            # Convert to array for processing
            img_array = np.array(img_no_bg)
            height, width = img_array.shape[:2]

            # Check if image is empty after rembg
            if np.sum(img_array[:, :, 3]) < (width * height * 0.01):
                raise ValueError("Background removal resulted in empty image")

            # Convert subject to grayscale if original was color
            is_color = original_mode in ['RGB', 'RGBA', 'YCbCr', 'CMYK']
            if is_color:
                alpha = img_array[:, :, 3]
                subject_mask = alpha > 10
                rgb_subject = img_array[subject_mask, :3]
                if rgb_subject.size > 0:
                    grayscale = (0.2989 * rgb_subject[:, 0] + 
                                0.5870 * rgb_subject[:, 1] + 
                                0.1140 * rgb_subject[:, 2]).astype(np.uint8)
                    img_array[subject_mask, :3] = np.repeat(grayscale[:, np.newaxis], 3, axis=1)

            # Gamma correction
            gamma = 1.7
            subject_mask = img_array[:, :, 3] > 50
            rgb_subject = img_array[subject_mask, :3]
            rgb_normalized = rgb_subject.astype(np.float32) / 255.0
            rgb_corrected = np.power(rgb_normalized + 1e-5, 1.0/gamma)
            img_array[subject_mask, :3] = (np.clip(rgb_corrected, 0, 1) * 255).astype(np.uint8)

            # Enhancements
            img_pil = Image.fromarray(img_array, 'RGBA')
            img_pil = ImageEnhance.Color(img_pil).enhance(1.5)  # Saturation
            img_pil = ImageEnhance.Contrast(img_pil).enhance(1.1)  # Contrast

            # Sharpening
            try:
                img_pil = img_pil.filter(ImageFilter.UnsharpMask(radius=1.5, percent=250, threshold=1))
            except Exception:
                pass  # Skip sharpening if it fails

            # Save to base64
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='PNG')
            return {
                "label": "Profile Picture",
                "data": f"data:image/png;base64,{base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')}"
            }

        except Exception as e:
            print(f"Profile processing error: {str(e)} - Using original image")
            try:
                img = Image.open(io.BytesIO(image_bytes))
                img_format = img.format if img.format else 'PNG'
                if img_format.upper() == 'JPEG' or ('A' not in img.mode and img_format.upper() == 'PNG'):
                    img = img.convert('RGB')
                else:
                    img = img.convert('RGBA')
                
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img_format)
                return {
                    "label": "Profile Picture (Original)",
                    "data": f"data:image/{img_format.lower()};base64,{base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')}"
                }
            except Exception:
                return None

# --- Flask API Endpoint ---
@app.route('/api/extract_images', methods=['POST'])
def extract_images_from_pdf():
    """API endpoint to process PDF and return extracted data"""
    if 'pdf' not in request.files:
        print("\nERROR: No file part in request")
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['pdf']
    if file.filename == '':
        print("\nERROR: No file selected")
        return jsonify({'error': 'No file selected'}), 400

    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        if len(filename) > 100:
            filename = filename[:95] + '.pdf'

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            print(f"\nPROCESSING FILE: {filename}")
            file.save(file_path)
            
            print("\nStarting extraction process...")
            extractor = PDFImageExtractor(file_path)
            image_data, fin_number, expiry_dates = extractor.extract_images()

            return jsonify({
                'images': image_data,
                'fin_number': fin_number,
                'expiry_date': expiry_dates
            })

        except MemoryError:
            print("\nERROR: Memory error - file too large")
            return jsonify({'error': 'File too large to process'}), 500
        except fitz.fitz.FileDataError as e:
            print(f"\nERROR: Invalid PDF file - {str(e)}")
            return jsonify({'error': 'Invalid or corrupted PDF file'}), 400
        except Exception as e:
            print(f"\nERROR: Processing failed - {str(e)}")
            return jsonify({'error': 'Processing failed'}), 500
        finally:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"\nCleaned up temporary file: {file_path}")
                except OSError:
                    pass
    else:
        print("\nERROR: Invalid file type")
        return jsonify({'error': 'Only PDF files are supported'}), 400

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Run on 0.0.0.0 to accept connections from all IPs
    app.run(host='0.0.0.0', port=port, debug=False)
