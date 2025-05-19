

from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np
import io
import base64
import re
import cv2  # OpenCV untuk pemrosesan gambar lebih cepat

# Initialize Flask app
app = Flask(__name__)

# Decode base64 image data
def decode_image(base64_data):
    # Extract the base64 content from data URL
    img_data = re.sub('^data:image/.+;base64,', '', base64_data)
    img_bytes = base64.b64decode(img_data)

    # Baca gambar dengan OpenCV (lebih cepat daripada PIL)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# Encode image to base64
def encode_image_to_base64(cv_image):
    # Konversi dari BGR ke RGB (CV2 menggunakan BGR)
    if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # Encode ke PNG
    success, buffer = cv2.imencode(".png", cv_image)
    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('ascii')
    return f"data:image/png;base64,{img_base64}"

# Process image with optimized algorithms
def process_image(image, filter_type, brightness, contrast, saturation):
    # Buat salinan untuk diproses
    processed = np.copy(image)

    # Terapkan filter yang dipilih menggunakan algoritma yang dioptimalkan
    if filter_type == 'blur':
        # Blur cepat dengan OpenCV
        processed = cv2.GaussianBlur(processed, (9, 9), 0)

    elif filter_type == 'edge':
        # Deteksi tepi cepat dengan OpenCV
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if len(processed.shape) == 3 else processed
        processed = cv2.Canny(gray, 100, 200)
        # Konversi kembali ke BGR jika input adalah gambar berwarna
        if len(image.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    elif filter_type == 'sharpen':
        # Sharpen cepat dengan kernel OpenCV
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed = cv2.filter2D(processed, -1, kernel)

    elif filter_type == 'emboss':
        # Emboss cepat dengan kernel OpenCV
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        processed = cv2.filter2D(processed, -1, kernel)

    elif filter_type == 'grayscale':
        # Grayscale cepat dengan OpenCV
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        # Konversi kembali ke BGR untuk konsistensi
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    elif filter_type == 'sepia':
        # Sepia cepat dengan matrix transformation
        sepia_kernel = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        # Pisahkan channels BGR dan transformasikan
        b, g, r = cv2.split(processed)
        b_new = np.clip(r * sepia_kernel[0, 0] + g * sepia_kernel[0, 1] + b * sepia_kernel[0, 2], 0, 255).astype(np.uint8)
        g_new = np.clip(r * sepia_kernel[1, 0] + g * sepia_kernel[1, 1] + b * sepia_kernel[1, 2], 0, 255).astype(np.uint8)
        r_new = np.clip(r * sepia_kernel[2, 0] + g * sepia_kernel[2, 1] + b * sepia_kernel[2, 2], 0, 255).astype(np.uint8)
        processed = cv2.merge([b_new, g_new, r_new])

    elif filter_type == 'invert':
        # Invert cepat dengan OpenCV
        processed = cv2.bitwise_not(processed)

    # Aplikasikan penyesuaian gambar: kecerahan, kontras, dan saturasi
    if brightness != 0:
        # Konversi dari -100 hingga 100 menjadi faktor kali
        brightness_factor = 1.0 + (float(brightness) / 100.0)
        processed = cv2.convertScaleAbs(processed, alpha=brightness_factor, beta=0)

    if contrast != 0:
        # Konversi dari -100 hingga 100 menjadi faktor kali
        contrast_factor = 1.0 + (float(contrast) / 100.0)
        mean = np.mean(processed)
        processed = cv2.convertScaleAbs(processed, alpha=contrast_factor, beta=(1.0 - contrast_factor) * mean)

    if saturation != 0 and len(processed.shape) == 3:
        # Konversi ke HSV untuk manipulasi saturasi
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        saturation_factor = 1.0 + (float(saturation) / 100.0)
        # Kalikan saturasi dengan faktor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255).astype(np.uint8)
        # Konversi kembali ke BGR
        processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return processed

# Main route serving the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint for image processing
@app.route('/process', methods=['POST'])
def process():
    try:
        # Get data from request
        data = request.json
        image_data = data.get('image')
        filter_type = data.get('filter', 'identity')
        brightness = int(data.get('brightness', 0))
        contrast = int(data.get('contrast', 0))
        saturation = int(data.get('saturation', 0))

        # Decode image from base64
        img = decode_image(image_data)

        # Skip processing completely if using identity filter with no adjustments
        if filter_type == 'identity' and brightness == 0 and contrast == 0 and saturation == 0:
            return jsonify({
                'processed_image': image_data  # Return original image
            })

        # Process the image with optimized algorithms
        processed_img = process_image(img, filter_type, brightness, contrast, saturation)

        # Encode back to base64
        processed_data = encode_image_to_base64(processed_img)

        # Return processed image
        return jsonify({
            'processed_image': processed_data
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

# Run app
if __name__ == '__main__':
    app.run(debug=True)
