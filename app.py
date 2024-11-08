from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "secret_key"  # Ganti dengan secret key Anda

# Paths to save temporary images (ubah ke /tmp untuk Vercel)
TEMP_DIR = '/tmp/images'
UPLOAD_FOLDER = '/tmp/uploads'

# Pastikan folder sementara ada
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def is_clean(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return False, "Gambar tidak ditemukan atau gagal dimuat."

    # Step 1: Konversi gambar ke grayscale untuk menghilangkan informasi warna (PreProcessing)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Gaussian Blur untuk mengurangi noise pada gambar grayscale (PreProcessing)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 3: Thresholding Otsu untuk memisahkan objek dari latar belakang (Segmentasi Citra)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 4: Operasi morfologi (Opening) untuk menghilangkan noise kecil yang tersisa (Segmentasi Citra)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Menyimpan gambar hasil setiap langkah pengolahan untuk ditampilkan di HTML
    steps = {
        'original_image': image,
        'gray_image': gray,
        'blurred_image': blurred,
        'threshold_image': thresh,
        'morph_image': opened,
    }

    # Menyimpan setiap gambar hasil pengolahan di folder TEMP_DIR sebagai .png agar dapat ditampilkan di HTML
    for step, img in steps.items():
        cv2.imwrite(f"{TEMP_DIR}/{step}.png", img)

    # Mendeteksi kontur dan melakukan pengecekan kebersihan
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area, max_area = 0, 0  # Inisialisasi total area dan area maksimum

    if not contours:
        return True, "Wadah bersih, proses selesai."

    # Menghitung area setiap kontur yang terdeteksi
    for contour in contours:
        area = cv2.contourArea(contour)  # Menghitung luas area kontur
        total_area += area  # Menambah area kontur ke total area
        if area > max_area:
            max_area = area  # Mencari area kontur terbesar
        if area > 300 and area < 5000:
            return False, f"Kontur kecil terdeteksi dengan area {area}. Wadah kotor, proses dihentikan."

    # Mengecek jika total area kontur kecil (total area - max_area) melebihi batas
    if total_area - max_area > 1200:
        return False, f"Total area kontur kecil {total_area - max_area}. Wadah kotor, proses dihentikan."

    # Jika lolos semua pengecekan, anggap wadah bersih dan proses bisa dilanjutkan
    return True, "Wadah bersih, proses bisa dilanjutkan."

def identify_fill_level(red_area, total_area):
    """Determine fill level based on red area's proportion of the bottle area."""
    fill_ratio = red_area / total_area

    if fill_ratio < 0.05:  # Almost no red area (empty)
        return 'Kosong'
    elif fill_ratio < 0.25:  # Small red area (1/4 filled)
        return '1/4'
    elif fill_ratio < 0.5:  # Moderate red area (1/2 filled)
        return '1/2'
    else:  # Large red area (full)
        return 'Penuh'

def segment_and_identify(image_path):
    # Load and resize the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (90, 204))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Otsu Thresholding
    _, otsu_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(TEMP_DIR, 'otsu_thresh.png'), otsu_thresh)

    # Step 2: Canny Edge Detection
    edges = cv2.Canny(otsu_thresh, 50, 150)
    cv2.imwrite(os.path.join(TEMP_DIR, 'edges.png'), edges)

    # Step 3: Watershed Segmentation
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    cv2.imwrite(os.path.join(TEMP_DIR, 'watershed.png'), image)

    # Step 4: Detect Red Liquid Area Using HSV Masking
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Combine masks for red color
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Calculate the red area and total area within the bottle
    red_area = np.sum(red_mask == 255)
    total_area = 90 * 204  # Approximate area based on the resized image size

    # Determine fill level
    fill_level = identify_fill_level(red_area, total_area)

    # Save the final prediction image
    final_prediction_path = os.path.join(TEMP_DIR, 'final_prediction.png')
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.title('Otsu Thresholding')
    plt.imshow(otsu_thresh, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('Canny Edge Detection')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title('Watershed Segmentation')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title(f'Final Prediction: {fill_level}')
    plt.imshow(red_mask, cmap='gray')
    plt.axis('off')
    plt.savefig(final_prediction_path)
    plt.close()

    return fill_level

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filepath = os.path.join(TEMP_DIR, 'uploaded_image.jpg')
            file.save(filepath)

            clean_status, message = is_clean(filepath)

            if clean_status:  # If clean, show fill level check
                return render_template(
                    'result_clean.html',
                    message=message,
                    is_clean=clean_status,
                    original_image='images/original_image.png',
                    gray_image='images/gray_image.png',
                    blurred_image='images/blurred_image.png',
                    threshold_image='images/threshold_image.png',
                    morph_image='images/morph_image.png'
                )
            else:
                return render_template('index.html', message=message)

    return render_template('index.html')

@app.route('/check_fill_level', methods=['GET', 'POST'])
def check_fill_level():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith(('jpg', 'jpeg', 'png')):  # Valid image formats
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            prediction = segment_and_identify(file_path)

            return render_template('result_fill_level.html', prediction=prediction)

    return redirect(url_for('upload_file'))

if __name__ == '__main__':
    app.run()