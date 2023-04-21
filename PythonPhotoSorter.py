import cv2
import os
import imageio.v2 as imageio 
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import shutil
from mtcnn import MTCNN
import yaml
import pywt
import queue
import threading

image_queue = queue.Queue()

def visualize_image(title, image, width):
    if image.shape[1] > width:
        aspect_ratio = float(image.shape[0]) / float(image.shape[1])
        new_width = width
        new_height = int(new_width * aspect_ratio)
        image = cv2.resize(image, (new_width, new_height))

    title += " (press Q to go to the next photo)"

    image_queue.put((title, image))

def gui_thread():
    while True:
        title, image = image_queue.get()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        cv2.putText(image, title, (20, 40), font, fontScale, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) 
        cv2.resizeWindow(title, image.shape[1], image.shape[0]) 
        while True:
            cv2.imshow(title, image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

gui_thread = threading.Thread(target=gui_thread)
gui_thread.daemon = True
gui_thread.start()

def detect_faces(image, visualize_mode, visualize_image_size, detector):
    min_confidence = 0.6
    min_face_size = 100
    
    faces = detector.detect_faces(image)

    faces = [face for face in faces if face['confidence'] >= min_confidence and face['box'][2] >= min_face_size and face['box'][3] >= min_face_size]

    faces = sorted(faces, key=lambda x: x['box'][2]*x['box'][3], reverse=True)

    if visualize_mode == 1 and len(faces) > 0:
        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"Confidence: {face['confidence']:.2f}\nBox: {face['box']}\nKeypoints: {face['keypoints']}"
            cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        visualize_image("Detected Faces", image, visualize_image_size)

    return faces

def detect_blur_on_faces(image, visualize_mode, visualize_image_size, detector):
    faces = detect_faces(image, visualize_mode, visualize_image_size, detector)
    if len(faces) == 0:
        return None

    max_blur = -1
    for face in faces:
        x, y, w, h = face['box']
        face_image = image[y:y+h, x:x+w]
        blur = detect_blur(face_image)
        if blur > max_blur:
            max_blur = blur

        if visualize_mode == 2:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, f"Blur: {blur:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if visualize_mode == 2:
        visualize_image("Faces and Blur", image, visualize_image_size)

    return max_blur

def estimate_noise(image):
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_image)

    y_noise = wavelet_mad(y)
    cr_noise = wavelet_mad(cr)
    cb_noise = wavelet_mad(cb)

    wavelet_noise = np.mean([y_noise, cr_noise, cb_noise])

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

    laplacian_noise = laplacian.std()

    weight = 0.5 
    noise_level = (1 - weight) * wavelet_noise + weight * laplacian_noise

    return noise_level * 10

def wavelet_mad(image_channel):
    coeffs = pywt.wavedec2(image_channel, 'db8', level=4, mode='per')

    detail_coeffs = coeffs[1:]
    mad_values = [np.median(np.abs(detail_coeff)) / 0.6745 for detail_coeff in detail_coeffs]

    return np.max(mad_values)

def detect_blur(gray_image):
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_norm = (sobel / np.max(sobel) * 255).astype(np.uint8)
    hist = cv2.calcHist([sobel_norm], [0], None, [10], [0, 256])
    hist = hist.flatten() / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return entropy * 1000

def estimate_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    L, A, B = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)
    
    kernel = np.ones((5, 5), np.uint8)
    min = cv2.erode(L_clahe, kernel, iterations=1)
    max = cv2.dilate(L_clahe, kernel, iterations=1)
    
    min = min.astype(np.float64)
    max = max.astype(np.float64)
    
    epsilon = 1e-6
    contrast = (max - min) / (max + min + epsilon)
    
    hist = cv2.calcHist([L_clahe], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    mean_brightness = np.dot(np.arange(256), hist)
    contrast_from_hist = np.sqrt(np.dot(np.square(np.arange(256) - mean_brightness), hist))

    weight = 0.5 
    average_contrast = (1 - weight) * 1000 * np.mean(contrast) + weight * contrast_from_hist

    return average_contrast

def estimate_sharpness(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    local_contrast = cv2.absdiff(image, blurred_image).astype(np.float32)
    
    global_contrast = np.std(image)

    sharpness = np.mean(local_contrast) / global_contrast

    return sharpness * 1000

def move_file(source, destination, jpeg_quality):
    if not os.path.exists(destination):
        os.makedirs(destination)
    basename, ext = os.path.splitext(os.path.basename(source))
    if ext.lower() in ('.png', '.jpeg', '.jpg', '.webp', '.gif'):
        try:
            image = imageio.imread(source)
        except Exception as e:
            print(f"Unable to read image {source}: {e}")
            return

        try:
            cv2.imwrite(os.path.join(destination, basename + '.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        except Exception as e:
            print(f"Unable to save image {source} as {destination}: {e}")
            return
    else:
        shutil.copy(source, destination)
    print(f"Copied {source} to {destination}")

def clear_output_folder(output_folder):
    for file_name in os.listdir(output_folder):
        file_path = os.path.join(output_folder, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed {file_path}")

def process_image(args):
    image_name, input_folder, output_folder, contrast_threshold, noise_threshold, blur_threshold, work_mode, sort_by, jpeg_quality, look_at_faces, visualize_mode, visualize_image_size, detector, sharpness_threshold = args
    image_path = os.path.join(input_folder, image_name)

    try:
        image = imageio.imread(image_path)
    except Exception as e:
        print(f"Skipping {image_name} - unable to read image file: {e}")
        return

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if image is None:
        print(f"Skipping {image_name} - unable to read image file")
        return
        
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = estimate_contrast(image)
    noise = estimate_noise(image)
    sharpness = estimate_sharpness(image)

    if look_at_faces == 1:
        blur = detect_blur_on_faces(image,visualize_mode, visualize_image_size, detector)
        if blur is None:
            blur = detect_blur(gray_image)
    else:
        blur = detect_blur(gray_image)

    meets_contrast = contrast >= contrast_threshold
    meets_noise = noise <= noise_threshold
    meets_blur = blur >= blur_threshold
    meets_sharpness = sharpness >= sharpness_threshold
    if work_mode == 1 or work_mode == 2:            
        if sort_by == 0:
            new_name = f"{noise:.0f}-noise,{blur:.0f}-blur,{sharpness:.0f}-sharpness,{contrast:.0f}-contrast.jpg"
        elif sort_by == 1: 
            new_name = f"{blur:.0f}-blur,{noise:.0f}-noise,{sharpness:.0f}-sharpness,{contrast:.0f}-contrast.jpg"
        elif sort_by == 2:  
            new_name = f"{sharpness:.0f}-sharpness,{noise:.0f}-noise,{blur:.0f}-blur,{contrast:.0f}-contrast.jpg"
        elif sort_by == 3: 
            new_name = f"{contrast:.0f}-contrast,{noise:.0f}-noise,{blur:.0f}-blur,{sharpness:.0f}-sharpness.jpg"
            
        new_image_path = os.path.join(input_folder, new_name)
        index = 1
        while os.path.exists(new_image_path):
            new_name = f"{os.path.splitext(new_name)[0]}_{index}{os.path.splitext(new_name)[1]}"
            new_image_path = os.path.join(input_folder, new_name)
            index += 1

        os.rename(image_path, new_image_path)
        image_path = new_image_path

        print(f"Renamed {image_name} to {new_name}")

    if work_mode == 0 or work_mode == 2:
        if meets_contrast and meets_noise and meets_blur and meets_sharpness:
            move_file(image_path, output_folder, jpeg_quality)

    return image_name, image_path, new_name if work_mode == 2 else None, meets_contrast and meets_noise and meets_blur

def main(num_processes):
    images = os.listdir(input_folder)
    print(f"{len(images)} files were found in the folder {input_folder}")
    clear_output_folder(output_folder)
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        args = zip(images, [input_folder] * len(images),
            [output_folder] * len(images), [contrast_threshold] * len(images),
            [noise_threshold] * len(images), [blur_threshold] * len(images), [work_mode] * len(images), 
            [sort_by] * len(images), [jpeg_quality] * len(images), [look_at_faces] * len(images),
            [visualize_mode] * len(images), [visualize_image_size] * len(images), [detector] * len(images), 
            [sharpness_threshold] * len(images))
        results = executor.map(process_image, args)
        for _ in results:
            pass

if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    detector = MTCNN()

    num_processes = config['num_processes']
    input_folder = config['input_folder']
    output_folder = config['output_folder']
    sort_by = config['sort_by']
    work_mode = config['work_mode']
    look_at_faces = config['look_at_faces']
    noise_threshold = config['noise_threshold']
    blur_threshold = config['blur_threshold']
    contrast_threshold = config['contrast_threshold']
    sharpness_threshold = config['sharpness_threshold']
    visualize_mode = config['visualize_mode']
    visualize_image_size = config['visualize_image_size']
    try:
        input_folder
    except NameError:
        input_folder = 'C:/input'
    try:
        output_folder
    except NameError:
        output_folder = 'C:/output'
    try:
        num_processes
    except NameError:
        num_processes = 5 
    try:
        contrast_threshold
    except NameError:
        contrast_threshold = 0 
    try:
        noise_threshold
    except NameError:
        noise_threshold = 220
    try:
        blur_threshold
    except NameError:
        blur_threshold = 50 
    try:
        sharpness_threshold
    except NameError:
        sharpness_threshold = 1.0
    try:
        sort_by
    except NameError:
        sort_by = 1 
    try:
        work_mode
    except NameError:
        work_mode = 0 
    try:
        look_at_faces
    except NameError:
        look_at_faces = 0
    try:
        visualize_mode
    except NameError:
        visualize_mode = 0 
    try:
        visualize_image_size
    except NameError:
        visualize_image_size = 1920 
    jpeg_quality = 95


    main(num_processes)
print("Processing complete")