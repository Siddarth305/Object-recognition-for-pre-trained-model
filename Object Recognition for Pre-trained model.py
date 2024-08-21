from PIL import Image
import numpy as np

def load_image(image_path):
    return Image.open(image_path).convert('L')  # Convert to grayscale

def image_to_array(image):
    return np.array(image)

def normalize_image(image_array):
    mean = np.mean(image_array)
    std = np.std(image_array)
    return (image_array - mean) / std

def save_sub_image(image_array, top_left, size, file_name):
    y, x = top_left
    h, w = size
    sub_image = image_array[y:y+h, x:x+w]
    Image.fromarray(sub_image).save(file_name)

def match_template(image, template):
    image_array = normalize_image(image_to_array(image))
    template_array = normalize_image(image_to_array(template))
    
    img_h, img_w = image_array.shape
    tmpl_h, tmpl_w = template_array.shape
    
    # Use FFT to compute the correlation
    correlation = np.fft.fft2(image_array) * np.conj(np.fft.fft2(template_array, s=image_array.shape))
    correlation = np.fft.ifft2(correlation).real  # Inverse FFT to get the correlation result
    
    # Normalize the correlation map
    correlation = correlation / (tmpl_h * tmpl_w)  # Adjust for the template size
    
    # Find the best match
    best_match = np.unravel_index(np.argmax(correlation), correlation.shape)
    best_score = correlation[best_match]

    # Save the best matching region for visualization
    save_sub_image(image_array, best_match, (tmpl_h, tmpl_w), 'best_match_region.jpg')
    
    return best_match, best_score

# Provide the paths to your image and template
image_path = "C:\\Users\\Lenovo\\Downloads\\download.jpeg"
template_path = "C:\\Users\\Lenovo\\Downloads\\download (1).jpeg"

# Load the image and template
image = load_image(image_path)
template = load_image(template_path)

# Perform template matching
best_match, best_score = match_template(image, template)

# Display the result
if best_match:
    print(f"Best match found at: {best_match} with score: {best_score}")
else:
    print("No match found")
print(f"Image size: {image.size}")
print(f"Template size: {template.size}")
