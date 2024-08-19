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
    
    best_match = None
    best_score = -float('inf')  # For NCC, higher values are better
    
    template_mean = np.mean(template_array)
    template_std = np.std(template_array)
    
    for y in range(img_h - tmpl_h + 1):
        for x in range(img_w - tmpl_w + 1):
            sub_image = image_array[y:y+tmpl_h, x:x+tmpl_w]
            sub_image_mean = np.mean(sub_image)
            sub_image_std = np.std(sub_image)
            
            norm_sub_image = (sub_image - sub_image_mean) / sub_image_std
            norm_template = (template_array - template_mean) / template_std
            
            score = np.sum(norm_sub_image * norm_template)
            print(f"Matching region at ({x}, {y}) with score: {score}")
            
            # Save the best matching region for visualization
            if score > best_score:
                best_score = score
                best_match = (x, y)
                save_sub_image(image_array, (y, x), (tmpl_h, tmpl_w), 'best_match_region.jpg')
    
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






output:
print(f"Image size: {image.size}")
print(f"Template size: {template.size}")


