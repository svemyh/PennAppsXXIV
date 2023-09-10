import json
import time

from PIL import Image

def average_pixel_value(image_path):
    try:
        # Open the image
        img = Image.open(image_path)
    except IOError:
        raise ValueError("Unable to read the image. Please provide a valid image path.")
    
    # Convert the image to grayscale
    img_gray = img.convert('L')
    
    # Calculate the average pixel value
    pixel_data = list(img_gray.getdata())
    average_value = sum(pixel_data) / len(pixel_data)
    
    # Normalize the average pixel value to the range [0, 1]
    normalized_average_value = average_value / 255.0
    
    return normalized_average_value


def dummy_predict(img_path):
    try:
        avg_value = average_pixel_value(img_path)
        timestamp = int(time.time())  # Get the current timestamp
        filename = img_path.split("/")[-1]  # Extract the filename from the path

        # Create a dictionary to represent the result
        result = {
            "average_pixel_value": avg_value,
            "timestamp": timestamp,
            "filename": filename
        }

        # Convert the result to a JSON string
        result_json = json.dumps(result)

        return result_json
    except Exception as e:
        return str(e)  # Return an error message if something goes wrong



# Example usage:
#image_path = 'path/to/your/image.jpg'
#result_json = dummy_predict(image_path)
#print(result_json)
