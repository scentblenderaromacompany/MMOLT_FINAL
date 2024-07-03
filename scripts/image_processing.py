import os
import logging
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ExifTags
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pytesseract
import base64
from rembg import remove
import joblib
import requests

# Load TensorFlow model for image recognition
model = hub.load("https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5")
labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
with open(labels_path) as f:
    labels = f.read().splitlines()

# Load models
price_model = joblib.load('models/price_prediction_model.pkl')
text_embedding_model = joblib.load('models/text_embedding_model.pkl')
image_search_model, image_paths = joblib.load('models/image_search_model.pkl')

# eBay API credentials
EBAY_APP_ID = 'YourAppIDHere'
EBAY_CERT_ID = 'YourCertIDHere'
EBAY_DEV_ID = 'YourDevIDHere'
EBAY_TOKEN = 'YourUserTokenHere'

# Helper functions
def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def recognize_image(image_path):
    image = load_image(image_path)
    predictions = model(image)
    predicted_label_index = np.argmax(predictions)
    predicted_label = labels[predicted_label_index]
    return predicted_label

def enhance_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        img = remove(img)  # Background removal
        return img
    except Exception as e:
        logging.error(f"Error enhancing image {image_path}: {e}")
        return None

def crop_and_align_image(image):
    try:
        img_gray = image.convert("L")
        bbox = img_gray.point(lambda x: 0 if x < 128 else 255).getbbox()
        if bbox:
            image = image.crop(bbox)
        image = image.resize((224, 224), Image.ANTIALIAS)
        return image
    except Exception as e:
        logging.error(f"Error cropping and aligning image: {e}")
        return image

def rotate_image(image_path):
    try:
        img = Image.open(image_path)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif is not None:
            orientation = exif.get(orientation)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
        return img
    except Exception as e:
        logging.error(f"Error rotating image {image_path}: {e}")
        return None

def convert_image_to_png(image_path):
    try:
        img = Image.open(image_path)
        output_path = f"{os.path.splitext(image_path)[0]}.png"
        img.save(output_path, format="PNG", quality=95)
        return output_path
    except Exception as e:
        logging.error(f"Error converting image to PNG: {e}")
        return image_path

def add_watermark(image, watermark_text, font_path='arial.ttf', opacity=128):
    try:
        width, height = image.size
        watermark = Image.new('RGBA', (width, height))
        draw = ImageDraw.Draw(watermark, 'RGBA')
        font = ImageFont.truetype(font_path, int(height / 20))
        text_width, text_height = draw.textsize(watermark_text, font)
        position = ((width - text_width) // 2, (height - text_height) // 2)
        draw.text(position, watermark_text, fill=(255, 255, 255, opacity), font=font)
        watermarked = Image.alpha_composite(image.convert('RGBA'), watermark)
        return watermarked.convert('RGB')
    except Exception as e:
        logging.error(f"Error adding watermark to image: {e}")
        return image

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding image to base64: {e}")
        return ""

def ebay_image_search(base64_image):
    url = "https://api.ebay.com/ws/api.dll"
    headers = {
        "Content-Type": "application/xml",
        "X-EBAY-API-CALL-NAME": "findItemsByImage",
        "X-EBAY-API-APP-ID": EBAY_APP_ID,
        "X-EBAY-API-SITE-ID": "0",
        "X-EBAY-API-COMPATIBILITY-LEVEL": "967",
        "Authorization": f"Bearer {EBAY_TOKEN}"
    }
    payload = f"""
    <?xml version="1.0" encoding="utf-8"?>
    <findItemsByImageRequest xmlns="http://www.ebay.com/marketplace/search/v1/services">
        <imageData>{base64_image}</imageData>
    </findItemsByImageRequest>
    """
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code == 200:
        return parse_ebay_response(response.text)
    else:
        logging.error(f"Error with eBay image search API: {response.status_code}")
        return []

def parse_ebay_response(response_text):
    import xml.etree.ElementTree as ET
    root = ET.fromstring(response_text)
    keywords = []
    for item in root.findall('.//item'):
        title = item.find('title').text
        if title:
            keywords.extend(title.split())
    return list(set(keywords))  # Remove duplicates

def get_item_details(sku):
    url = f"https://api.ebay.com/sell/inventory/v1/inventory_item/{sku}"
    headers = {
        "Authorization": f"Bearer {EBAY_TOKEN}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Error retrieving item details for SKU {sku}: {response.status_code}")
        return {}

def write_keywords_to_file(folder_path, image_name, keywords):
    try:
        keywords_file = os.path.join(folder_path, 'metadata', f"{os.path.splitext(image_name)[0]}_keywords.txt")
        os.makedirs(os.path.dirname(keywords_file), exist_ok=True)
        with open(keywords_file, 'w') as file:
            file.write("\n".join(keywords))
    except Exception as e:
        logging.error(f"Error writing keywords to file: {e}")

def update_model_with_keywords(image_path, keywords):
    try:
        # Load the existing training data if it exists
        training_data_path = 'training_data.csv'
        if os.path.exists(training_data_path):
            training_data = pd.read_csv(training_data_path)
        else:
            training_data = pd.DataFrame(columns=['image_path', 'keywords'])

        # Add the new data
        new_data = pd.DataFrame({'image_path': [image_path], 'keywords': [keywords]})
        training_data = pd.concat([training_data, new_data], ignore_index=True)

        # Save the updated training data
        training_data.to_csv(training_data_path, index=False)
        logging.info(f"Updated training data with keywords for image {image_path}")

        # Retrain the model with the updated training data
        retrain_model(training_data)

    except Exception as e:
        logging.error(f"Error updating model with keywords for image {image_path}: {e}")

def preprocess_data(training_data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(training_data['keywords'].values)
    sequences = tokenizer.texts_to_sequences(training_data['keywords'].values)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences)
    return data, word_index

def build_model(input_length, vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=input_length))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def retrain_model(training_data):
    try:
        data, word_index = preprocess_data(training_data)
        vocab_size = len(word_index) + 1
        input_length = data.shape[1]
        
        model = build_model(input_length, vocab_size)
        
        labels = to_categorical(training_data.index.values, num_classes=vocab_size)
        
        model.fit(data, labels, epochs=10, batch_size=32, validation_split=0.2)
        
        model.save('models/retrained_model.h5')
        logging.info("Model retrained and saved as retrained_model.h5")
        
    except Exception as e:
        logging.error(f"Error retraining model: {e}")

def compress_image(image_path, quality=85):
    try:
        img = Image.open(image_path)
        img.save(image_path, "JPEG", quality=quality)
        logging.info(f"Compressed image {image_path}")
    except Exception as e:
        logging.error(f"Error compressing image {image_path}: {e}")

# Add call to compress_image in process_and_categorize_images
def process_and_categorize_images(folder_path, watermark_text):
    try:
        processed_images = []
        categories = []
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.heic')):
                image_path = os.path.join(folder_path, filename)
                
                # Convert to PNG
                image_path = convert_image_to_png(image_path)
                
                # Enhance image
                img = enhance_image(image_path)
                
                # Crop and align image
                img = crop_and_align_image(img)
                
                # Rotate image if needed
                img = rotate_image(image_path)
                
                if img:
                    # Add watermark
                    img = add_watermark(img, watermark_text)
                    output_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}_processed.png")
                    img.save(output_path)
                    processed_images.append(output_path)
                    
                    # Create metadata directories
                    base64_folder = os.path.join(folder_path, 'metadata', 'base64')
                    os.makedirs(base64_folder, exist_ok=True)
                    
                    # Encode and save base64
                    base64_str = encode_image_to_base64(output_path)
                    base64_file = os.path.join(base64_folder, f"{os.path.splitext(filename)[0]}_base64.txt")
                    with open(base64_file, 'w') as file:
                        file.write(base64_str)
                    
                    # Extract and write keywords
                    keywords = parse_ebay_response(image_path)
                    write_keywords_to_file(folder_path, filename, keywords)
                    
                    # Categorize product
                    categories.append(recognize_image(image_path))
                    
                    # Create and save thumbnails
                    thumbnails_folder = os.path.join(folder_path, 'thumbnails')
                    os.makedirs(thumbnails_folder, exist_ok=True)
                    thumbnail_path = os.path.join(thumbnails_folder, f"{os.path.splitext(filename)[0]}_thumbnail.png")
                    img.thumbnail((128, 128))
                    img.save(thumbnail_path)
                    
                    # Compress image
                    compress_image(output_path)
                    
        return processed_images, categories
    except Exception as e:
        logging.error(f"Error processing and categorizing images in folder {folder_path}: {e}")
        return [], []
