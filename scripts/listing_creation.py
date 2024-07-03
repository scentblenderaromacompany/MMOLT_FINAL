import os
import json
import logging
import requests
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from datetime import datetime
import csv
from statistics import median
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import joblib
import pandas as pd
from scripts import image_processing, api_requests
from tenacity import retry, stop_after_attempt, wait_exponential

# Load pre-trained models
tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# Load price prediction model if it exists, otherwise train a new one
if os.path.exists('models/price_prediction_model.pkl'):
    price_model = joblib.load('models/price_prediction_model.pkl')
else:
    price_model = LinearRegression()

# Configure logging
logging.basicConfig(filename='logs/app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load settings
def load_settings():
    with open('config/settings.json', 'r') as f:
        return json.load(f)

settings = load_settings()

# Load policies
def load_policies():
    with open('config/policies.json', 'r') as f:
        return json.load(f)

policies = load_policies()

def generate_sku():
    return str(uuid4())

def generate_seo_title(keywords):
    prompt = f"Generate a SEO-optimized title for jewelry with keywords: {', '.join(keywords)}."
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=80, min_length=60, num_beams=4, early_stopping=True)
    title = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return title

def generate_seo_description(keywords):
    prompt = f"Generate a detailed and precise SEO-optimized description for jewelry with keywords: {', '.join(keywords)}."
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=300, min_length=150, num_beams=4, early_stopping=True)
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return description

def predict_price(features):
    return price_model.predict([features])[0]

def get_ebay_sold_prices(title):
    url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
    headers = {
        "Authorization": f"Bearer {settings['ebay_api']['auth_token']}",
        "Content-Type": "application/json"
    }
    params = {
        "q": title,
        "filter": "sold:true",
        "sort": "endTime:desc",
        "limit": 50,
        "date_range": "last30days"
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        items = response.json().get('itemSummaries', [])
        prices = [item['price']['value'] for item in items if 'price' in item]
        if prices:
            return median(prices)
    logging.error(f"Failed to get sold prices for {title}: {response.content}")
    return None

def create_listing_data(folder_path, sku, keywords):
    try:
        title = generate_seo_title(keywords)
        description = generate_seo_description(keywords)
        median_price = get_ebay_sold_prices(title)
        if (median_price is None):
            median_price = 19.99  # Default price if no median price found

        features = [len(keywords), median_price, len(title)]  # Example features, customize as needed
        predicted_price = predict_price(features)

        currency = settings["currency"]
        images = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('_watermarked.jpg')]

        # Combine policies into the description
        description += "\n\n"
        for policy in policies.values():
            description += f"{policy['title']}\n"
            for line in policy['content']:
                description += f"{line}\n"
            description += "\n"

        listing_data = {
            "sku": sku,
            "title": title,
            "description": description,
            "price": predicted_price,
            "currency": currency,
            "images": images,
            "keywords": keywords,
            "brand": "UNBRANDED",
            "color": "Various",
            "type": "Jewelry",
            "closure": "Various",
            "materials": "Various",
            "place_of_origin": "Unknown",
            "shape": "Various",
            "pendant_shape": "Various",
            "stone": "Various",
            "length": "Various",
            "theme": "Fashion & Costumes",
            "occasion": "Various",
            "gender": "Unisex",
            "metal_properties": "Various",
            "metal_purities": "Various"
        }

        return listing_data
    except Exception as e:
        logging.error(f"Error creating listing data: {e}")
        return None

def process_product_folder(folder_path, sku, watermark_text):
    try:
        processed_images = image_processing.process_images_in_folder(folder_path, watermark_text)
        keywords = []
        for image_path in processed_images:
            keywords.extend(image_processing.extract_keywords_from_image(image_path))
        
        keywords = list(set(keywords))  # Remove duplicates
        listing_data = create_listing_data(folder_path, sku, keywords)

        with open(os.path.join(folder_path, 'keywords.txt'), 'w') as f:
            f.write("\n".join(keywords))

        with open(os.path.join(folder_path, 'base64.txt'), 'w') as f:
            for image_path in processed_images:
                base64_image = image_processing.encode_image_to_base64(image_path)
                f.write(f"{os.path.basename(image_path)}: {base64_image}\n")

        thumbnails = image_processing.generate_thumbnails(processed_images)
        for thumb in thumbnails:
            os.makedirs(os.path.join(folder_path, 'thumbnails'), exist_ok=True)
            os.rename(thumb, os.path.join(folder_path, 'thumbnails', os.path.basename(thumb)))

        if listing_data:
            with open(os.path.join(folder_path, 'listing.txt'), 'w') as f:
                json.dump(listing_data, f, indent=4)

        return listing_data
    except Exception as e:
        logging.error(f"Error processing product folder {folder_path}: {e}")
        return None

def create_output_directory():
    timestamp = datetime.now().strftime('%m-%d-%Y_%I-%M%p')
    output_dir = os.path.join('output/processed', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def process_main_folder(main_folder_path, watermark_text, progress_callback=None):
    listings = []
    try:
        skus = {}
        output_dir = create_output_directory()
        with open(os.path.join(output_dir, 'sku_tracking.csv'), 'w', newline='') as csvfile:
            fieldnames = ['SKU', 'Folder']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            with ThreadPoolExecutor() as executor:
                futures = []
                for i, folder in enumerate(os.listdir(main_folder_path)):
                    if os.path.isdir(os.path.join(main_folder_path, folder)):
                        sku = generate_sku()
                        skus[sku] = folder
                        futures.append(executor.submit(process_product_folder, os.path.join(main_folder_path, folder), sku, watermark_text))

                for i, future in enumerate(futures):
                    result = future.result()
                    if result:
                        listings.append(result)
                    if progress_callback:
                        progress_callback(i + 1, len(futures))
                
                for sku, folder in skus.items():
                    writer.writerow({'SKU': sku, 'Folder': folder})

        return listings
    except Exception as e:
        logging.error(f"Error processing main folder {main_folder_path}: {e}")
        return listings

# Retry configuration for posting listings
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def post_listing_with_retry(post_function, listing_data):
    post_function(listing_data)

def post_to_ebay(listing_data):
    url = "https://api.ebay.com/sell/inventory/v1/offer"
    headers = {
        "Authorization": f"Bearer {settings['ebay_api']['auth_token']}",
        "Content-Type": "application/json"
    }
    payload = {
        "sku": listing_data['sku'],
        "marketplaceId": "EBAY_US",
        "format": "FIXED_PRICE",
        "listingDescription": listing_data['description'],
        "availableQuantity": 1,
        "categoryId": "category_id",
        "listingPolicies": {
            "fulfillmentPolicyId": settings['ebay_api']['fulfillment_policy_id'],
            "paymentPolicyId": settings['ebay_api']['payment_policy_id'],
            "returnPolicyId": settings['ebay_api']['return_policy_id']
        },
        "pricingSummary": {
            "price": {
                "value": listing_data['price'],
                "currency": settings['currency']
            }
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 201:
        logging.info(f"Successfully posted to eBay: {listing_data['title']}")
    else:
        logging.error(f"Failed to post to eBay: {response.content}")

def post_to_etsy(listing_data):
    url = f"https://api.etsy.com/v3/application/shops/{settings['etsy_api']['shop_id']}/listings"
    headers = {
        "x-api-key": settings['etsy_api']['api_key'],
        "Authorization": f"Bearer {settings['etsy_api']['access_token']}",
        "Content-Type": "application/json"
    }
    payload = {
        "title": listing_data['title'],
        "description": listing_data['description'],
        "price": listing_data['price'],
        "quantity": 1,
        "shop_section_id": settings['etsy_api']['shop_section_id'],
        "shipping_template_id": settings['etsy_api']['shipping_template_id'],
        "taxonomy_id": settings['etsy_api']['taxonomy_id'],
        "materials": listing_data['materials'],
        "tags": listing_data['keywords'],
        "who_made": "i_did",
        "is_supply": False,
        "when_made": "made_to_order"
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 201:
        logging.info(f"Successfully posted to Etsy: {listing_data['title']}")
    else:
        logging.error(f"Failed to post to Etsy: {response.content}")

def post_to_facebook_marketplace(listing_data):
    url = f"https://graph.facebook.com/v11.0/{settings['facebook_marketplace_api']['catalog_id']}/products"
    headers = {
        "Authorization": f"Bearer {settings['facebook_marketplace_api']['access_token']}",
        "Content-Type": "application/json"
    }
    payload = {
        "retailer_id": listing_data['sku'],
        "name": listing_data['title'],
        "description": listing_data['description'],
        "price": listing_data['price'],
        "currency": settings['currency'],
        "image_url": listing_data['images'][0],
        "availability": "in stock",
        "condition": "new"
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        logging.info(f"Successfully posted to Facebook Marketplace: {listing_data['title']}")
    else:
        logging.error(f"Failed to post to Facebook Marketplace: {response.content}")

def post_to_instagram(listing_data):
    url = f"https://graph.facebook.com/v11.0/{settings['instagram_api']['catalog_id']}/products"
    headers = {
        "Authorization": f"Bearer {settings['instagram_api']['access_token']}",
        "Content-Type": "application/json"
    }
    payload = {
        "retailer_id": listing_data['sku'],
        "name": listing_data['title'],
        "description": listing_data['description'],
        "price": listing_data['price'],
        "currency": settings['currency'],
        "image_url": listing_data['images'][0],
        "availability": "in stock",
        "condition": "new"
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        logging.info(f"Successfully posted to Instagram: {listing_data['title']}")
    else:
        logging.error(f"Failed to post to Instagram: {response.content}")

def post_to_mercari(listing_data):
    url = f"https://api.mercari.com/v3/listings"
    headers = {
        "Authorization": f"Bearer {settings['mercari_api']['access_token']}",
        "Content-Type": "application/json"
    }
    payload = {
        "title": listing_data['title'],
        "description": listing_data['description'],
        "price": listing_data['price'],
        "category_id": settings['mercari_api']['category_id'],
        "brand": listing_data['brand'],
        "condition": "new",
        "shipping_payer": "seller",
        "shipping_method": "mercari_pack",
        "item_status": "on_sale",
        "item_number": listing_data['sku'],
        "stock": 1,
        "image_urls": listing_data['images']
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        logging.info(f"Successfully posted to Mercari: {listing_data['title']}")
    else:
        logging.error(f"Failed to post to Mercari: {response.content}")

def post_to_poshmark(listing_data):
    url = f"https://api.poshmark.com/v3/listings"
    headers = {
        "Authorization": f"Bearer {settings['poshmark_api']['access_token']}",
        "Content-Type": "application/json"
    }
    payload = {
        "title": listing_data['title'],
        "description": listing_data['description'],
        "price": listing_data['price'],
        "category": settings['poshmark_api']['category'],
        "brand": listing_data['brand'],
        "color": listing_data['color'],
        "size": "one size",
        "condition": "new",
        "availability": "in stock",
        "sku": listing_data['sku'],
        "images": listing_data['images']
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        logging.info(f"Successfully posted to Poshmark: {listing_data['title']}")
    else:
        logging.error(f"Failed to post to Poshmark: {response.content}")

def post_to_offerup(listing_data):
    url = f"https://api.offerup.com/v3/listings"
    headers = {
        "Authorization": f"Bearer {settings['offerup_api']['access_token']}",
        "Content-Type": "application/json"
    }
    payload = {
        "title": listing_data['title'],
        "description": listing_data['description'],
        "price": listing_data['price'],
        "category_id": settings['offerup_api']['category_id'],
        "brand": listing_data['brand'],
        "condition": "new",
        "availability": "in stock",
        "sku": listing_data['sku'],
        "images": listing_data['images']
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        logging.info(f"Successfully posted to OfferUp: {listing_data['title']}")
    else:
        logging.error(f"Failed to post to OfferUp: {response.content}")

def post_to_pinterest(listing_data):
    url = f"https://api.pinterest.com/v3/catalogs/products"
    headers = {
        "Authorization": f"Bearer {settings['pinterest_api']['access_token']}",
        "Content-Type": "application/json"
    }
    payload = {
        "retailer_id": listing_data['sku'],
        "title": listing_data['title'],
        "description": listing_data['description'],
        "price": listing_data['price'],
        "currency": settings['currency'],
        "image_url": listing_data['images'][0],
        "availability": "in stock",
        "condition": "new"
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 201:
        logging.info(f"Successfully posted to Pinterest: {listing_data['title']}")
    else:
        logging.error(f"Failed to post to Pinterest: {response.content}")

def post_to_twitter(listing_data):
    url = f"https://api.twitter.com/2/tweets"
    headers = {
        "Authorization": f"Bearer {settings['twitter_api']['access_token']}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": f"Check out our new listing: {listing_data['title']} for ${listing_data['price']}. {listing_data['description']} {listing_data['images'][0]}"
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 201:
        logging.info(f"Successfully posted to Twitter: {listing_data['title']}")
    else:
        logging.error(f"Failed to post to Twitter: {response.content}")

def post_listings_to_marketplaces(listings, marketplaces):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for listing in listings:
            for marketplace in marketplaces:
                if marketplace == "eBay":
                    futures.append(executor.submit(post_listing_with_retry, post_to_ebay, listing))
                elif marketplace == "Etsy":
                    futures.append(executor.submit(post_listing_with_retry, post_to_etsy, listing))
                elif marketplace == "Facebook Marketplace":
                    futures.append(executor.submit(post_listing_with_retry, post_to_facebook_marketplace, listing))
                elif marketplace == "Instagram":
                    futures.append(executor.submit(post_listing_with_retry, post_to_instagram, listing))
                elif marketplace == "Mercari":
                    futures.append(executor.submit(post_listing_with_retry, post_to_mercari, listing))
                elif marketplace == "Poshmark":
                    futures.append(executor.submit(post_listing_with_retry, post_to_poshmark, listing))
                elif marketplace == "OfferUp":
                    futures.append(executor.submit(post_listing_with_retry, post_to_offerup, listing))
                elif marketplace == "Pinterest":
                    futures.append(executor.submit(post_listing_with_retry, post_to_pinterest, listing))
                elif marketplace == "Twitter":
                    futures.append(executor.submit(post_listing_with_retry, post_to_twitter, listing))
        
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error posting listing: {e}")

def train_price_prediction_model(historical_data_path):
    try:
        historical_data = pd.read_csv(historical_data_path)
        X = historical_data[['feature1', 'feature2', 'feature3']]  # Replace with actual feature names
        y = historical_data['price']
        
        model = LinearRegression()
        model.fit(X, y)
        
        joblib.dump(model, 'models/price_prediction_model.pkl')
        logging.info("Price prediction model trained and saved as price_prediction_model.pkl")
    except Exception as e:
        logging.error(f"Error training price prediction model: {e}")

def analyze_customer_feedback(feedback):
    try:
        analysis = TextBlob(feedback)
        sentiment = analysis.sentiment
        return sentiment.polarity, sentiment.subjectivity
    except Exception as e:
        logging.error(f"Error analyzing customer feedback: {e}")
        return None, None

def main():
    settings = load_settings()
    policies = load_policies()
    
    main_folder_path = 'input/EEE/raw'
    watermark_text = settings["watermark_text"]
    listings = process_main_folder(main_folder_path, watermark_text, progress_callback=None)
    
    marketplaces = settings["selected_marketplaces"]
    
    post_listings_to_marketplaces(listings, marketplaces)

if __name__ == "__main__":
    main()
