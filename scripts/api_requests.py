import os
import json
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor
import joblib

# Configure logging
logging.basicConfig(filename='logs/api_requests.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load settings
def load_settings():
    with open('config/settings.json', 'r') as f:
        return json.load(f)

settings = load_settings()

# Load models
price_model = joblib.load('models/price_prediction_model.pkl')
text_embedding_model = joblib.load('models/text_embedding_model.pkl')
image_search_model, image_paths = joblib.load('models/image_search_model.pkl')

# General function to send an API request with retry logic
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def send_api_request(url, headers, data, request_type='POST'):
    try:
        if request_type == 'POST':
            response = requests.post(url, headers=headers, json=data)
        elif request_type == 'PUT':
            response = requests.put(url, headers=headers, json=data)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        raise

# Function to post a listing to eBay
def post_to_ebay(listing):
    url = settings['ebay_api']['production']['endpoints']['inventory_offer']
    headers = {
        "Authorization": f"Bearer {settings['ebay_api']['production']['auth_token']}",
        "Content-Type": "application/json"
    }
    data = {
        "sku": listing['sku'],
        "marketplaceId": "EBAY_US",
        "format": "FIXED_PRICE",
        "listingDescription": listing['description'],
        "availableQuantity": 1,
        "categoryId": "YOUR_EBAY_CATEGORY_ID",
        "listingPolicies": {
            "fulfillmentPolicyId": settings['ebay_api']['fulfillment_policy_id'],
            "paymentPolicyId": settings['ebay_api']['payment_policy_id'],
            "returnPolicyId": settings['ebay_api']['return_policy_id']
        },
        "pricingSummary": {
            "price": {
                "value": listing['price'],
                "currency": settings['currency']
            }
        }
    }
    response = send_api_request(url, headers, data)
    if response:
        logging.info(f"Successfully posted listing {listing['title']} to eBay")
    else:
        logging.error(f"Failed to post listing {listing['title']} to eBay")

# Function to post a listing to Etsy
def post_to_etsy(listing):
    url = settings['etsy_api']['endpoints']['create_listing'].replace('{shop_id}', settings['etsy_api']['shop_id'])
    headers = {
        "x-api-key": settings['etsy_api']['api_key'],
        "Authorization": f"Bearer {settings['etsy_api']['access_token']}",
        "Content-Type": "application/json"
    }
    data = {
        "title": listing['title'],
        "description": listing['description'],
        "price": listing['price'],
        "quantity": 1,
        "shipping_template_id": settings['etsy_api']['shipping_template_id'],
        "shop_section_id": settings['etsy_api']['shop_section_id'],
        "taxonomy_id": settings['etsy_api']['taxonomy_id'],
        "materials": ["Various"],
        "who_made": "i_did",
        "is_supply": "false",
        "when_made": "made_to_order"
    }
    response = send_api_request(url, headers, data)
    if response:
        logging.info(f"Successfully posted listing {listing['title']} to Etsy")
    else:
        logging.error(f"Failed to post listing {listing['title']} to Etsy")

# Function to post a listing to Facebook Marketplace
def post_to_facebook(listing, folder_path):
    url = settings['facebook_marketplace_api']['endpoints']['add_product'].replace('{catalog_id}', settings['facebook_marketplace_api']['catalog_id'])
    headers = {
        "Authorization": f"Bearer {settings['facebook_marketplace_api']['access_token']}",
        "Content-Type": "application/json"
    }
    data = {
        "retailer_id": listing['sku'],
        "name": listing['title'],
        "description": listing['description'],
        "price": listing['price'],
        "currency": settings['currency'],
        "image_url": listing['images'][0],
        "availability": "in stock",
        "condition": "new"
    }
    response = send_api_request(url, headers, data)
    if response:
        logging.info(f"Successfully posted listing {listing['title']} to Facebook Marketplace")
    else:
        logging.error(f"Failed to post listing {listing['title']} to Facebook Marketplace")
        backup_file = os.path.join(folder_path, 'facebook_backup.json')
        with open(backup_file, 'w') as f:
            json.dump(listing, f)

# Function to post a listing to Instagram
def post_to_instagram(listing, folder_path):
    create_container_url = f"https://graph.instagram.com/v11.0/{settings['instagram_api']['user_id']}/media"
    publish_url = f"https://graph.instagram.com/v11.0/{settings['instagram_api']['user_id']}/media_publish"
    headers = {
        "Authorization": f"Bearer {settings['instagram_api']['access_token']}",
        "Content-Type": "application/json"
    }
    container_data = {
        "image_url": listing['images'][0],
        "caption": f"{listing['title']}\n\n{listing['description']}",
        "access_token": settings['instagram_api']['access_token']
    }
    container_response = send_api_request(create_container_url, headers, container_data)
    if container_response:
        container_id = container_response.json().get('id')
        publish_data = {
            "creation_id": container_id,
            "access_token": settings['instagram_api']['access_token']
        }
        publish_response = send_api_request(publish_url, headers, publish_data)
        if publish_response:
            logging.info(f"Successfully posted listing {listing['title']} to Instagram")
        else:
            logging.error(f"Failed to publish listing {listing['title']} to Instagram")
    else:
        logging.error(f"Failed to create media container for listing {listing['title']} on Instagram")
        backup_file = os.path.join(folder_path, 'instagram_backup.json')
        with open(backup_file, 'w') as f:
            json.dump(listing, f)

# Function to post a listing to Mercari
def post_to_mercari(listing):
    url = settings['mercari_api']['endpoints']['create_listing']
    headers = {
        "Authorization": f"Bearer {settings['mercari_api']['access_token']}",
        "Content-Type": "application/json"
    }
    data = {
        "name": listing['title'],
        "description": listing['description'],
        "price": listing['price'],
        "category_id": settings['mercari_api']['category_id'],
        "brand": listing['brand'],
        "condition": "new",
        "shipping_payer": "seller",
        "item_status": "on_sale",
        "shipping_method": "ship_by_seller",
        "image_urls": listing['images']
    }
    response = send_api_request(url, headers, data)
    if response:
        logging.info(f"Successfully posted listing {listing['title']} to Mercari")
    else:
        logging.error(f"Failed to post listing {listing['title']} to Mercari")

# Function to post a listing to Poshmark
def post_to_poshmark(listing):
    url = settings['poshmark_api']['endpoints']['create_listing']
    headers = {
        "Authorization": f"Bearer {settings['poshmark_api']['access_token']}",
        "Content-Type": "application/json"
    }
    data = {
        "title": listing['title'],
        "description": listing['description'],
        "price": listing['price'],
        "quantity": 1,
        "category_id": settings['poshmark_api']['category_id'],
        "brand": listing['brand'],
        "color": ["Various"],
        "size": "One Size",
        "condition": "New",
        "image_urls": listing['images']
    }
    response = send_api_request(url, headers, data)
    if response:
        logging.info(f"Successfully posted listing {listing['title']} to Poshmark")
    else:
        logging.error(f"Failed to post listing {listing['title']} to Poshmark")

# Function to post a listing to OfferUp
def post_to_offerup(listing):
    url = settings['offerup_api']['endpoints']['create_listing']
    headers = {
        "Authorization": f"Bearer {settings['offerup_api']['access_token']}",
        "Content-Type": "application/json"
    }
    data = {
        "title": listing['title'],
        "description": listing['description'],
        "price": listing['price'],
        "category_id": settings['offerup_api']['category_id'],
        "condition": "New",
        "image_urls": listing['images']
    }
    response = send_api_request(url, headers, data)
    if response:
        logging.info(f"Successfully posted listing {listing['title']} to OfferUp")
    else:
        logging.error(f"Failed to post listing {listing['title']} to OfferUpSure, here is the enhanced `api_requests.py` script saved to a file:
# Function to post a listing to Pinterest
def post_to_pinterest(listing):
    url = settings['pinterest_api']['endpoints']['add_product']
    headers = {
        "Authorization": f"Bearer {settings['pinterest_api']['access_token']}",
        "Content-Type": "application/json"
    }
    data = {
        "retailer_id": listing['sku'],
        "title": listing['title'],
        "description": listing['description'],
        "price": listing['price'],
        "currency": settings['currency'],
        "image_url": listing['images'][0],
        "availability": "in stock",
        "condition": "new"
    }
    response = send_api_request(url, headers, data)
    if response:
        logging.info(f"Successfully posted listing {listing['title']} to Pinterest")
    else:
        logging.error(f"Failed to post listing {listing['title']} to Pinterest")

# Function to post a listing to Twitter
def post_to_twitter(listing):
    url = settings['twitter_api']['endpoints']['create_tweet']
    headers = {
        "Authorization": f"Bearer {settings['twitter_api']['access_token']}",
        "Content-Type": "application/json"
    }
    data = {
        "text": f"Check out our new listing: {listing['title']} for ${listing['price']}. {listing['description']} {listing['images'][0]}"
    }
    response = send_api_request(url, headers, data)
    if response:
        logging.info(f"Successfully posted listing {listing['title']} to Twitter")
    else:
        logging.error(f"Failed to post listing {listing['title']} to Twitter")

# Function to post a listing to multiple marketplaces concurrently
def post_to_marketplace_concurrently(listing, marketplace, folder_path):
    if marketplace == "eBay":
        post_to_ebay(listing)
    elif marketplace == "Etsy":
        post_to_etsy(listing)
    elif marketplace == "Facebook Marketplace":
        post_to_facebook(listing, folder_path)
    elif marketplace == "Instagram":
        post_to_instagram(listing, folder_path)
    elif marketplace == "Mercari":
        post_to_mercari(listing)
    elif marketplace == "Poshmark":
        post_to_poshmark(listing)
    elif marketplace == "OfferUp":
        post_to_offerup(listing)
    elif marketplace == "Pinterest":
        post_to_pinterest(listing)
    elif marketplace == "Twitter":
        post_to_twitter(listing)
    else:
        logging.error(f"Unsupported marketplace: {marketplace}")

def post_listings_concurrently(listings, marketplaces, folder_path):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for listing in listings:
            for marketplace in marketplaces:
                futures.append(executor.submit(post_to_marketplace_concurrently, listing, marketplace, folder_path))
        
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error posting listing: {e}")

# Example usage
# listing = {
#     "sku": "SKU123",
#     "title": "Beautiful Necklace",
#     "description": "A stunning necklace with exquisite details.",
#     "price": 29.99,
#     "currency": "USD",
#     "images": ["https://example.com/image1.jpg"],
#     "brand": "UNBRANDED"
# }
# post_listings_concurrently([listing], ["eBay", "Etsy", "Facebook Marketplace", "Instagram", "Mercari", "Poshmark", "OfferUp", "Pinterest", "Twitter"], "/path/to/folder")
