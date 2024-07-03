import os
import logging
import json
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from scripts import listing_creation, api_requests
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(filename='logs/schedule_tasks.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load settings
def load_settings():
    with open('config/settings.json', 'r') as f:
        return json.load(f)

settings = load_settings()

# Retry configuration
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def post_listings_with_retry(listings, marketplaces):
    listing_creation.post_listings_to_marketplaces(listings, marketplaces)

def process_scheduled_folder(folder_path, watermark_text):
    try:
        listings = listing_creation.process_main_folder(folder_path, watermark_text)
        if listings:
            marketplaces = settings["selected_marketplaces"]
            post_listings_with_retry(listings, marketplaces)
            logging.info(f"Successfully processed and posted listings from {folder_path}")
        else:
            logging.error(f"Failed to process listings from {folder_path}")
    except Exception as e:
        logging.error(f"Error processing scheduled folder {folder_path}: {e}")

def schedule_processing(folder_path, watermark_text, schedule_time):
    dag_id = f"process_folder_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    default_args = {
        'owner': 'airflow',
        'depends_on_past': False,
        'start_date': datetime(2023, 1, 1),
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    }

    dag = DAG(
        dag_id=dag_id,
        default_args=default_args,
        description='A DAG to process a folder',
        schedule_interval=schedule_time,
    )

    process_folder_task = PythonOperator(
        task_id='process_folder',
        python_callable=process_scheduled_folder,
        op_args=[folder_path, watermark_text],
        dag=dag,
    )

    globals()[dag_id] = dag
    logging.info(f"Scheduled processing for folder {folder_path} at {schedule_time}")

# Example usage to schedule processing of a folder at a specific time
# schedule_processing('/path/to/folder', 'Eternal Elegance Emporium', '0 12 * * *')  # Daily at noon

# Advanced functions and features

# 1. Concurrent processing using ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor

def concurrent_post_listings(listings, marketplaces):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(post_listings_with_retry, [listing], marketplaces) for listing in listings]
        for future in futures:
            future.result()  # Wait for all futures to complete

def process_scheduled_folder_concurrently(folder_path, watermark_text):
    try:
        listings = listing_creation.process_main_folder(folder_path, watermark_text)
        if listings:
            marketplaces = settings["selected_marketplaces"]
            concurrent_post_listings(listings, marketplaces)
            logging.info(f"Successfully processed and posted listings from {folder_path}")
        else:
            logging.error(f"Failed to process listings from {folder_path}")
    except Exception as e:
        logging.error(f"Error processing scheduled folder {folder_path}: {e}")

# 2. Fetch and validate API documentation dynamically
def fetch_marketplace_api_docs():
    # Example function to fetch API docs
    api_docs = {
        "amazon": "https://api.amazon.com/documentation",
        "ebay": "https://api.ebay.com/documentation",
        "etsy": "https://api.etsy.com/documentation",
        # Add more marketplaces as needed
    }
    return api_docs

def validate_marketplace_apis():
    api_docs = fetch_marketplace_api_docs()
    for marketplace, doc_url in api_docs.items():
        try:
            response = requests.get(doc_url)
            response.raise_for_status()
            logging.info(f"{marketplace} API documentation fetched successfully")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch {marketplace} API documentation: {e}")

# 3. Enhanced error handling and notification
def notify_on_error(context):
    # Function to notify on error (e.g., send an email or Slack message)
    error_message = f"Task {context['task_instance_key_str']} failed"
    logging.error(error_message)
    # Add notification code here (e.g., send email or Slack message)

default_args.update({
    'on_failure_callback': notify_on_error,
})

# 4. Dynamic scheduling based on business hours
def dynamic_schedule():
    now = datetime.now()
    if now.weekday() < 5 and 9 <= now.hour < 17:  # Business hours (9 AM to 5 PM, Mon-Fri)
        return '*/15 * * * *'  # Every 15 minutes
    else:
        return '0 * * * *'  # Every hour

# 5. Integration with monitoring tools (e.g., Prometheus)
def setup_monitoring():
    # Example function to setup monitoring
    logging.info("Monitoring setup complete")

setup_monitoring()

# Refactored scheduling with dynamic interval
def schedule_dynamic_processing(folder_path, watermark_text):
    schedule_time = dynamic_schedule()
    schedule_processing(folder_path, watermark_text, schedule_time)
