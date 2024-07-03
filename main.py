import sys
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
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import joblib
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QProgressBar,
                             QTextEdit, QMessageBox, QCheckBox, QInputDialog, QFormLayout, QLineEdit, QAction)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon
from scripts import image_processing, api_requests, listing_creation, schedule_tasks, social_media_posting, backup_restore
from logging.handlers import RotatingFileHandler
import time
import shutil
from calendar_ui import CalendarUI  # Import the calendar UI

# Configure logging with log rotation
log_handler = RotatingFileHandler('logs/app.log', maxBytes=5*1024*1024, backupCount=3)
log_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)
logging.getLogger().addHandler(log_handler)

progress_log_handler = RotatingFileHandler('logs/progress.log', maxBytes=5*1024*1024, backupCount=3)
progress_log_handler.setLevel(logging.INFO)
progress_log_handler.setFormatter(formatter)
progress_logger = logging.getLogger('progress')
progress_logger.addHandler(progress_log_handler)

# Load settings and policies
def load_settings():
    with open('config/settings.json', 'r') as f:
        return json.load(f)

def load_policies():
    with open('config/policies.json', 'r') as f:
        return json.load(f)

settings = load_settings()
policies = load_policies()

# Check required settings
required_settings = ["watermark_text", "currency", "ebay_api"]
def check_settings(settings, required_keys):
    missing_keys = [key for key in required_keys if key not in settings]
    if missing_keys:
        logging.error(f"Missing required settings: {', '.join(missing_keys)}")
        sys.exit(1)

check_settings(settings, required_settings)

# Load pre-trained models
tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# Load or train price prediction model
price_model_path = 'models/price_prediction_model.pkl'
if os.path.exists(price_model_path):
    price_model = joblib.load(price_model_path)
else:
    price_model = LinearRegression()

# Utility functions
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

def get_with_retry(url, headers, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                return response
            logging.warning(f"Attempt {attempt+1} failed: {response.status_code} {response.content}")
        except requests.RequestException as e:
            logging.error(f"Request error: {e}")
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    return None

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
    response = get_with_retry(url, headers, params)
    if response and response.status_code == 200:
        items = response.json().get('itemSummaries', [])
        prices = [item['price']['value'] for item in items if 'price' in item]
        if prices:
            return median(prices)
    logging.error(f"Failed to get sold prices for {title}: {response.content}")
    return None

def safe_open(file_path, mode):
    try:
        return open(file_path, mode)
    except IOError as e:
        logging.error(f"Failed to open file {file_path}: {e}")
        return None

def compress_image(image_path, quality=85):
    try:
        img = Image.open(image_path)
        img.save(image_path, "JPEG", quality=quality)
        logging.info(f"Compressed image {image_path}")
    except Exception as e:
        logging.error(f"Error compressing image {image_path}: {e}")

def create_listing_data(folder_path, sku, keywords):
    try:
        title = generate_seo_title(keywords)
        description = generate_seo_description(keywords)
        median_price = get_ebay_sold_prices(title)
        if median_price is None:
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

        with safe_open(os.path.join(folder_path, 'keywords.txt'), 'w') as f:
            if f:
                f.write("\n".join(keywords))

        with safe_open(os.path.join(folder_path, 'base64.txt'), 'w') as f:
            if f:
                for image_path in processed_images:
                    base64_image = image_processing.encode_image_to_base64(image_path)
                    f.write(f"{os.path.basename(image_path)}: {base64_image}\n")

        thumbnails = image_processing.generate_thumbnails(processed_images)
        for thumb in thumbnails:
            os.makedirs(os.path.join(folder_path, 'thumbnails'), exist_ok=True)
            os.rename(thumb, os.path.join(folder_path, 'thumbnails', os.path.basename(thumb)))

        # Compress images after processing and copying for base64 extraction
        for image_path in processed_images:
            compress_image(image_path)

        if listing_data:
            with safe_open(os.path.join(folder_path, 'listing.txt'), 'w') as f:
                if f:
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

def save_processed_listings(listings, output_dir):
    listings_file = os.path.join(output_dir, 'processed_listings.json')
    with safe_open(listings_file, 'w') as f:
        if f:
            json.dump(listings, f, indent=4)
        logging.info(f"Processed listings saved to {listings_file}")

def process_main_folder(main_folder_path, watermark_text, progress_callback=None, is_running=lambda: True):
    listings = []
    try:
        skus = {}
        output_dir = create_output_directory()
        with safe_open(os.path.join(output_dir, 'sku_tracking.csv'), 'w', newline='') as csvfile:
            if csvfile:
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
                        if not is_running():
                            break
                        result = future.result()
                        if result:
                            listings.append(result)
                        if progress_callback:
                            progress_callback(i + 1, len(futures))
                    
                    for sku, folder in skus.items():
                        writer.writerow({'SKU': sku, 'Folder': folder})

        save_processed_listings(listings, output_dir)
        return listings
    except Exception as e:
        logging.error(f"Error processing main folder {main_folder_path}: {e}")
        return listings

# Posting functions for various marketplaces
def post_listing(listing_data, marketplace):
    post_functions = {
        "eBay": api_requests.post_to_ebay,
        "Etsy": api_requests.post_to_etsy,
        "Facebook Marketplace": api_requests.post_to_facebook,
        "Instagram": api_requests.post_to_instagram,
        "Mercari": api_requests.post_to_mercari,
        "Poshmark": api_requests.post_to_poshmark,
        "OfferUp": api_requests.post_to_offerup,
        "Pinterest": api_requests.post_to_pinterest,
        "Twitter": api_requests.post_to_twitter
    }
    try:
        post_functions[marketplace](listing_data)
        logging.info(f"Posted {listing_data['title']} to {marketplace}")
    except Exception as e:
        logging.error(f"Failed to post {listing_data['title']} to {marketplace}: {e}")
        backup_file = os.path.join(os.path.dirname(listing_data['images'][0]), f"{marketplace}_backup.json")
        with safe_open(backup_file, 'w') as f:
            if f:
                json.dump(listing_data, f)

def post_listings_to_marketplaces(listings, marketplaces):
    for listing in listings:
        for marketplace in marketplaces:
            post_listing(listing, marketplace)

# PyQt5 GUI Application
class ProcessingThread(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(list)

    def __init__(self, folder_path, watermark_text, output_dir, settings):
        super().__init__()
        self.folder_path = folder_path
        self.watermark_text = watermark_text
        self.output_dir = output_dir
        self.settings = settings
        self._is_running = True

    def run(self):
        listings = process_main_folder(self.folder_path, self.watermark_text, self.update_progress, self.is_running)
        if self._is_running:
            self.finished.emit(listings)

    def stop(self):
        self._is_running = False

    def is_running(self):
        return self._is_running

    def update_progress(self, current, total):
        self.progress.emit(current, total)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Multi-Marketplace Listing Tool')
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        self.layout = QVBoxLayout()
        central_widget.setLayout(self.layout)

        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        # Settings action
        settings_action = QAction('Settings', self)
        settings_action.triggered.connect(self.open_settings)
        file_menu.addAction(settings_action)

        # Export settings action
        export_action = QAction('Export Settings', self)
        export_action.triggered.connect(self.export_settings)
        file_menu.addAction(export_action)

        # Import settings action
        import_action = QAction('Import Settings', self)
        import_action.triggered.connect(self.import_settings)
        file_menu.addAction(import_action)

        # Export logs action
        export_logs_action = QAction('Export Logs', self)
        export_logs_action.triggered.connect(self.export_logs)
        file_menu.addAction(export_logs_action)

        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help action
        help_menu = menubar.addMenu('Help')
        help_action = QAction('Help', self)
        help_action.triggered.connect(self.open_help)
        help_menu.addAction(help_action)

        # Label
        self.label = QLabel('Multi-Marketplace Listing Tool', self)
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        # Buttons
        self.process_button = QPushButton('Select Folder for Processing', self)
        self.process_button.clicked.connect(self.select_folder)
        self.layout.addWidget(self.process_button)

        self.schedule_button = QPushButton('Schedule Folder Processing', self)
        self.schedule_button.clicked.connect(self.schedule_processing)
        self.layout.addWidget(self.schedule_button)

        self.cancel_button = QPushButton('Cancel Processing', self)
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.layout.addWidget(self.cancel_button)

        self.error_button = QPushButton('View Error Dashboard', self)
        self.error_button.clicked.connect(self.view_errors)
        self.layout.addWidget(self.error_button)

        self.marketplace_button = QPushButton('Select Marketplaces', self)
        self.marketplace_button.clicked.connect(self.select_marketplaces)
        self.layout.addWidget(self.marketplace_button)

        self.social_media_button = QPushButton('Post to Social Media', self)
        self.social_media_button.clicked.connect(self.post_to_social_media)
        self.layout.addWidget(self.social_media_button)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.progress_bar)

        # Timer for real-time updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_real_time)
        self.timer.start(1000)  # Update every second

        # Drag and drop area
        self.drag_drop_area = QLabel('Drag and Drop Folders Here', self)
        self.drag_drop_area.setAlignment(Qt.AlignCenter)
        self.drag_drop_area.setStyleSheet("QLabel { border: 2px dashed #aaa; }")
        self.drag_drop_area.setFixedHeight(200)
        self.drag_drop_area.setAcceptDrops(True)
        self.layout.addWidget(self.drag_drop_area)

        # Event handlers for drag and drop
        self.drag_drop_area.dragEnterEvent = self.drag_enter_event
        self.drag_drop_area.dragMoveEvent = self.drag_move_event
        self.drag_drop_area.dropEvent = self.drop_event

        # Dark mode toggle
        self.dark_mode_checkbox = QCheckBox('Enable Dark Mode', self)
        self.dark_mode_checkbox.stateChanged.connect(self.toggle_dark_mode)
        self.layout.addWidget(self.dark_mode_checkbox)

        # Calendar button
        self.calendar_button = QPushButton('Open Calendar Scheduler', self)
        self.calendar_button.clicked.connect(self.open_calendar)
        self.layout.addWidget(self.calendar_button)

    def drag_enter_event(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def drag_move_event(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def drop_event(self, event):
        for url in event.mimeData().urls():
            folder_path = url.toLocalFile()
            if os.path.isdir(folder_path):
                self.process_folder(folder_path)

    def toggle_dark_mode(self, state):
        if state == Qt.Checked:
            self.setStyleSheet("QWidget { background-color: #2e2e2e; color: #ffffff; }")
        else:
            self.setStyleSheet("")

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select the main folder containing product folders")
        if not folder_path:
            return

        self.progress_bar.setValue(0)
        self.output_dir = create_output_directory()
        self.thread = ProcessingThread(folder_path, settings["watermark_text"], self.output_dir, settings)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.processing_finished)
        self.thread.start()

    def update_progress(self, current, total):
        self.progress_bar.setValue(int((current / total) * 100))

    def processing_finished(self, listings):
        if listings:
            QMessageBox.information(self, "Processing Complete", f"Processed {len(listings)} products successfully!")
            next_message = "Do you want to proceed to schedule the listings?"
            self.prompt_next_step(next_message, self.schedule_listings)
        else:
            QMessageBox.critical(self, "Processing Error", "An error occurred during processing. Check logs for details.")

    def cancel_processing(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
            logging.info("Processing cancelled.")
            QMessageBox.information(self, "Cancelled", "Processing has been cancelled.")

    def schedule_processing(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select the main folder containing product folders to schedule processing")
        if not folder_path:
            return
        
        schedule_time, ok = QInputDialog.getText(self, "Schedule Processing", "Enter time to schedule (HH:MM):")
        if ok and schedule_time:
            schedule_tasks.schedule_processing(folder_path, settings["watermark_text"], schedule_time)
            QMessageBox.information(self, "Scheduled", "Folder processing has been scheduled.")
            next_message = "Do you want to proceed to social media posting?"
            self.prompt_next_step(next_message, self.post_to_social_media)

    def post_to_social_media(self):
        listings = listing_creation.load_listings()
        if listings:
            social_media_posting.post_listings(listings, settings)
            QMessageBox.information(self, "Social Media Posting", "Listings posted to social media successfully!")
        else:
            QMessageBox.critical(self, "No Listings", "No listings found to post to social media.")

    def open_settings(self):
        settings_win = QWidget()
        settings_win.setWindowTitle("Settings")
        settings_layout = QFormLayout()

        self.watermark_text_input = QLineEdit(settings["watermark_text"])
        settings_layout.addRow("Watermark Text:", self.watermark_text_input)

        self.currency_input = QLineEdit(settings["currency"])
        settings_layout.addRow("Currency:", self.currency_input)

        self.ebay_api_input = QLineEdit(settings.get("ebay_api", {}).get("auth_token", ""))
        settings_layout.addRow("eBay API Token:", self.ebay_api_input)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        settings_layout.addWidget(save_button)

        settings_win.setLayout(settings_layout)
        settings_win.show()

    def save_settings(self):
        settings["watermark_text"] = self.watermark_text_input.text()
        settings["currency"] = self.currency_input.text()
        if not settings["watermark_text"] or not settings["currency"]:
            QMessageBox.critical(self, "Invalid Settings", "Watermark text and currency cannot be empty.")
            return
        settings["ebay_api"] = {"auth_token": self.ebay_api_input.text()}
        with open('config/settings.json', 'w') as f:
            json.dump(settings, f, indent=4)
        QMessageBox.information(self, "Settings Saved", "Settings have been saved successfully.")

    def export_settings(self):
        file_path = QFileDialog.getSaveFileName(self, "Export Settings", filter="JSON files (*.json)")[0]
        if file_path:
            backup_restore.export_settings(file_path)
            QMessageBox.information(self, "Export Complete", "Settings have been exported successfully.")

    def import_settings(self):
        file_path = QFileDialog.getOpenFileName(self, "Import Settings", filter="JSON files (*.json)")[0]
        if file_path:
            backup_restore.import_settings(file_path)
            QMessageBox.information(self, "Import Complete", "Settings have been imported successfully.")

    def export_logs(self):
        logs_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Logs")
        if logs_dir:
            for log_file in os.listdir('logs'):
                full_log_path = os.path.join('logs', log_file)
                if os.path.isfile(full_log_path):
                    shutil.copy(full_log_path, logs_dir)
            QMessageBox.information(self, "Export Complete", "Logs have been exported successfully.")

    def open_help(self):
        help_win = QWidget()
        help_win.setWindowTitle("Help")
        help_text = """
        Multi-Marketplace Listing Tool Help
        -----------------------------------

        1. Select Folder for Processing:
           Select a folder containing product images for processing.

        2. Schedule Folder Processing:
           Schedule automatic processing of a folder at a specific time.

        3. View Error Dashboard:
           View the error log and processing status.

        4. Settings:
           Configure application settings such as watermark text, preferred language, theme, and currency.

        5. Export Settings:
           Export your current settings to a file.

        6. Import Settings:
           Import settings from a file.

        If you need further assistance, please contact support.
        """
        help_text_widget = QTextEdit(help_win)
        help_text_widget.setText(help_text)
        help_text_widget.setReadOnly(True)
        layout = QVBoxLayout()
        layout.addWidget(help_text_widget)
        help_win.setLayout(layout)
        help_win.show()

    def select_marketplaces(self):
        market_win = QWidget()
        market_win.setWindowTitle("Select Marketplaces")

        layout = QVBoxLayout()

        marketplaces = ["eBay", "Etsy", "Facebook Marketplace", "Instagram", "Mercari", "Poshmark", "OfferUp", "Pinterest", "Twitter"]
        self.marketplace_checkboxes = []

        for marketplace in marketplaces:
            checkbox = QCheckBox(marketplace)
            layout.addWidget(checkbox)
            self.marketplace_checkboxes.append(checkbox)

        select_button = QPushButton("Select", self)
        select_button.clicked.connect(self.save_marketplace_selection)
        layout.addWidget(select_button)

        market_win.setLayout(layout)
        market_win.show()

    def save_marketplace_selection(self):
        selected_marketplaces = [checkbox.text() for checkbox in self.marketplace_checkboxes if checkbox.isChecked()]
        settings["selected_marketplaces"] = selected_marketplaces
        with open('config/settings.json', 'w') as f:
            json.dump(settings, f, indent=4)
        QMessageBox.information(self, "Selection Saved", "Selected marketplaces have been saved.")

    def view_errors(self):
        error_log_win = QWidget()
        error_log_win.setWindowTitle("Error Dashboard")
        log_text = QTextEdit(error_log_win)
        with open('logs/app.log', 'r') as f:
            log_text.setText(f.read())
        log_text.setReadOnly(True)
        layout = QVBoxLayout()
        layout.addWidget(log_text)
        error_log_win.setLayout(layout)
        error_log_win.show()

    def prompt_next_step(self, message, next_step_function):
        reply = QMessageBox.question(self, "Next Step", message, QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            next_step_function()

    def update_real_time(self):
        # Placeholder for real-time updates
        pass

    def open_calendar(self):
        self.calendar_ui = CalendarUI()
        self.calendar_ui.exec_()

def main():
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
