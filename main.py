from Integrater import IntegratedProductScraper

SCRAPER_OUTPUT = "D:/Vector/ProductSeeker_db"  # Where to save scraped files
DATABASE_PATH = "D:/Vector/ProductSeeker_data"  # Where to store vector database
COLLECTION_NAME = "ecommerce_test"  # Database collection name
URL = "https://webscraper.io/test-sites/e-commerce/allinone"
MODEL_NAME = "clip-ViT-B-32"

# Initialize integrated scraper
scraper = IntegratedProductScraper(
    url=URL,
    scraper_output_dir=SCRAPER_OUTPUT,
    db_path=DATABASE_PATH,
    collection_name=COLLECTION_NAME,
    model_name=MODEL_NAME

)
