from Integrater import IntegratedProductScraper

SCRAPER_OUTPUT = "D:/Vector/ProductSeeker_db"  # Where to save scraped files
DATABASE_PATH = "D:/Vector/ProductSeeker_data"  # Where to store vector database
COLLECTION_NAME = "ecommerce_test"  # Database collection name
URL = "https://webscraper.io/test-sites/e-commerce/allinone"
MODEL_NAME = "clip-ViT-B-32"


def scrape_and_parse():
    # Initialize integrated scraper
    scraper = IntegratedProductScraper(
        url=URL,
        scraper_output_dir=SCRAPER_OUTPUT,
        db_path=DATABASE_PATH,
        collection_name=COLLECTION_NAME,
        model_name=MODEL_NAME
    )

    # THIS IS WHAT YOU WERE MISSING - Actually run the scraping!
    print("🚀 Starting integrated scraping and storage...")
    results = scraper.scrape_and_store(
        max_products_per_category=30,  # Limit products per category
        batch_size=50,  # Batch size for DB insertion
        save_json=True  # Save JSON backup
    )

    if results['status'] == 'completed':
        print("\n✅ Operation completed successfully!")
        print(f"📊 Scraped and stored {results['stored_products']} products")

        # Optional: Test some searches
        print("\n🔍 Testing search functionality...")
        search_results = scraper.search_products("laptop", n_results=3)

        # Optional: Start interactive search
        choice = input("\n🤔 Start interactive search? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            scraper.interactive_search()

    else:
        print(f"❌ Operation failed: {results.get('error', 'Unknown error')}")
        print("Check the logs for more details.")
