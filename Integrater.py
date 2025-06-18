import json
import logging
from pathlib import Path
from datetime import datetime
from Scraper import EcommerceScraper
from Vector import ProductSeekerVectorDB

"""
Integrated script that combines web scraping with ProductSeeker Vector DB
This script scrapes products and immediately adds them to the vector database
"""

# Import the scraper (make sure the file is in the same directory)
# from ecommerce_scraper import EcommerceScraper

# Import ProductSeeker (make sure the file is in the same directory)
# from product_seeker import ProductSeekerVectorDB

# For demonstration - you'll need to uncomment the imports above
logger = logging.getLogger(__name__)


class IntegratedProductScraper:
    """
    Integrated scraper that combines web scraping with vector database storage
    """

    def __init__(self, url, scraper_output_dir, db_path, collection_name, model_name):

        # Initialize scraper
        self.scraper = EcommerceScraper(
            base_url=url,
            output_dir=scraper_output_dir
        )

        # Initialize ProductSeeker Vector DB
        self.db = ProductSeekerVectorDB(
            db_path=db_path,
            collection_name=collection_name,
            model_name=model_name
        )

        self.scraper_output_dir = Path(scraper_output_dir)

        logger.info("🚀 Integrated ProductScraper initialized")
        logger.info(f"📁 Scraper output: {scraper_output_dir}")
        logger.info(f"🗄️ Database path: {db_path}")

    def scrape_and_store(self,
                         max_products_per_category: int = 30,
                         batch_size: int = 50,
                         save_json: bool = True) -> dict:
        """
        Complete pipeline: Scrape products and store in vector database

        Args:
            max_products_per_category: Maximum products to scrape per category
            batch_size: Batch size for vector database insertion
            save_json: Whether to save JSON backup

        Returns:
            Dictionary with operation statistics
        """

        logger.info("🎯 Starting integrated scrape and store operation...")

        results = {
            'scraped_products': 0,
            'stored_products': 0,
            'failed_products': 0,
            'categories': 0,
            'images_downloaded': 0,
            'json_file': None,
            'database_stats': {},
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'status': 'running'
        }

        try:
            # Step 1: Scrape products
            logger.info("📥 Step 1: Scraping products from website...")
            products = self.scraper.scrape_all_products(max_products_per_category)

            if not products:
                logger.error("❌ No products were scraped!")
                results['status'] = 'failed'
                return results

            results['scraped_products'] = len(products)
            results['categories'] = len(set(p.get('category', 'Unknown') for p in products))
            results['images_downloaded'] = sum(1 for p in products if p.get('image_path'))

            logger.info(f"✅ Scraped {len(products)} products from {results['categories']} categories")

            # Step 2: Save JSON backup (optional)
            if save_json:
                logger.info("💾 Step 2: Saving JSON backup...")
                json_file = self.scraper.save_products(
                    products,
                    f"products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                results['json_file'] = json_file
                logger.info(f"✅ JSON backup saved: {json_file}")

            # Step 3: Store in vector database
            logger.info("🗄️ Step 3: Adding products to vector database...")
            db_stats = self.db.add_products(products, batch_size=batch_size)

            results['stored_products'] = db_stats.get('added', 0)
            results['failed_products'] = db_stats.get('failed', 0)

            logger.info(f"✅ Stored {results['stored_products']} products in vector database")

            # Step 4: Get database statistics
            results['database_stats'] = self.db.get_database_stats()

            results['status'] = 'completed'
            results['end_time'] = datetime.now().isoformat()

            # Print summary
            self._print_summary(results)

            return results

        except Exception as e:
            logger.error(f"❌ Operation failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            results['end_time'] = datetime.now().isoformat()
            return results

    def _print_summary(self, results: dict):
        """Print operation summary"""
        print("\n" + "=" * 70)
        print("🎉 INTEGRATED SCRAPING & STORAGE COMPLETED")
        print("=" * 70)
        print(f"📊 Products scraped: {results['scraped_products']}")
        print(f"🗄️ Products stored in DB: {results['stored_products']}")
        print(f"❌ Failed products: {results['failed_products']}")
        print(f"📂 Categories processed: {results['categories']}")
        print(f"🖼️ Images downloaded: {results['images_downloaded']}")

        if results.get('json_file'):
            print(f"💾 JSON backup: {results['json_file']}")

        db_stats = results.get('database_stats', {})
        if db_stats:
            print(f"🗄️ Total products in DB: {db_stats.get('total_products', 'Unknown')}")
            print(f"📊 Products with images: {db_stats.get('products_with_images', 'Unknown')}")

        print(f"⏱️ Started: {results['start_time']}")
        print(f"⏱️ Completed: {results['end_time']}")
        print("=" * 70)

    def search_products(self, query: str, n_results: int = 5, search_type: str = "text"):
        """
        Search products in the vector database

        Args:
            query: Search query (text) or path to image file
            n_results: Number of results to return
            search_type: "text" or "image"
        """

        logger.info(f"🔍 Searching for: '{query}' (type: {search_type})")

        try:
            if search_type == "text":
                results = self.db.search_by_text(query, n_results=n_results)
            elif search_type == "image":
                results = self.db.search_by_image(query, n_results=n_results)
            else:
                logger.error(f"Invalid search type: {search_type}")
                return None

            if results.get('error'):
                logger.error(f"Search failed: {results['error']}")
                return results

            # Print results
            print(f"\n🔍 Search Results for '{query}':")
            print("-" * 50)

            for i, result in enumerate(results['results'], 1):
                metadata = result['metadata']
                print(f"{i}. {metadata['title']}")
                print(f"   💰 Price: {metadata['price']}")
                print(f"   📂 Category: {metadata['category']}")
                print(f"   🏷️ Brand: {metadata['brand']}")
                print(f"   📊 Similarity: {result['similarity']:.3f} ({result['confidence']})")

                if metadata.get('description'):
                    desc = metadata['description'][:100] + "..." if len(metadata['description']) > 100 else metadata[
                        'description']
                    print(f"   📝 Description: {desc}")

                if metadata.get('url'):
                    print(f"   🔗 URL: {metadata['url']}")

                print()

            return results

        except Exception as e:
            logger.error(f"Search error: {e}")
            return {'error': str(e), 'results': [], 'count': 0}

    def interactive_search(self):
        """Interactive search interface"""
        print("\n🔍 Interactive Product Search")
        print("Type 'quit' to exit")
        print("-" * 30)

        while True:
            try:
                query = input("\n💭 Enter search query: ").strip()

                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                if not query:
                    continue

                # Check if it's an image file path
                if Path(query).exists() and Path(query).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    self.search_products(query, search_type="image")
                else:
                    self.search_products(query, search_type="text")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Search error: {e}")

    def get_statistics(self):
        """Get comprehensive statistics"""
        db_stats = self.db.get_database_stats()

        # Additional file system stats
        images_dir = self.scraper_output_dir / "images"
        image_count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0

        stats = {
            'database': db_stats,
            'files': {
                'images_downloaded': image_count,
                'scraper_output_dir': str(self.scraper_output_dir),
                'images_dir': str(images_dir)
            }
        }

        return stats


SCRAPER_OUTPUT = "D:/Vector/ProductSeeker_db"  # Where to save scraped files
DATABASE_PATH = "D:/Vector/ProductSeeker_data"  # Where to store vector database
COLLECTION_NAME = "ecommerce_test"  # Database collection name
URL = "https://books.toscrape.com/"
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

# def main():
#     """Main function demonstrating the integrated scraper"""
#     # Configure custom paths
#     SCRAPER_OUTPUT = "D:/Vector/ProductSeeker_db"  # Where to save scraped files
#     DATABASE_PATH = "D:/Vector/ProductSeeker_data"  # Where to store vector database
#     COLLECTION_NAME = "ecommerce_test"  # Database collection name
#     URL = "https://books.toscrape.com/"
#     MODEL_NAME = "clip-ViT-B-32"
#     # Initialize integrated scraper
#     scraper = IntegratedProductScraper(
#         scraper_output_dir=SCRAPER_OUTPUT,
#         db_path=DATABASE_PATH,
#         collection_name=COLLECTION_NAME,
#         url=URL,
#         model_name=MODEL_NAME
#     )
#
#     # Run the complete pipeline
#     print("🚀 Starting integrated scraping and storage...")
#     results = scraper.scrape_and_store(
#         max_products_per_category=30,  # Limit products per category
#         batch_size=50,  # Batch size for DB insertion
#         save_json=True  # Save JSON backup
#     )
#
#     if results['status'] == 'completed':
#         print("\n✅ Operation completed successfully!")
#
#         # Show some example searches
#         print("\n🔍 Testing some searches...")
#
#         # Example searches
#         search_queries = [
#             "laptop gaming",
#             "phone",
#             "tablet",
#             "notebook computer"
#         ]
#
#         for query in search_queries:
#             print(f"\n--- Searching for: '{query}' ---")
#             results = scraper.search_products(query, n_results=3)
#             if results and results.get('count', 0) > 0:
#                 print(f"Found {results['count']} results")
#             else:
#                 print("No results found")
#
#         # Start interactive search
#         choice = input("\n🤔 Start interactive search? (y/n): ").strip().lower()
#         if choice in ['y', 'yes']:
#             scraper.interactive_search()
#
#     else:
#         print(f"❌ Operation failed: {results.get('error', 'Unknown error')}")
#
#
# if __name__ == "__main__":
#     main()
