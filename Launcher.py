#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from LangGraphProductSearchSystem import LangGraphProductSearcher

"""
Complete Product Search System Launcher
Combines scraping, LangGraph integration, and image search bot
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Configuration
SCRAPER_OUTPUT = "D:/Vector/ProductSeeker_db"
DATABASE_PATH = "D:/Vector/ProductSeeker_data"
COLLECTION_NAME = "ecommerce_test"
URL = "https://books.toscrape.com/"
MODEL_NAME = "clip-ViT-B-32"


def setup_environment():
    """Setup environment and check dependencies"""
    try:
        # Check if required modules exist
        required_modules = [
            'Integrater',
            'LangGraphProductSearchSystem',
            'ImageSearchBot',
            'Vector',
            'Scraper'
        ]

        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)

        if missing_modules:
            logger.error(f"Missing modules: {missing_modules}")
            return False

        return True

    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        return False


def run_scraper():
    """Run the web scraper to populate database"""
    logger.info("🚀 Starting web scraper...")

    try:
        from Integrater import IntegratedProductScraper
        # Initialize and run scraper
        scraper = IntegratedProductScraper(
            url=URL,
            scraper_output_dir=SCRAPER_OUTPUT,
            db_path=DATABASE_PATH,
            collection_name=COLLECTION_NAME,
            model_name=MODEL_NAME
        )

        print("📥 Scraping products from website...")
        results = scraper.scrape_and_store(
            max_products_per_category=30,
            batch_size=50,
            save_json=True
        )

        if results['status'] == 'completed':
            print(f"✅ Successfully scraped and stored {results['stored_products']} products")
            return True
        else:
            print(f"❌ Scraping failed: {results.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        logger.error(f"Scraper failed: {e}")
        return False


def run_langgraph_system():
    """Test the LangGraph system"""
    logger.info("🤖 Testing LangGraph system...")

    try:
        # Initialize system
        search_system = LangGraphProductSearcher(
            db_path=DATABASE_PATH,
            collection_name=COLLECTION_NAME,
            model_name=MODEL_NAME
        )
        # Test searches
        test_queries = [
            "gaming laptop",
            "smartphone android",
            "wireless headphones"
        ]
        print("🔍 Testing LangGraph searches...")
        for query in test_queries:
            print(f"\n--- Testing: '{query}' ---")

            # Use the correct method signature - only query and search_type
            result = search_system.search(query, search_type="auto")

            if result['success']:
                results_count = len(result.get('results', []))
                print(f"✅ Found {results_count} results")
                print(f"🔄 Refinements made: {result.get('refinement_count', 0)}")

                # Show the AI response messages
                messages = result.get('messages', [])
                for message in messages:
                    if hasattr(message, 'content') and message.content:
                        # Show first 200 characters of the response
                        content = message.content[:200] + "..." if len(message.content) > 200 else message.content
                        print(f"🤖 Response preview: {content}")
                        break
            else:
                print(f"❌ Search failed: {result.get('error')}")

        return True

    except Exception as e:
        logger.error(f"LangGraph system failed: {e}")
        return False


def run_image_bot(interface="streamlit"):
    """Run the image search bot"""
    logger.info(f"🖼️ Starting image search bot ({interface} interface)...")

    try:
        from ImageSearchBot import ImageSearchBot

        # Initialize bot
        bot = ImageSearchBot(
            db_path=DATABASE_PATH,
            collection_name=COLLECTION_NAME,
            model_name=MODEL_NAME
        )

        if interface == "streamlit":
            print("🌐 Starting Streamlit web interface...")
            print("📱 Open your browser to: http://localhost:8501")
            bot.run_streamlit_app()
        else:
            print("💻 Starting console interface...")
            bot.run_console_interface()

        return True

    except Exception as e:
        logger.error(f"Image bot failed: {e}")
        return False


def check_database_status():
    """Check database status and statistics"""
    logger.info("📊 Checking database status...")

    try:
        from Vector import ProductSeekerVectorDB

        db = ProductSeekerVectorDB(
            db_path=DATABASE_PATH,
            collection_name=COLLECTION_NAME,
            model_name=MODEL_NAME
        )

        stats = db.get_database_stats()

        print("📊 Database Statistics:")
        print(f"   Total Products: {stats.get('total_products', 0)}")
        print(f"   Products with Images: {stats.get('products_with_images', 0)}")
        print(f"   Database Path: {DATABASE_PATH}")
        print(f"   Collection: {COLLECTION_NAME}")

        if stats.get('total_products', 0) == 0:
            print("⚠️  Database is empty - run scraper first!")
            return False
        else:
            print("✅ Database is ready for searches!")
            return True

    except Exception as e:
        logger.error(f"Database check failed: {e}")
        return False


def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Product Search System - Complete AI-powered e-commerce search solution"
    )

    parser.add_argument(
        'command',
        choices=['scrape', 'langgraph', 'bot', 'console-bot', 'status', 'full-setup'],
        help='Command to execute'
    )

    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip environment checks'
    )

    args = parser.parse_args()

    print("🔍 Product Search System Launcher")
    print("=" * 50)

    # Environment setup
    if not args.skip_checks:
        print("🔧 Checking environment...")
        if not setup_environment():
            print("❌ Environment check failed!")
            sys.exit(1)
        print("✅ Environment ready!")

    # Execute command
    if args.command == 'status':
        check_database_status()

    elif args.command == 'scrape':
        success = run_scraper()
        if not success:
            sys.exit(1)

    elif args.command == 'langgraph':
        # Check database first
        if not check_database_status():
            print("❌ Database not ready - run 'scrape' first!")
            sys.exit(1)

        success = run_langgraph_system()
        if not success:
            sys.exit(1)

    elif args.command == 'bot':
        # Check database first
        if not check_database_status():
            print("❌ Database not ready - run 'scrape' first!")
            sys.exit(1)

        success = run_image_bot("streamlit")
        if not success:
            sys.exit(1)

    elif args.command == 'console-bot':
        # Check database first
        if not check_database_status():
            print("❌ Database not ready - run 'scrape' first!")
            sys.exit(1)

        success = run_image_bot("console")
        if not success:
            sys.exit(1)

    elif args.command == 'full-setup':
        print("🚀 Running complete setup...")

        # Step 1: Scrape
        print("\n📥 Step 1: Scraping products...")
        if not run_scraper():
            print("❌ Scraping failed!")
            sys.exit(1)

        # Step 2: Test LangGraph
        print("\n🤖 Step 2: Testing LangGraph system...")
        if not run_langgraph_system():
            print("❌ LangGraph test failed!")
            sys.exit(1)

        # Step 3: Final status check
        print("\n📊 Step 3: Final status check...")
        if not check_database_status():
            print("❌ Database not ready!")
            sys.exit(1)

        print("\n🎉 Complete setup finished successfully!")
        print("Now you can run:")
        print("  python launcher.py bot          # Web interface")
        print("  python launcher.py console-bot  # Console interface")


if __name__ == "__main__":
    sys.argv = ['Launcher.py', 'console-bot']
    main()
# 'scrape', 'langgraph', 'bot', 'console-bot', 'status', 'full-setup', --console
