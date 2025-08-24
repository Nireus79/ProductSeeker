#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

# Import SocraticGenProductSeeker
try:
    from SocraticGenProductSeeker import (
        main as product_seeker_main,
        run_tests as product_seeker_tests,
        ProductInput,
        ProductMatch,
        run_pipeline_sync
    )

    SOCRATIC_AVAILABLE = True
except ImportError as e:
    SOCRATIC_AVAILABLE = False
    print(f"Warning: SocraticGenProductSeeker not available: {e}")

try:
    from LangGraphProductSearchSystem import LangGraphProductSearcher

    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    print(f"Warning: LangGraphProductSearchSystem not available: {e}")

"""
Fixed Product Search System Launcher
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SCRAPER_OUTPUT = "D:/Vector/ProductSeeker_db"
DATABASE_PATH = "D:/Vector/ProductSeeker_data"
COLLECTION_NAME = "ecommerce_test"
URL = "https://books.toscrape.com/"
MODEL_NAME = "clip-ViT-B-32"


def safe_run_socratic_pipeline(input_data):
    """Safe wrapper for Socratic pipeline that handles the AddableValuesDict issue"""
    try:
        result = run_pipeline_sync(input_data)

        # Handle different result types
        if hasattr(result, 'product_id'):
            return {
                'success': True,
                'product_id': result.product_id,
                'confidence': getattr(result, 'confidence', 0.0),
                'alternatives': getattr(result, 'alternatives', [])
            }
        elif hasattr(result, 'final_result'):
            final = result.final_result
            if hasattr(final, 'product_id'):
                return {
                    'success': True,
                    'product_id': final.product_id,
                    'confidence': getattr(final, 'confidence', 0.0),
                    'alternatives': getattr(final, 'alternatives', [])
                }
        elif isinstance(result, dict):
            return {
                'success': True,
                'product_id': result.get('product_id', 'unknown'),
                'confidence': result.get('confidence', 0.0),
                'alternatives': result.get('alternatives', [])
            }
        else:
            # Handle AddableValuesDict or other complex objects
            if hasattr(result, '__dict__'):
                result_dict = result.__dict__
                return {
                    'success': True,
                    'product_id': result_dict.get('product_id', 'unknown'),
                    'confidence': result_dict.get('confidence', 0.0),
                    'alternatives': result_dict.get('alternatives', [])
                }
            else:
                return {
                    'success': False,
                    'error': f'Unknown result format: {type(result)}'
                }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def run_scraper():
    """Run the web scraper to populate database"""
    logger.info("🚀 Starting web scraper...")

    try:
        from Integrater import IntegratedProductScraper

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


def check_database():
    """Simple database check"""
    try:
        from Vector import ProductSeekerVectorDB

        db = ProductSeekerVectorDB(
            db_path=DATABASE_PATH,
            collection_name=COLLECTION_NAME,
            model_name=MODEL_NAME
        )

        stats = db.get_database_stats()
        total_products = stats.get('total_products', 0)

        print(f"📊 Database has {total_products} products")

        if total_products == 0:
            print("⚠️ Database is empty - run scraper first!")
            return False

        return True

    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False


def test_langgraph():
    """Test LangGraph system with simple queries"""
    if not LANGGRAPH_AVAILABLE:
        print("❌ LangGraph system not available")
        return False

    try:
        searcher = LangGraphProductSearcher(
            db_path=DATABASE_PATH,
            collection_name=COLLECTION_NAME,
            model_name=MODEL_NAME
        )

        print("🤖 Testing LangGraph system...")

        # Try different search types to find what works
        search_types = ["vector", "semantic", "auto"]
        query = "book"

        for search_type in search_types:
            try:
                print(f"   Trying {search_type} search...")
                result = searcher.search(query, search_type=search_type)

                if result.get('success') and result.get('results'):
                    results_count = len(result['results'])
                    print(f"   ✅ {search_type.title()} search found {results_count} results")

                    # Show sample results
                    for i, product in enumerate(result['results'][:3], 1):
                        print(f"     {i}. {product.get('title', 'N/A')}")

                    return True
                else:
                    print(f"   ❌ {search_type.title()} search failed: {result.get('error', 'No results')}")

            except Exception as e:
                print(f"   ❌ {search_type.title()} search error: {e}")
                continue

        print("❌ All search types failed")
        return False

    except Exception as e:
        print(f"❌ LangGraph test failed: {e}")
        return False


def test_socratic():
    """Test Socratic system"""
    if not SOCRATIC_AVAILABLE:
        print("❌ Socratic system not available")
        return False

    try:
        print("🎯 Testing Socratic system...")

        # Test text query
        input_data = ProductInput(
            text_query="book",
            weights={"text": 1.0, "image": 0.0, "voice": 0.0}
        )

        result = safe_run_socratic_pipeline(input_data)

        if result['success']:
            print(f"✅ Socratic search successful")
            print(f"   Product ID: {result['product_id']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Alternatives: {len(result['alternatives'])}")
            return True
        else:
            print(f"❌ Socratic search failed: {result['error']}")
            return False

    except Exception as e:
        print(f"❌ Socratic test failed: {e}")
        return False


def run_interactive_search():
    """Interactive search using both systems"""
    print("\n🔍 Interactive Product Search")
    print("=" * 40)

    # Check which systems are available
    langgraph_ready = LANGGRAPH_AVAILABLE and check_database()
    socratic_ready = SOCRATIC_AVAILABLE

    if not langgraph_ready and not socratic_ready:
        print("❌ No search systems available!")
        return False

    print(f"Available systems:")
    if langgraph_ready:
        print("  🤖 LangGraph (Vector-based)")
    if socratic_ready:
        print("  🎯 Socratic (Multimodal)")

    while True:
        query = input("\n🔍 Enter search query (or 'quit' to exit): ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            break

        if not query:
            print("Please enter a search query")
            continue

        print(f"\nSearching for: '{query}'")
        print("-" * 30)

        # Try LangGraph
        if langgraph_ready:
            try:
                searcher = LangGraphProductSearcher(
                    db_path=DATABASE_PATH,
                    collection_name=COLLECTION_NAME,
                    model_name=MODEL_NAME
                )

                result = searcher.search(query, search_type="auto")

                if result.get('success') and result.get('results'):
                    print(f"🤖 LangGraph found {len(result['results'])} results:")
                    for i, product in enumerate(result['results'][:5], 1):
                        print(f"   {i}. {product.get('title', 'N/A')} - {product.get('price', 'N/A')}")
                else:
                    print(f"🤖 LangGraph: No results")

            except Exception as e:
                print(f"🤖 LangGraph error: {e}")

        # Try Socratic
        if socratic_ready:
            try:
                input_data = ProductInput(
                    text_query=query,
                    weights={"text": 1.0, "image": 0.0, "voice": 0.0}
                )

                result = safe_run_socratic_pipeline(input_data)

                if result['success']:
                    print(f"🎯 Socratic result:")
                    print(f"   Product: {result['product_id']}")
                    print(f"   Confidence: {result['confidence']:.2f}")
                else:
                    print(f"🎯 Socratic: {result['error']}")

            except Exception as e:
                print(f"🎯 Socratic error: {e}")

    print("👋 Goodbye!")
    return True


def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Fixed Product Search System")
    parser.add_argument(
        'command',
        choices=['scrape', 'test', 'search', 'langgraph', 'socratic', 'setup'],
        help='Command to execute',
        nargs='?',
        default='search'
    )

    args = parser.parse_args()

    print("🔍 Product Search System - Fixed Version")
    print("=" * 50)

    if args.command == 'scrape':
        if not run_scraper():
            sys.exit(1)

    elif args.command == 'test':
        print("🧪 Testing all systems...")

        # Check database
        db_ok = check_database()
        if not db_ok:
            print("Run 'scrape' first to populate database")
            sys.exit(1)

        # Test systems
        langgraph_ok = test_langgraph()
        socratic_ok = test_socratic()

        if langgraph_ok or socratic_ok:
            print("✅ At least one system working!")
        else:
            print("❌ No systems working!")
            sys.exit(1)

    elif args.command == 'langgraph':
        if not check_database():
            print("Run 'scrape' first!")
            sys.exit(1)
        if not test_langgraph():
            sys.exit(1)

    elif args.command == 'socratic':
        if not test_socratic():
            sys.exit(1)

    elif args.command == 'search':
        run_interactive_search()

    elif args.command == 'setup':
        print("🚀 Complete setup...")

        # Step 1: Scrape
        if not run_scraper():
            print("❌ Scraping failed!")
            sys.exit(1)

        # Step 2: Test
        if not check_database():
            print("❌ Database check failed!")
            sys.exit(1)

        # Step 3: Test systems
        langgraph_ok = test_langgraph()
        socratic_ok = test_socratic()

        if langgraph_ok or socratic_ok:
            print("✅ Setup complete!")
            print("Run: python Launcher.py search")
        else:
            print("❌ Setup failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()



