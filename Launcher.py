#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from LangGraphProductSearchSystem import LangGraphProductSearcher

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

"""
Complete Product Search System Launcher
Combines scraping, LangGraph integration, image search bot, and SocraticGenProductSeeker
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


def run_interactive_langgraph():
    """Run LangGraph system in interactive mode"""
    logger.info("🤖 Starting LangGraph interactive mode...")

    try:
        # Initialize system
        search_system = LangGraphProductSearcher(
            db_path=DATABASE_PATH,
            collection_name=COLLECTION_NAME,
            model_name=MODEL_NAME
        )

        print("\n=== LangGraph Product Search System ===")
        print("Enter 'quit' to exit")

        while True:
            query = input("\n🔍 Enter your search query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                break

            if not query:
                print("Please enter a search query.")
                continue

            print("Searching...")
            result = search_system.search(query, search_type="auto")

            if result['success']:
                results_count = len(result.get('results', []))
                print(f"✅ Found {results_count} results")
                print(f"🔄 Refinements made: {result.get('refinement_count', 0)}")

                # Show results
                results = result.get('results', [])
                for i, product in enumerate(results[:5], 1):  # Show top 5
                    print(f"\n{i}. {product.get('title', 'N/A')}")
                    print(f"   Price: {product.get('price', 'N/A')}")
                    if product.get('description'):
                        desc = product['description'][:100] + "..." if len(product['description']) > 100 else product[
                            'description']
                        print(f"   Description: {desc}")

                # Show AI response
                messages = result.get('messages', [])
                for message in messages:
                    if hasattr(message, 'content') and message.content:
                        print(f"\n🤖 AI Analysis: {message.content}")
                        break

            else:
                print(f"❌ Search failed: {result.get('error')}")

        print("👋 Goodbye!")
        return True

    except Exception as e:
        logger.error(f"Interactive LangGraph failed: {e}")
        return False


def run_socratic_seeker():
    """Run the SocraticGenProductSeeker"""
    logger.info("🎯 Starting Socratic Product Seeker...")

    if not SOCRATIC_AVAILABLE:
        print("❌ SocraticGenProductSeeker is not available!")
        print("Make sure SocraticGenProductSeeker.py is in the same directory.")
        return False

    try:
        # Run the main function from SocraticGenProductSeeker
        product_seeker_main()
        return True

    except Exception as e:
        logger.error(f"Socratic Product Seeker failed: {e}")
        return False


def run_socratic_tests():
    """Run SocraticGenProductSeeker tests"""
    logger.info("🧪 Running Socratic Product Seeker tests...")

    if not SOCRATIC_AVAILABLE:
        print("❌ SocraticGenProductSeeker is not available!")
        return False

    try:
        success = product_seeker_tests()
        if success:
            print("✅ All Socratic tests passed!")
        else:
            print("❌ Some Socratic tests failed!")
        return success

    except Exception as e:
        logger.error(f"Socratic tests failed: {e}")
        return False


def run_search_chooser():
    """Interactive chooser between search systems"""
    print("\n🎯 Product Search System Selector")
    print("=" * 40)
    print("Choose your search system:")
    print("1. LangGraph System (Vector DB based)")
    print("2. Socratic Product Seeker (Multimodal)")
    print("3. Both systems comparison")
    print("0. Return to main menu")

    while True:
        choice = input("\nSelect option (0-3): ").strip()

        if choice == '0':
            return True
        elif choice == '1':
            return run_interactive_langgraph()
        elif choice == '2':
            return run_socratic_seeker()
        elif choice == '3':
            return run_comparison_mode()
        else:
            print("Invalid choice. Please select 0-3.")


def run_comparison_mode():
    """Run both systems for comparison"""
    print("\n🔄 Comparison Mode")
    print("Enter a search query to test both systems:")

    query = input("Search query: ").strip()
    if not query:
        print("No query entered.")
        return False

    print(f"\n🔍 Testing query: '{query}'")
    print("=" * 50)

    # Test LangGraph system
    print("\n🤖 LangGraph System Results:")
    print("-" * 30)
    try:
        search_system = LangGraphProductSearcher(
            db_path=DATABASE_PATH,
            collection_name=COLLECTION_NAME,
            model_name=MODEL_NAME
        )
        result = search_system.search(query, search_type="auto")

        if result['success']:
            results_count = len(result.get('results', []))
            print(f"✅ Found {results_count} results")

            # Show top 3 results
            results = result.get('results', [])
            for i, product in enumerate(results[:3], 1):
                print(f"{i}. {product.get('title', 'N/A')} - {product.get('price', 'N/A')}")
        else:
            print(f"❌ Search failed: {result.get('error')}")

    except Exception as e:
        print(f"❌ LangGraph error: {e}")

    # Test Socratic system (with text-only input)
    print("\n🎯 Socratic System Results:")
    print("-" * 30)
    if SOCRATIC_AVAILABLE:
        try:
            input_data = ProductInput(
                text_query=query,
                weights={"text": 1.0, "image": 0.0, "voice": 0.0}
            )
            result = run_pipeline_sync(input_data)
            print(f"✅ Best match: {result.product_id}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Alternatives: {len(result.alternatives)}")

        except Exception as e:
            print(f"❌ Socratic error: {e}")
    else:
        print("❌ Socratic system not available")

    return True


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


def show_system_status():
    """Show status of all systems"""
    print("\n🔧 System Status Check")
    print("=" * 30)

    # Check database
    print("📊 Database Status:")
    db_ready = check_database_status()

    # Check Socratic system
    print(f"\n🎯 Socratic System: {'✅ Available' if SOCRATIC_AVAILABLE else '❌ Not Available'}")

    # Check LangGraph system
    try:
        search_system = LangGraphProductSearcher(
            db_path=DATABASE_PATH,
            collection_name=COLLECTION_NAME,
            model_name=MODEL_NAME
        )
        print("🤖 LangGraph System: ✅ Available")
    except Exception as e:
        print(f"🤖 LangGraph System: ❌ Error - {e}")

    return db_ready


def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Product Search System - Complete AI-powered e-commerce search solution"
    )

    parser.add_argument(
        'command',
        choices=[
            'scrape', 'langgraph', 'langgraph-interactive', 'socratic', 'socratic-test',
            'search', 'compare', 'bot', 'console-bot', 'status', 'full-setup', 'menu'
        ],
        help='Command to execute',
        nargs='?',  # Make command optional
        default='menu'  # Default to menu instead of None
    )

    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip environment checks'
    )

    args = parser.parse_args()

    print("🔍 Product Search System Launcher")
    print("=" * 50)
    print(f"🎯 Socratic System: {'Available' if SOCRATIC_AVAILABLE else 'Not Available'}")

    # If no command provided or menu command, show interactive menu
    if args.command == 'menu':
        print("\nAvailable commands:")
        print("  scrape              - Run web scraper")
        print("  langgraph           - Test LangGraph system")
        print("  langgraph-interactive - Interactive LangGraph mode")
        print("  socratic            - Run Socratic Product Seeker")
        print("  socratic-test       - Test Socratic system")
        print("  search              - Choose search system")
        print("  compare             - Compare both systems")
        print("  bot                 - Image search bot (web)")
        print("  console-bot         - Image search bot (console)")
        print("  status              - System status check")
        print("  full-setup          - Complete setup")

        command = input("\nEnter command: ").strip().lower()
        if not command:
            sys.exit(0)

        # Validate the entered command
        valid_commands = [
            'scrape', 'langgraph', 'langgraph-interactive', 'socratic', 'socratic-test',
            'search', 'compare', 'bot', 'console-bot', 'status', 'full-setup'
        ]

        if command not in valid_commands:
            print(f"❌ Invalid command: {command}")
            print(f"Valid commands: {', '.join(valid_commands)}")
            sys.exit(1)

        args.command = command

    # Environment setup
    if not args.skip_checks and args.command not in ['socratic', 'socratic-test', 'status', 'menu']:
        print("🔧 Checking environment...")
        if not setup_environment():
            print("❌ Environment check failed!")
            if args.command not in ['socratic', 'socratic-test']:
                sys.exit(1)
        print("✅ Environment ready!")

    # Execute command
    if args.command == 'status':
        show_system_status()

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

    elif args.command == 'langgraph-interactive':
        # Check database first
        if not check_database_status():
            print("❌ Database not ready - run 'scrape' first!")
            sys.exit(1)
        success = run_interactive_langgraph()
        if not success:
            sys.exit(1)

    elif args.command == 'socratic':
        success = run_socratic_seeker()
        if not success:
            sys.exit(1)

    elif args.command == 'socratic-test':
        success = run_socratic_tests()
        if not success:
            sys.exit(1)

    elif args.command == 'search':
        success = run_search_chooser()
        if not success:
            sys.exit(1)

    elif args.command == 'compare':
        if not check_database_status():
            print("❌ Database not ready - run 'scrape' first!")
            sys.exit(1)
        success = run_comparison_mode()
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

        # Step 3: Test Socratic (if available)
        if SOCRATIC_AVAILABLE:
            print("\n🎯 Step 3: Testing Socratic system...")
            if not run_socratic_tests():
                print("⚠️ Socratic tests failed, but continuing...")

        # Step 4: Final status check
        print(f"\n📊 Step 4: Final status check...")
        if not show_system_status():
            print("❌ Some systems not ready!")
            sys.exit(1)

        print("\n🎉 Complete setup finished successfully!")
        print("Now you can run:")
        print("  python Launcher.py search        # Choose search system")
        print("  python Launcher.py compare       # Compare systems")
        print("  python Launcher.py socratic      # Socratic system only")
        print("  python Launcher.py langgraph-interactive # LangGraph interactive")
        print("  python Launcher.py bot           # Web interface")
        print("  python Launcher.py console-bot   # Console interface")

    else:
        print(f"❌ Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
