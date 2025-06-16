from Vector import ProductSeekerVectorDB
import json
from pathlib import Path


def main():
    print("🚀 Starting ProductSeeker...")

    # Initialize
    db = ProductSeekerVectorDB()

    # Load products if file exists
    if Path("products.json").exists():
        with open("products.json", "r", encoding="utf-8") as f:
            products = json.load(f)

        print(f"📦 Adding {len(products)} products...")
        stats = db.add_products(products)
        print(f"✅ Success: {stats}")

    # Show stats
    db_stats = db.get_database_stats()
    print(f"📊 Database: {db_stats}")

    # Test search
    query = input("\n🔍 Enter search query (or press Enter to skip): ")
    if query.strip():
        results = db.search_by_text(query, n_results=3)

        print(f"\n📋 Found {results['count']} results:")
        for i, result in enumerate(results['results'], 1):
            print(f"{i}. {result['metadata']['title']}")
            print(f"   💯 Confidence: {result['confidence']}")
            print(f"   💰 Price: {result['metadata']['price']}")
            print()


if __name__ == "__main__":
    main()


# ProductSeeker/
# ├── scraper/
# │   ├── scrape_products.py
# │   └── images/
# ├── database/
# │   ├── vector_db.py
# │   └── chroma_db/
# ├── bot/
# │   ├── chatbot.py
# │   └── ui.py
# └── requirements.txt
