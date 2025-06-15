#!/usr/bin/env python3
"""
VisualShop AI - Automatic Project Setup Script
==============================================

This script will create the entire project structure and files automatically.
Run this script in the directory where you want to create the VisualShop AI project.

Usage: python setup_visualshop.py
"""

import os
import sys
from pathlib import Path


def create_directory_structure():
    """Create the project directory structure"""

    project_name = "visualshop_ai"
    base_path = Path(project_name)

    # Define directory structure
    directories = [
        "scraper",
        "scraper/images",
        "database",
        "database/chroma_data",
        "bot",
        "config",
        "utils"
    ]

    print(f"Creating project: {project_name}")
    print("=" * 50)

    # Create base directory
    base_path.mkdir(exist_ok=True)
    print(f"✓ Created: {base_path}")

    # Create subdirectories
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")

    return base_path


def create_requirements_txt(base_path):
    """Create requirements.txt file"""
    requirements_content = """chromadb==0.4.22
requests==2.31.0
beautifulsoup4==4.12.2
pillow==10.2.0
sentence-transformers==2.2.2
streamlit==1.31.0
langchain==0.1.7
langgraph==0.0.26
tqdm==4.66.1
validators==0.22.0
numpy>=1.21.0
torch>=1.9.0
"""

    file_path = base_path / "requirements.txt"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(requirements_content)

    print(f"✓ Created: {file_path}")


def create_init_files(base_path):
    """Create __init__.py files in all packages"""
    init_dirs = ["scraper", "database", "bot", "config", "utils"]

    for directory in init_dirs:
        init_path = base_path / directory / "__init__.py"
        with open(init_path, 'w', encoding='utf-8') as f:
            f.write(f'"""\\n{directory.title()} package for VisualShop AI\\n"""\\n')
        print(f"✓ Created: {init_path}")


def create_product_scraper(base_path):
    """Create the product scraper file"""
    scraper_content =
import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin, urlparse
from PIL import Image
import json
from tqdm import tqdm
import validators

class ProductScraper:
    def __init__(self, base_url="https://webscraper.io/test-sites/e-commerce/allinone", 
                 images_dir="scraper/images", 
                 delay=1):
        self.base_url = base_url
        self.images_dir = images_dir
        self.delay = delay  # Delay between requests (be respectful)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Create directories
        os.makedirs(self.images_dir, exist_ok=True)

        self.products = []

    def get_page(self, url):
        """Fetch a webpage with error handling"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def download_image(self, img_url, product_id):
        """Download and save product image"""
        try:
            # Make URL absolute
            img_url = urljoin(self.base_url, img_url)

            if not validators.url(img_url):
                print(f"Invalid image URL: {img_url}")
                return None

            response = self.session.get(img_url, timeout=10)
            response.raise_for_status()

            # Get file extension
            parsed_url = urlparse(img_url)
            ext = os.path.splitext(parsed_url.path)[1]
            if not ext:
                ext = '.jpg'  # Default extension

            # Save image
            filename = f"product_{product_id}{ext}"
            filepath = os.path.join(self.images_dir, filename)

            with open(filepath, 'wb') as f:
                f.write(response.content)

            # Verify image
            try:
                with Image.open(filepath) as img:
                    img.verify()
                return filepath
            except Exception as e:
                print(f"Invalid image file: {filepath}, error: {e}")
                os.remove(filepath)
                return None

        except Exception as e:
            print(f"Error downloading image {img_url}: {e}")
            return None

    def scrape_products(self, max_products=50):
        """Scrape products from the test e-commerce site"""
        print("Starting to scrape products from WebScraper.io test site...")

        # Get main page
        response = self.get_page(self.base_url)
        if not response:
            print("Failed to fetch main page")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find product links (adjust selectors based on site structure)
        product_links = []

        # Look for product cards/links
        # Common patterns for e-commerce sites
        selectors_to_try = [
            'a[href*="product"]',
            '.product a',
            '.item a',
            'a.title',
            'h4 a'
        ]

        for selector in selectors_to_try:
            links = soup.select(selector)
            if links:
                print(f"Found {len(links)} product links using selector: {selector}")
                product_links = links[:max_products]
                break

        if not product_links:
            print("No product links found. Let's examine the page structure...")
            # Print some of the page content for debugging
            print("Page title:", soup.title.text if soup.title else "No title")
            print("First few links found:")
            all_links = soup.find_all('a', href=True)[:10]
            for link in all_links:
                print(f"  - {link.get('href')} | Text: {link.text.strip()[:50]}")
            return []

        print(f"Found {len(product_links)} product links to scrape")

        # Scrape each product
        for i, link in enumerate(tqdm(product_links, desc="Scraping products")):
            if len(self.products) >= max_products:
                break

            product_url = urljoin(self.base_url, link.get('href'))
            product_data = self.scrape_single_product(product_url, i+1)

            if product_data:
                self.products.append(product_data)

            # Be respectful - add delay
            time.sleep(self.delay)

        print(f"Successfully scraped {len(self.products)} products")
        return self.products

    def scrape_single_product(self, product_url, product_id):
        """Scrape a single product page"""
        try:
            response = self.get_page(product_url)
            if not response:
                return None

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract product information
            # These selectors work for the webscraper.io test site
            title_selectors = ['h1', '.title', 'h2', '.product-title']
            price_selectors = ['.price', '.cost', '[class*="price"]']
            description_selectors = ['.description', '.product-description', 'p']
            image_selectors = ['img[src*="jpg"]', 'img[src*="png"]', '.product-image img', 'img']

            # Extract title
            title = None
            for selector in title_selectors:
                element = soup.select_one(selector)
                if element and element.text.strip():
                    title = element.text.strip()
                    break

            # Extract price
            price = None
            for selector in price_selectors:
                element = soup.select_one(selector)
                if element and element.text.strip():
                    price = element.text.strip()
                    break

            # Extract description
            description = None
            for selector in description_selectors:
                element = soup.select_one(selector)
                if element and element.text.strip() and len(element.text.strip()) > 20:
                    description = element.text.strip()[:500]  # Limit description length
                    break

            # Extract image
            image_path = None
            for selector in image_selectors:
                img_element = soup.select_one(selector)
                if img_element and img_element.get('src'):
                    image_path = self.download_image(img_element.get('src'), product_id)
                    if image_path:  # If download was successful
                        break

            # Create product data
            product_data = {
                'id': product_id,
                'title': title or f"Product {product_id}",
                'price': price or "N/A",
                'description': description or "No description available",
                'url': product_url,
                'image_path': image_path,
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            return product_data

        except Exception as e:
            print(f"Error scraping product {product_url}: {e}")
            return None

    def save_products_json(self, filename="products.json"):
        """Save scraped products to JSON file"""
        filepath = os.path.join(os.path.dirname(self.images_dir), filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.products, f, indent=2, ensure_ascii=False)
        print(f"Products saved to {filepath}")
        return filepath

# Example usage
if __name__ == "__main__":
    scraper = ProductScraper()
    products = scraper.scrape_products(max_products=20)  # Start with 20 products

    if products:
        scraper.save_products_json()
        print(f"\\nScraped {len(products)} products successfully!")
        print("Sample product:")
        print(json.dumps(products[0], indent=2) if products else "No products found")
    else:
        print("No products were scraped. Check the website structure.")
'''

    file_path = base_path / "scraper" / "product_scraper.py"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(scraper_content)

    print(f"✓ Created: {file_path}")


def create_vector_db(base_path):
    """Create the vector database file"""
    vector_db_content = '''import chromadb
from chromadb.config import Settings
import os
import json
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Any
from tqdm import tqdm

class VisualShopVectorDB:
    def __init__(self, 
                 db_path="database/chroma_data",
                 collection_name="products",
                 model_name="clip-ViT-B-32"):

        self.db_path = db_path
        self.collection_name = collection_name

        # Create database directory
        os.makedirs(db_path, exist_ok=True)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Load or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection '{collection_name}' with {self.collection.count()} items")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "VisualShop AI product embeddings"}
            )
            print(f"Created new collection '{collection_name}'")

        # Initialize embedding model
        print("Loading CLIP model for image embeddings...")
        self.model = SentenceTransformer(model_name)
        print(f"Loaded model: {model_name}")

        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model.to(self.device)

    def create_image_embedding(self, image_path: str) -> np.ndarray:
        """Create embedding for an image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')

            # Create embedding using CLIP
            embedding = self.model.encode(image, convert_to_numpy=True)

            return embedding

        except Exception as e:
            print(f"Error creating embedding for {image_path}: {e}")
            return None

    def create_text_embedding(self, text: str) -> np.ndarray:
        """Create embedding for text (title + description)"""
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"Error creating text embedding: {e}")
            return None

    def add_products(self, products: List[Dict[str, Any]]):
        """Add products to the vector database"""
        print(f"Adding {len(products)} products to vector database...")

        embeddings = []
        metadatas = []
        documents = []
        ids = []

        for product in tqdm(products, desc="Creating embeddings"):
            try:
                product_id = str(product['id'])

                # Skip if product already exists
                try:
                    existing = self.collection.get(ids=[product_id])
                    if existing['ids']:
                        print(f"Product {product_id} already exists, skipping...")
                        continue
                except:
                    pass  # Product doesn't exist, proceed

                # Create image embedding if image exists
                embedding = None
                if product.get('image_path') and os.path.exists(product['image_path']):
                    embedding = self.create_image_embedding(product['image_path'])

                # If no image embedding, create text embedding
                if embedding is None:
                    text_content = f"{product.get('title', '')} {product.get('description', '')}"
                    embedding = self.create_text_embedding(text_content)

                if embedding is not None:
                    embeddings.append(embedding.tolist())

                    # Prepare metadata
                    metadata = {
                        'title': product.get('title', ''),
                        'price': product.get('price', ''),
                        'description': product.get('description', '')[:1000],  # Limit description
                        'url': product.get('url', ''),
                        'image_path': product.get('image_path', ''),
                        'scraped_at': product.get('scraped_at', ''),
                        'has_image': bool(product.get('image_path') and os.path.exists(product.get('image_path', '')))
                    }
                    metadatas.append(metadata)

                    # Document content for text search
                    document = f"Title: {product.get('title', '')}\\nDescription: {product.get('description', '')}\\nPrice: {product.get('price', '')}"
                    documents.append(document)

                    ids.append(product_id)
                else:
                    print(f"Failed to create embedding for product {product_id}")

            except Exception as e:
                print(f"Error processing product {product.get('id', 'unknown')}: {e}")
                continue

        # Add to ChromaDB
        if embeddings:
            try:
                self.collection.add(
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents,
                    ids=ids
                )
                print(f"Successfully added {len(embeddings)} products to the database")
            except Exception as e:
                print(f"Error adding products to database: {e}")
        else:
            print("No embeddings created, nothing to add to database")

    def search_by_image(self, query_image_path: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for similar products using an image"""
        try:
            # Create embedding for query image
            query_embedding = self.create_image_embedding(query_image_path)

            if query_embedding is None:
                return {"error": "Failed to create embedding for query image"}

            # Search in database
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=['metadatas', 'documents', 'distances']
            )

            return self._format_search_results(results)

        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}

    def search_by_text(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for products using text description"""
        try:
            # Create embedding for query text
            query_embedding = self.create_text_embedding(query_text)

            if query_embedding is None:
                return {"error": "Failed to create embedding for query text"}

            # Search in database
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=['metadatas', 'documents', 'distances']
            )

            return self._format_search_results(results)

        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}

    def _format_search_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format search results for easier use"""
        formatted_results = []

        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'metadata': results['metadatas'][0][i],
                    'document': results['documents'][0][i]
                }
                formatted_results.append(result)

        return {
            'results': formatted_results,
            'count': len(formatted_results)
        }

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database"""
        try:
            count = self.collection.count()
            return {
                'total_products': count,
                'collection_name': self.collection_name,
                'database_path': self.db_path
            }
        except Exception as e:
            return {'error': str(e)}

    def reset_database(self):
        """Reset the database (delete all data)"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "VisualShop AI product embeddings"}
            )
            print("Database reset successfully")
        except Exception as e:
            print(f"Error resetting database: {e}")

# Example usage
if __name__ == "__main__":
    # Initialize database
    db = VisualShopVectorDB()

    # Load products from JSON (created by scraper)
    products_file = "products.json"
    if os.path.exists(products_file):
        with open(products_file, 'r', encoding='utf-8') as f:
            products = json.load(f)

        # Add products to database
        db.add_products(products)

        # Show database stats
        stats = db.get_database_stats()
        print(f"Database stats: {stats}")

        # Example search by text
        print("\\n=== Text Search Example ===")
        text_results = db.search_by_text("laptop computer", n_results=3)
        if 'results' in text_results:
            for result in text_results['results']:
                print(f"Product: {result['metadata']['title']}")
                print(f"Similarity: {result['similarity']:.3f}")
                print(f"Price: {result['metadata']['price']}")
                print("---")
    else:
        print(f"Products file {products_file} not found. Run the scraper first.")
'''

    file_path = base_path / "database" / "vector_db.py"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(vector_db_content)

    print(f"✓ Created: {file_path}")


def create_main_file(base_path):
    """Create the main entry point file"""
    main_content = '''#!/usr/bin/env python3
"""
VisualShop AI - Main Entry Point
===============================

This is the main entry point for the VisualShop AI project.
Choose what you want to do:
1. Scrape products
2. Build vector database
3. Run chatbot
4. All of the above
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("🛍️  VisualShop AI - Product Image Search System")
    print("=" * 50)

    while True:
        print("\\nWhat would you like to do?")
        print("1. 🕷️  Scrape products from e-commerce site")
        print("2. 🗄️  Build vector database from scraped products") 
        print("3. 🤖 Run chatbot interface")
        print("4. 🚀 Run complete pipeline (scrape + build DB + run bot)")
        print("5. 📊 Show database statistics")
        print("6. 🔄 Reset database")
        print("0. ❌ Exit")

        choice = input("\\nEnter your choice (0-6): ").strip()

        if choice == "1":
            run_scraper()
        elif choice == "2":
            build_database()
        elif choice == "3":
            run_chatbot()
        elif choice == "4":
            run_complete_pipeline()
        elif choice == "5":
            show_database_stats()
        elif choice == "6":
            reset_database()
        elif choice == "0":
            print("Goodbye! 👋")
            break
        else:
            print("Invalid choice. Please try again.")

def run_scraper():
    """Run the product scraper"""
    try:
        from scraper.product_scraper import ProductScraper

        max_products = input("How many products to scrape? (default: 20): ").strip()
        max_products = int(max_products) if max_products.isdigit() else 20

        print(f"\\n🕷️  Starting scraper for {max_products} products...")
        scraper = ProductScraper()
        products = scraper.scrape_products(max_products=max_products)

        if products:
            scraper.save_products_json()
            print(f"✅ Successfully scraped {len(products)} products!")
        else:
            print("❌ No products were scraped.")

    except Exception as e:
        print(f"❌ Error running scraper: {e}")

def build_database():
    """Build the vector database"""
    try:
        from database.vector_db import VisualShopVectorDB
        import json

        if not os.path.exists("products.json"):
            print("❌ No products.json found. Please run the scraper first.")
            return

        print("\\n🗄️  Building vector database...")
        db = VisualShopVectorDB()

        with open("products.json", 'r', encoding='utf-8') as f:
            products = json.load(f)

        db.add_products(products)
        stats = db.get_database_stats()
        print(f"✅ Database built successfully! {stats}")

    except Exception as e:
        print(f"❌ Error building database: {e}")

def run_chatbot():
    """Run the chatbot interface"""
    try:
        print("\\n🤖 Starting chatbot interface...")
        print("Note: Chatbot UI will be implemented in the next phase.")
        print("For now, you can test search functionality in the database module.")

        from database.vector_db import VisualShopVectorDB

        # Simple text search example
        db = VisualShopVectorDB()
        query = input("Enter search query: ").strip()

        if query:
            results = db.search_by_text(query, n_results=3)
            if 'results' in results and results['results']:
                print(f"\\n🔍 Found {len(results['results'])} results:")
                for i, result in enumerate(results['results'], 1):
                    print(f"{i}. {result['metadata']['title']}")
                    print(f"   Price: {result['metadata']['price']}")
                    print(f"   Similarity: {result['similarity']:.3f}")
                    print()
            else:
                print("❌ No results found.")

    except Exception as e:
        print(f"❌ Error running chatbot: {e}")

def run_complete_pipeline():
    """Run the complete pipeline"""
    print("\\n🚀 Running complete pipeline...")
    run_scraper()
    build_database() 
    print("\\n✅ Pipeline completed! You can now run the chatbot.")

def show_database_stats():
    """Show database statistics"""
    try:
        from database.vector_db import VisualShopVectorDB

        db = VisualShopVectorDB()
        stats = db.get_database_stats()

        print("\\n📊 Database Statistics:")
        print("-" * 30)
        for key, value in stats.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"❌ Error getting database stats: {e}")

def reset_database():
    """Reset the database"""
    try:
        from database.vector_db import VisualShopVectorDB

        confirm = input("⚠️  Are you sure you want to reset the database? (y/N): ").strip().lower()
        if confirm == 'y':
            db = VisualShopVectorDB()
            db.reset_database()
            print("✅ Database reset successfully!")
        else:
            print("Database reset cancelled.")

    except Exception as e:
        print(f"❌ Error resetting database: {e}")

if __name__ == "__main__":
    main()
'''

    file_path = base_path / "main.py"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(main_content)

    print(f"✓ Created: {file_path}")


def create_readme(base_path):
    """Create README.md file"""
    readme_content = """# 🛍️ VisualShop AI

**Εκπαιδευτικό project για αναζήτηση προϊόντων με χρήση εικόνων και AI**

## 📋 Περιγραφή

Το VisualShop AI είναι ένα σύστημα που:
1. Κάνει scraping προϊόντων από e-commerce sites
2. Δημιουργεί vector database με CLIP embeddings
3. Επιτρέπει αναζήτηση προϊόντων μέσω εικόνων
4. Παρέχει chatbot interface για διαδραστική χρήση

## 🚀 Γρήγορη Εκκίνηση

### 1. Εγκατάσταση

```bash
# Δημιουργία virtual environment
python -m venv visualshop_env

# Ενεργοποίηση (Windows)
visualshop_env\\Scripts\\activate

# Εγκατάσταση dependencies
pip install -r requirements.txt
```

### 2. Εκτέλεση

```bash
python main.py
```

## 📁 Δομή Project

```
visualshop_ai/
├── scraper/           # Web scraping functionality
├── database/          # Vector database management
├── bot/              # Chatbot interface
├── config/           # Configuration files
├── utils/            # Utility functions
├── main.py           # Main entry point
└── requirements.txt  # Dependencies
```

## 🔧 Χαρακτηριστικά

- ✅ Web scraping με respect για robots.txt
- ✅ CLIP embeddings για εικόνες
- ✅ ChromaDB για vector storage
- ✅ Text και image search
- 🚧 Streamlit chatbot UI (προγραμματίζεται)
- 🚧 LangGraph integration (προγραμματίζεται)

## 📚 Χρήση

1. **Scraping**: Συλλογή προϊόντων από test e-commerce site
2. **Database**: Δημιουργία embeddings και αποθήκευση
3. **Search**: Αναζήτηση με εικόνες ή κείμενο
4. **Chatbot**: Διαδραστική διεπαφή

## ⚙️ Τεχνικές Λεπτομέρειες

- **Vector DB**: ChromaDB (local) με δυνατότητα migration σε Pinecone
- **Embeddings**:'''

