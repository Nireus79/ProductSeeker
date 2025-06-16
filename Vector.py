import chromadb
from chromadb.config import Settings
import os
import json
import logging
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
import warnings
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
db_directory = "D:/Vector/ProductSeeker_db"


class ProductSeekerVectorDB:
    """
    ProductSeeker Vector Database for AI-powered product search using CLIP embeddings.
    Supports both image and text-based similarity search.
    """

    def __init__(self,
                 db_path: str = db_directory,
                 collection_name: str = "products",
                 model_name: str = "clip-ViT-B-32"):
        """
        Initialize ProductSeeker Vector Database

        Args:
            db_path: Path to store ChromaDB data
            collection_name: Name of the ChromaDB collection
            model_name: CLIP model name for embeddings
        """

        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.model_name = model_name

        # Create database directory
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self._initialize_database()

        # Initialize embedding model
        self._initialize_model()

        logger.info(f"ProductSeeker initialized with {self.get_database_stats()['total_products']} products")

    def _initialize_database(self) -> None:
        """Initialize ChromaDB client and collection"""
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Load or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection '{self.collection_name}' with {self.collection.count()} items")
            except ValueError:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "ProductSeeker AI product embeddings", "version": "1.0"}
                )
                logger.info(f"Created new collection '{self.collection_name}'")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def _initialize_model(self) -> None:
        """Initialize CLIP model for embeddings"""
        try:
            logger.info(f"Loading CLIP model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

            # Set device (CUDA if available)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

            logger.info(f"Model loaded successfully on device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def create_image_embedding(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Create CLIP embedding for an image

        Args:
            image_path: Path to the image file

        Returns:
            numpy array of embedding or None if failed
        """
        try:
            image_path = Path(image_path)

            if not image_path.exists():
                logger.warning(f"Image file not found: {image_path}")
                return None

            # Load and preprocess image
            with Image.open(image_path) as image:
                image = image.convert('RGB')

                # Create embedding using CLIP
                embedding = self.model.encode(image, convert_to_numpy=True)

                return embedding

        except Exception as e:
            logger.error(f"Error creating embedding for {image_path}: {e}")
            return None

    def create_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Create CLIP embedding for text

        Args:
            text: Text content to embed

        Returns:
            numpy array of embedding or None if failed
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                return None

            embedding = self.model.encode(text.strip(), convert_to_numpy=True)
            return embedding

        except Exception as e:
            logger.error(f"Error creating text embedding: {e}")
            return None

    def add_products(self, products: List[Dict[str, Any]], batch_size: int = 50) -> Dict[str, int]:
        """
        Add products to the vector database in batches

        Args:
            products: List of product dictionaries
            batch_size: Number of products to process in each batch

        Returns:
            Dictionary with statistics about added products
        """
        logger.info(f"Adding {len(products)} products to vector database...")

        stats = {"added": 0, "skipped": 0, "failed": 0}

        # Process products in batches
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]
            batch_stats = self._add_product_batch(batch)

            # Update statistics
            for key in stats:
                stats[key] += batch_stats[key]

        logger.info(
            f"Product addition completed. Added: {stats['added']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")
        return stats

    def _add_product_batch(self, products: List[Dict[str, Any]]) -> Dict[str, int]:
        """Add a batch of products to the database"""
        embeddings = []
        metadatas = []
        documents = []
        ids = []
        stats = {"added": 0, "skipped": 0, "failed": 0}

        for product in tqdm(products, desc="Processing batch", leave=False):
            try:
                product_id = str(product.get('id', ''))

                if not product_id:
                    logger.warning("Product missing ID, skipping")
                    stats["failed"] += 1
                    continue

                # Check if product already exists
                if self._product_exists(product_id):
                    stats["skipped"] += 1
                    continue

                # Create embedding (prioritize image over text)
                embedding = self._create_product_embedding(product)

                if embedding is not None:
                    embeddings.append(embedding.tolist())
                    metadatas.append(self._create_product_metadata(product))
                    documents.append(self._create_product_document(product))
                    ids.append(product_id)
                else:
                    logger.warning(f"Failed to create embedding for product {product_id}")
                    stats["failed"] += 1

            except Exception as e:
                logger.error(f"Error processing product {product.get('id', 'unknown')}: {e}")
                stats["failed"] += 1

        # Add batch to ChromaDB
        if embeddings:
            try:
                self.collection.add(
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents,
                    ids=ids
                )
                stats["added"] = len(embeddings)

            except Exception as e:
                logger.error(f"Error adding batch to database: {e}")
                stats["failed"] += len(embeddings)
                stats["added"] = 0

        return stats

    def _product_exists(self, product_id: str) -> bool:
        """Check if a product already exists in the database"""
        try:
            existing = self.collection.get(ids=[product_id])
            return len(existing['ids']) > 0
        except:
            return False

    def _create_product_embedding(self, product: Dict[str, Any]) -> Optional[np.ndarray]:
        """Create embedding for a product (image first, then text fallback)"""
        # Try image embedding first
        image_path = product.get('image_path')
        if image_path and Path(image_path).exists():
            embedding = self.create_image_embedding(image_path)
            if embedding is not None:
                return embedding

        # Fallback to text embedding
        text_content = self._create_text_content(product)
        return self.create_text_embedding(text_content)

    def _create_text_content(self, product: Dict[str, Any]) -> str:
        """Create text content from product data"""
        title = product.get('title', '').strip()
        description = product.get('description', '').strip()
        category = product.get('category', '').strip()
        brand = product.get('brand', '').strip()

        # Combine available text fields
        text_parts = [part for part in [title, brand, category, description] if part]
        return ' '.join(text_parts)

    def _create_product_metadata(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata dictionary for a product"""
        image_path = product.get('image_path', '')

        return {
            'title': product.get('title', '')[:500],  # Limit title length
            'price': str(product.get('price', '')),
            'description': product.get('description', '')[:1000],  # Limit description
            'url': product.get('url', ''),
            'image_path': image_path,
            'category': product.get('category', ''),
            'brand': product.get('brand', ''),
            'scraped_at': product.get('scraped_at', ''),
            'has_image': bool(image_path and Path(image_path).exists())
        }

    def _create_product_document(self, product: Dict[str, Any]) -> str:
        """Create document text for a product"""
        title = product.get('title', '')
        description = product.get('description', '')
        price = product.get('price', '')
        category = product.get('category', '')
        brand = product.get('brand', '')

        return f"Title: {title}\nBrand: {brand}\nCategory: {category}\nDescription: {description}\nPrice: {price}"

    def search_by_image(self, query_image_path: Union[str, Path], n_results: int = 10) -> Dict[str, Any]:
        """
        Search for similar products using an image

        Args:
            query_image_path: Path to query image
            n_results: Number of results to return

        Returns:
            Dictionary with search results
        """
        try:
            query_embedding = self.create_image_embedding(query_image_path)

            if query_embedding is None:
                return {"error": "Failed to create embedding for query image", "results": [], "count": 0}

            return self._execute_search(query_embedding, n_results)

        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return {"error": f"Search failed: {str(e)}", "results": [], "count": 0}

    def search_by_text(self, query_text: str, n_results: int = 10) -> Dict[str, Any]:
        """
        Search for products using text description

        Args:
            query_text: Text query
            n_results: Number of results to return

        Returns:
            Dictionary with search results
        """
        try:
            if not query_text or not query_text.strip():
                return {"error": "Empty query text provided", "results": [], "count": 0}

            query_embedding = self.create_text_embedding(query_text.strip())

            if query_embedding is None:
                return {"error": "Failed to create embedding for query text", "results": [], "count": 0}

            return self._execute_search(query_embedding, n_results)

        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return {"error": f"Search failed: {str(e)}", "results": [], "count": 0}

    def _execute_search(self, query_embedding: np.ndarray, n_results: int) -> Dict[str, Any]:
        """Execute search query against the database"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(n_results, self.collection.count()),
                include=['metadatas', 'documents', 'distances']
            )

            return self._format_search_results(results)

        except Exception as e:
            logger.error(f"Database search failed: {e}")
            return {"error": f"Database search failed: {str(e)}", "results": [], "count": 0}

    def _format_search_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format search results for easier consumption"""
        formatted_results = []

        if results.get('ids') and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                similarity = max(0, 1 - distance)  # Convert distance to similarity, ensure non-negative

                result = {
                    'id': results['ids'][0][i],
                    'distance': distance,
                    'similarity': similarity,
                    'confidence': self._calculate_confidence(similarity),
                    'metadata': results['metadatas'][0][i],
                    'document': results['documents'][0][i]
                }
                formatted_results.append(result)

        return {
            'results': formatted_results,
            'count': len(formatted_results),
            'query_successful': True
        }

    def _calculate_confidence(self, similarity: float) -> str:
        """Calculate confidence level based on similarity score"""
        if similarity >= 0.9:
            return "very_high"
        elif similarity >= 0.8:
            return "high"
        elif similarity >= 0.7:
            return "medium"
        elif similarity >= 0.6:
            return "low"
        else:
            return "very_low"

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the database"""
        try:
            total_count = self.collection.count()

            # Get sample of products to analyze
            sample_size = min(100, total_count)
            if sample_size > 0:
                sample = self.collection.get(limit=sample_size, include=['metadatas'])

                # Count products with images
                with_images = sum(1 for meta in sample['metadatas'] if meta.get('has_image', False))
                image_percentage = (with_images / sample_size) * 100 if sample_size > 0 else 0
            else:
                image_percentage = 0

            return {
                'total_products': total_count,
                'products_with_images': f"{image_percentage:.1f}%",
                'collection_name': self.collection_name,
                'database_path': str(self.db_path),
                'model_name': self.model_name,
                'device': self.device
            }

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {'error': str(e)}

    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific product by ID"""
        try:
            result = self.collection.get(
                ids=[product_id],
                include=['metadatas', 'documents']
            )

            if result['ids'] and len(result['ids']) > 0:
                return {
                    'id': result['ids'][0],
                    'metadata': result['metadatas'][0],
                    'document': result['documents'][0]
                }
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to get product {product_id}: {e}")
            return None

    def delete_product(self, product_id: str) -> bool:
        """Delete a product from the database"""
        try:
            self.collection.delete(ids=[product_id])
            logger.info(f"Deleted product {product_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete product {product_id}: {e}")
            return False

    def reset_database(self) -> bool:
        """Reset the database (delete all data)"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "ProductSeeker AI product embeddings", "version": "1.0"}
            )
            logger.info("Database reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            return False

    def export_products(self, output_file: Union[str, Path]) -> bool:
        """Export all products to JSON file"""
        try:
            output_file = Path(output_file)

            # Get all products
            all_products = self.collection.get(include=['metadatas'])

            # Format for export
            export_data = {
                'metadata': {
                    'total_products': len(all_products['ids']),
                    'collection_name': self.collection_name,
                    'exported_at': str(torch.datetime.now() if hasattr(torch, 'datetime') else 'unknown'),
                    'model_name': self.model_name
                },
                'products': []
            }

            for i, product_id in enumerate(all_products['ids']):
                product_data = {
                    'id': product_id,
                    **all_products['metadatas'][i]
                }
                export_data['products'].append(product_data)

            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(all_products['ids'])} products to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to export products: {e}")
            return False


def main():
    """Main function demonstrating ProductSeeker usage"""

    # Initialize database
    logger.info("Initializing ProductSeeker...")
    db = ProductSeekerVectorDB(
        db_path="database/product_seeker_data",
        collection_name="products_v1"
    )

    # Load products from JSON file
    products_file = "products.json"

    if Path(products_file).exists():
        logger.info(f"Loading products from {products_file}")

        with open(products_file, 'r', encoding='utf-8') as f:
            products = json.load(f)

        # Add products to database
        if products:
            stats = db.add_products(products)
            logger.info(f"Product addition completed: {stats}")

        # Show database statistics
        db_stats = db.get_database_stats()
        logger.info(f"Database statistics: {db_stats}")

        # Example searches
        if db_stats.get('total_products', 0) > 0:
            logger.info("\n=== Example Text Search ===")
            text_results = db.search_by_text("laptop computer gaming", n_results=3)

            if text_results.get('results'):
                for i, result in enumerate(text_results['results'], 1):
                    logger.info(f"{i}. {result['metadata']['title']}")
                    logger.info(f"   Similarity: {result['similarity']:.3f} ({result['confidence']})")
                    logger.info(f"   Price: {result['metadata']['price']}")
                    logger.info("   ---")

            # Example image search (if you have a query image)
            query_image = "query_image.jpg"  # Replace with actual image path
            if Path(query_image).exists():
                logger.info("\n=== Example Image Search ===")
                image_results = db.search_by_image(query_image, n_results=3)

                if image_results.get('results'):
                    for i, result in enumerate(image_results['results'], 1):
                        logger.info(f"{i}. {result['metadata']['title']}")
                        logger.info(f"   Similarity: {result['similarity']:.3f} ({result['confidence']})")
                        logger.info(f"   Price: {result['metadata']['price']}")
                        logger.info("   ---")

    else:
        logger.warning(f"Products file {products_file} not found. Please run the scraper first.")
        logger.info("You can still use the database for searches if it contains data from previous runs.")


if __name__ == "__main__":
    main()

'''
ProductSeeker Installation & Setup Guide
1. System Requirements

Python: 3.8 or higher
RAM: At least 4GB (8GB recommended)
Storage: 2-5GB free space (for models and database)
OS: Windows, macOS, or Linux

2. Installation Steps
Step 1: Create Project Directory
bashmkdir ProductSeeker
cd ProductSeeker
Step 2: Create Virtual Environment (Recommended)
bash# Create virtual environment
python -m venv productseeker_env

# Activate it
# Windows:
productseeker_env\Scripts\activate
# macOS/Linux:
source productseeker_env/bin/activate
Step 3: Install Required Packages
bash# Install core dependencies
pip install chromadb==0.4.18
pip install sentence-transformers==2.2.2
pip install torch torchvision torchaudio
pip install pillow==10.1.0
pip install tqdm==4.66.1
pip install numpy==1.24.3

# Optional: For better performance with CUDA (if you have NVIDIA GPU)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Alternative: Install from requirements.txt
Create a requirements.txt file:
txtchromadb==0.4.18
sentence-transformers==2.2.2
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
pillow==10.1.0
tqdm==4.66.1
numpy==1.24.3
pathlib2>=2.3.7
Then install:
bashpip install -r requirements.txt
3. Project Structure
Create this folder structure:
ProductSeeker/
├── product_seeker.py          # Main vector DB script
├── products.json              # Your scraped products (create this)
├── database/                  # Will be created automatically
│   └── product_seeker_data/   # ChromaDB storage
├── images/                    # Product images (optional)
├── query_images/              # Test images for searching
└── requirements.txt           # Dependencies
4. Prepare Sample Data
Create a sample products.json file for testing:
json[
  {
    "id": "1",
    "title": "Gaming Laptop ASUS ROG",
    "description": "High-performance gaming laptop with RTX 4060",
    "price": "€1299.99",
    "url": "https://example.com/laptop1",
    "category": "Laptops",
    "brand": "ASUS",
    "image_path": "images/laptop1.jpg",
    "scraped_at": "2024-01-15T10:30:00"
  },
  {
    "id": "2", 
    "title": "iPhone 15 Pro Max",
    "description": "Latest iPhone with titanium design and A17 Pro chip",
    "price": "€1199.00",
    "url": "https://example.com/iphone15",
    "category": "Smartphones",
    "brand": "Apple",
    "image_path": "images/iphone15.jpg",
    "scraped_at": "2024-01-15T10:31:00"
  }
]
5. First Run & Testing
Step 1: Basic Test
python# test_productseeker.py
from product_seeker import ProductSeekerVectorDB
import json

# Initialize database
db = ProductSeekerVectorDB()

# Check if it works
stats = db.get_database_stats()
print(f"Database initialized: {stats}")
Step 2: Load Products
python# Load sample products
with open('products.json', 'r', encoding='utf-8') as f:
    products = json.load(f)

# Add to database
result = db.add_products(products)
print(f"Added products: {result}")
Step 3: Test Search
python# Text search
results = db.search_by_text("gaming laptop", n_results=5)
print(f"Found {results['count']} results")

for result in results['results']:
    print(f"- {result['metadata']['title']}")
    print(f"  Similarity: {result['similarity']:.3f}")
    print(f"  Price: {result['metadata']['price']}")
6. PyCharm Setup
Step 1: Open Project in PyCharm

Open PyCharm
File → Open → Select your ProductSeeker folder
PyCharm will detect it as a Python project

Step 2: Configure Python Interpreter

File → Settings → Project → Python Interpreter
Click gear icon → Add
Select "Existing environment"
Browse to: ProductSeeker/productseeker_env/Scripts/python.exe (Windows)
or ProductSeeker/productseeker_env/bin/python (macOS/Linux)

Step 3: Install Packages in PyCharm

Go to Python Interpreter settings
Click "+" to add packages
Search and install: chromadb, sentence-transformers, etc.

7. Common Issues & Solutions
Issue 1: CUDA/GPU Problems
bash# If you get CUDA errors, install CPU-only version:
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
Issue 2: ChromaDB Errors
bash# If ChromaDB fails to install:
pip install --upgrade pip setuptools wheel
pip install chromadb --no-cache-dir
Issue 3: Memory Issues
python# In your code, add:
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Or reduce batch size:
db.add_products(products, batch_size=10)  # Instead of default 50
Issue 4: Slow Model Loading
python# The first run will be slow as it downloads the CLIP model
# Subsequent runs will be faster as the model is cached
8. Performance Tips
For Better Performance:

Use GPU: Install CUDA version of PyTorch if you have NVIDIA GPU
Batch Processing: Process products in batches (default: 50)
Image Optimization: Resize large images before processing
SSD Storage: Store database on SSD for faster access

Memory Usage:

CLIP Model: ~1.5GB RAM
Database: ~100MB per 10,000 products
Embeddings: ~512 floats per product

9. Quick Start Script
Create quick_start.py:
pythonfrom product_seeker import ProductSeekerVectorDB
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
10. Next Steps

Run the quick start script to verify everything works
Add your real product data to products.json
Test with different queries to see search quality
Add product images for image-based search
Integrate with your web scraper for automatic updates'''
