import chromadb
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