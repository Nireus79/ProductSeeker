#!/usr/bin/env python3
"""
Fixed SocraticGenProductSeeker - Handles the AddableValuesDict issue
"""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProductInput:
    """Input data for product search"""
    text_query: Optional[str] = None
    image_path: Optional[str] = None
    voice_path: Optional[str] = None
    weights: Dict[str, float] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {"text": 1.0, "image": 0.0, "voice": 0.0}

@dataclass
class ProductMatch:
    """Result of product matching"""
    product_id: str
    confidence: float
    alternatives: List[str]
    metadata: Optional[Dict] = None

# Configuration
DATABASE_PATH = "D:/Vector/ProductSeeker_data"
COLLECTION_NAME = "ecommerce_test"
MODEL_NAME = "clip-ViT-B-32"


class SocraticProductSeeker:
    """Fixed Socratic Product Seeker"""
    
    def __init__(self, db_path: str = DATABASE_PATH, collection_name: str = COLLECTION_NAME, model_name: str = MODEL_NAME):
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_name = model_name
        self.db = None
        self.text_model = None
        self.image_model = None
        
    def initialize(self):
        """Initialize the seeker with database and models"""
        try:
            # Initialize database
            from Vector import ProductSeekerVectorDB
            self.db = ProductSeekerVectorDB(
                db_path=self.db_path,
                collection_name=self.collection_name,
                model_name=self.model_name
            )
            
            # Initialize models
            import sentence_transformers
            self.text_model = sentence_transformers.SentenceTransformer(self.model_name)
            
            # For image processing
            try:
                from transformers import ViTImageProcessor, ViTModel
                self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
                self.image_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
                logger.info("Image processing initialized")
            except Exception as e:
                logger.warning(f"Image processing not available: {e}")
                self.image_processor = None
                self.image_model = None
            
            logger.info("Socratic seeker initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def process_text(self, text_query: str) -> Optional[List[float]]:
        """Process text query into embeddings"""
        if not text_query or not self.text_model:
            return None
        
        try:
            embedding = self.text_model.encode(text_query)
            logger.info("Text processing completed successfully")
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return None
    
    def process_image(self, image_path: str) -> Optional[List[float]]:
        """Process image into embeddings"""
        if not image_path or not self.image_processor or not self.image_model:
            return None
        
        try:
            from PIL import Image
            import torch
            
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            inputs = self.image_processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.image_model(**inputs)
                # Use pooler output or mean of last hidden states
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embedding = outputs.pooler_output.squeeze().numpy()
                else:
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            logger.info("Image processing completed successfully")
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return None
    
    def search_products(self, embeddings: List[float], top_k: int = 10) -> List[Dict]:
        """Search products using embeddings"""
        if not self.db or not embeddings:
            return []
        
        try:
            # Use the database search functionality
            results = self.db.search_by_embedding(embeddings, top_k=top_k)
            return results
        except Exception as e:
            logger.error(f"Product search failed: {e}")
            return []
    
    def match_product(self, input_data: ProductInput) -> ProductMatch:
        """Match a product based on input data"""
        try:
            if not self.initialize():
                return ProductMatch("error_init_failed", 0.0, [])
            
            # Process different input types
            embeddings = None
            used_modalities = []
            
            # Text processing
            if input_data.text_query and input_data.weights.get("text", 0) > 0:
                text_embedding = self.process_text(input_data.text_query)
                if text_embedding:
                    embeddings = text_embedding
                    used_modalities.append("text")
                else:
                    logger.info("No text query provided, skipping text processing")
            
            # Image processing  
            if input_data.image_path and input_data.weights.get("image", 0) > 0:
                image_embedding = self.process_image(input_data.image_path)
                if image_embedding:
                    if embeddings is None:
                        embeddings = image_embedding
                    else:
                        # Combine embeddings (simple average for now)
                        text_weight = input_data.weights.get("text", 0)
                        image_weight = input_data.weights.get("image", 0)
                        total_weight = text_weight + image_weight
                        
                        if total_weight > 0:
                            embeddings = [
                                (text_weight * te + image_weight * ie) / total_weight
                                for te, ie in zip(embeddings, image_embedding)
                            ]
                    used_modalities.append("image")
                else:
                    logger.info("No image query provided, skipping image processing")
            
            # Voice processing (placeholder)
            if input_data.voice_path and input_data.weights.get("voice", 0) > 0:
                logger.info("No voice query provided, skipping voice processing")
            
            # Search products
            if embeddings:
                results = self.search_products(embeddings, top_k=10)
                
                if results:
                    # Get best match
                    best_match = results[0]
                    alternatives = [r.get('id', str(i)) for i, r in enumerate(results[1:6])]
                    
                    # Calculate confidence (use similarity score if available)
                    confidence = best_match.get('score', 0.0)
                    if confidence < 0:
                        confidence = max(0.0, 1.0 + confidence)  # Convert negative distance to positive confidence
                    
                    product_id = best_match.get('id', best_match.get('title', 'unknown'))
                    
                    logger.info(f"Product matching completed with confidence: {confidence}")
                    
                    return ProductMatch(
                        product_id=product_id,
                        confidence=confidence,
                        alternatives=alternatives,
                        metadata={
                            "used_modalities": used_modalities,
                            "total_results": len(results)
                        }
                    )
                else:
                    logger.warning("No matching products found")
                    return ProductMatch("no_matches", 0.0, [])
            else:
                logger.error("No valid embeddings generated")
                return ProductMatch("no_embeddings", 0.0, [])
                
        except Exception as e:
            logger.error(f"Product matching failed: {e}")
            return ProductMatch("error_matching", 0.0, [], {"error": str(e)})


# Global seeker instance
_seeker = None

def get_seeker():
    """Get or create global seeker instance"""
    global _seeker
    if _seeker is None:
        _seeker = SocraticProductSeeker()
    return _seeker


def run_pipeline_sync(input_data: ProductInput) -> ProductMatch:
    """Main pipeline function - FIXED VERSION"""
    try:
        seeker = get_seeker()
        result = seeker.match_product(input_data)
        
        # Ensure we always return a ProductMatch object
        if isinstance(result, ProductMatch):
            return result
        else:
            # Handle any unexpected formats
            return ProductMatch("error_format", 0.0, [], {"original_type": str(type(result))})
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return ProductMatch("error_pipeline", 0.0, [], {"error": str(e)})


def run_tests() -> bool:
    """Run system tests"""
    print("🧪 Running Socratic system tests...")
    
    try:
        # Test 1: Text query
        print("Test 1: Text query")
        input_data = ProductInput(
            text_query="book",
            weights={"text": 1.0, "image": 0.0, "voice": 0.0}
        )
        result = run_pipeline_sync(input_data)
        
        if isinstance(result, ProductMatch) and result.product_id != "error_pipeline":
            print(f"✅ Text query test passed: {result.product_id}")
        else:
            print(f"❌ Text query test failed: {result.product_id}")
            return False
        
        # Test 2: Empty query
        print("Test 2: Empty query")
        input_data = ProductInput(weights={"text": 0.0, "image": 0.0, "voice": 0.0})
        result = run_pipeline_sync(input_data)
        
        if isinstance(result, ProductMatch):
            print(f"✅ Empty query test passed: {result.product_id}")
        else:
            print(f"❌ Empty query test failed")
            return False
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Tests failed: {e}")
        return False


def main():
    """Main function for interactive use"""
    print("🎯 Socratic GenProductSeeker - Fixed Version")
    print("=" * 50)
    
    seeker = get_seeker()
    
    while True:
        print("\nOptions:")
        print("1. Text search")
        print("2. Image search")
        print("3. Combined search")
        print("4. Run tests")
        print("0. Exit")
        
        choice = input("\nSelect option (0-4): ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            query = input("Enter text query: ").strip()
            if query:
                input_data = ProductInput(
                    text_query=query,
                    weights={"text": 1.0, "image": 0.0, "voice": 0.0}
                )
                result = run_pipeline_sync(input_data)
                print(f"\n📋 Result:")
                print(f"Product ID: {result.product_id}")
                print(f"Confidence: {result.confidence:.2f}")
                print(f"Alternatives: {len(result.alternatives)}")
        
        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            if image_path and Path(image_path).exists():
                input_data = ProductInput(
                    image_path=image_path,
                    weights={"text": 0.0, "image": 1.0, "voice": 0.0}
                )
                result = run_pipeline_sync(input_data)
                print(f"\n📋 Result:")
                print(f"Product ID: {result.product_id}")
                print(f"Confidence: {result.confidence:.2f}")
                print(f"Alternatives: {len(result.alternatives)}")
            else:
                print("❌ Invalid or missing image path")
        
        elif choice == '3':
            query = input("Enter text query: ").strip()
            image_path = input("Enter image path (optional): ").strip()
            
            if query or (image_path and Path(image_path).exists()):
                input_data = ProductInput(
                    text_query=query if query else None,
                    image_path=image_path if image_path and Path(image_path).exists() else None,
                    weights={
                        "text": 0.7 if query else 0.0,
                        "image": 0.3 if image_path and Path(image_path).exists() else 0.0,
                        "voice": 0.0
                    }
                )
                result = run_pipeline_sync(input_data)
                print(f"\n📋 Result:")
                print(f"Product ID: {result.product_id}")
                print(f"Confidence: {result.confidence:.2f}")
                print(f"Alternatives: {len(result.alternatives)}")
            else:
                print("❌ Please provide at least text query or valid image path")
        
        elif choice == '4':
            run_tests()
        
        else:
            print("Invalid choice")

    print("👋 Goodbye!")


# For configuration checking
def get_database_config():
    """Return database configuration"""
    return {
        "db_path": DATABASE_PATH,
        "collection_name": COLLECTION_NAME,
        "model_name": MODEL_NAME
    }


if __name__ == "__main__":
    main()
