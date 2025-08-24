from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
import numpy as np
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTModel
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# State Schema Definition
class ProductSearchState(BaseModel):
    """State schema for the product search pipeline"""
    image_path: Optional[str] = None
    text_query: Optional[str] = None
    voice_query: Optional[str] = None
    weights: Dict[str, float] = Field(
        default_factory=lambda: {"image": 0.6, "text": 0.3, "voice": 0.1}
    )
    image_features: Optional[np.ndarray] = None
    text_features: Optional[str] = None
    voice_features: Optional[str] = None
    final_result: Optional['ProductMatch'] = None

    class Config:
        arbitrary_types_allowed = True


# Event Schema Definitions
class ProductInput(BaseModel):
    """Schema for product search input"""
    image_path: Optional[str] = Field(None, description="Path to input image")
    text_query: Optional[str] = Field(None, description="Text search query")
    voice_query: Optional[str] = Field(None, description="Voice input transcript")
    weights: Dict[str, float] = Field(
        default_factory=lambda: {"image": 0.6, "text": 0.3, "voice": 0.1}
    )


class ProductMatch(BaseModel):
    """Schema for matched product results"""
    product_id: str
    confidence: float
    similarity_scores: Dict[str, float]
    alternatives: List[Dict[str, Union[str, float]]]


# Node Functions
def process_image(state: ProductSearchState) -> ProductSearchState:
    """Processes image inputs and extracts features"""
    try:
        if not state.image_path:
            logger.info("No image path provided, skipping image processing")
            state.image_features = None
            return state

        # Initialize models (in production, these should be cached/singleton)
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        model = ViTModel.from_pretrained('google/vit-base-patch16-224')

        image = Image.open(state.image_path)
        inputs = feature_extractor(images=image, return_tensors="pt")

        with torch.no_grad():  # More efficient inference
            outputs = model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).detach().numpy()

        state.image_features = features
        logger.info("Image processing completed successfully")
        return state

    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        state.image_features = None
        return state


def process_text(state: ProductSearchState) -> ProductSearchState:
    """Processes text inputs for semantic matching"""
    try:
        if not state.text_query:
            logger.info("No text query provided, skipping text processing")
            state.text_features = None
            return state

        # Implement text processing logic here
        # Placeholder for demonstration - in production, use proper text embeddings
        processed_text = f"processed_{state.text_query.lower().replace(' ', '_')}"
        state.text_features = processed_text

        logger.info("Text processing completed successfully")
        return state

    except Exception as e:
        logger.error(f"Text processing failed: {e}")
        state.text_features = None
        return state


def process_voice(state: ProductSearchState) -> ProductSearchState:
    """Processes voice inputs"""
    try:
        if not state.voice_query:
            logger.info("No voice query provided, skipping voice processing")
            state.voice_features = None
            return state

        # Placeholder for voice processing
        processed_voice = f"voice_processed_{state.voice_query.lower().replace(' ', '_')}"
        state.voice_features = processed_voice

        logger.info("Voice processing completed successfully")
        return state

    except Exception as e:
        logger.error(f"Voice processing failed: {e}")
        state.voice_features = None
        return state


def match_products(state: ProductSearchState) -> ProductSearchState:
    """Combines features and performs product matching"""
    try:
        # Compute weighted combination score
        combined_score = compute_combined_score(
            state.image_features,
            state.text_features,
            state.voice_features,
            state.weights
        )

        # Create similarity scores based on available features
        similarity_scores = {}
        if state.image_features is not None:
            similarity_scores["image"] = 0.9  # Placeholder
        if state.text_features is not None:
            similarity_scores["text"] = 0.8  # Placeholder
        if state.voice_features is not None:
            similarity_scores["voice"] = 0.75  # Placeholder

        # Create result
        result = ProductMatch(
            product_id="sample_product_123",
            confidence=combined_score,
            similarity_scores=similarity_scores,
            alternatives=[
                {"product_id": "alt_1", "confidence": 0.75},
                {"product_id": "alt_2", "confidence": 0.70},
                {"product_id": "alt_3", "confidence": 0.65},
            ]
        )

        state.final_result = result
        logger.info(f"Product matching completed with confidence: {combined_score}")
        return state

    except Exception as e:
        logger.error(f"Product matching failed: {e}")
        raise


def compute_combined_score(
        image_features: Optional[np.ndarray],
        text_features: Optional[str],
        voice_features: Optional[str],
        weights: Dict[str, float]
) -> float:
    """Compute weighted combination of feature scores"""
    total_weight = 0
    weighted_score = 0

    # Image score
    if image_features is not None:
        image_score = 0.9  # Placeholder - implement actual similarity calculation
        weighted_score += weights.get("image", 0) * image_score
        total_weight += weights.get("image", 0)

    # Text score
    if text_features is not None:
        text_score = 0.8  # Placeholder - implement actual similarity calculation
        weighted_score += weights.get("text", 0) * text_score
        total_weight += weights.get("text", 0)

    # Voice score
    if voice_features is not None:
        voice_score = 0.75  # Placeholder - implement actual similarity calculation
        weighted_score += weights.get("voice", 0) * voice_score
        total_weight += weights.get("voice", 0)

    # Normalize by total weight to get final score
    if total_weight > 0:
        return weighted_score / total_weight
    else:
        return 0.0


def create_product_matching_pipeline():
    """Creates and configures the product matching pipeline using LangGraph"""
    try:
        # Create the graph
        workflow = StateGraph(ProductSearchState)

        # Add nodes
        workflow.add_node("process_image", process_image)
        workflow.add_node("process_text", process_text)
        workflow.add_node("process_voice", process_voice)
        workflow.add_node("match_products", match_products)

        # Set entry point
        workflow.set_entry_point("process_image")

        # Add edges
        workflow.add_edge("process_image", "process_text")
        workflow.add_edge("process_text", "process_voice")
        workflow.add_edge("process_voice", "match_products")
        workflow.add_edge("match_products", END)

        # Compile the graph
        app = workflow.compile()
        return app

    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        raise


async def run_pipeline(input_data: ProductInput) -> ProductMatch:
    """Run the product matching pipeline"""
    try:
        # Create pipeline
        app = create_product_matching_pipeline()

        # Create initial state
        initial_state = ProductSearchState(
            image_path=input_data.image_path,
            text_query=input_data.text_query,
            voice_query=input_data.voice_query,
            weights=input_data.weights
        )

        # Execute pipeline
        result = app.invoke(initial_state)

        if result.final_result is None:
            raise ValueError("Pipeline failed to produce a result")

        logger.info(f"Pipeline completed successfully")
        return result.final_result

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


async def main():
    """Main execution function"""
    try:
        # Example input with image and text
        test_input = ProductInput(
            image_path="test_image.jpg",
            text_query="blue casual shirt",
            weights={"image": 0.7, "text": 0.3}
        )

        # Execute pipeline
        result = await run_pipeline(test_input)
        logger.info(f"Matching results: {result}")

        return result
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    import asyncio


    # Basic test function
    async def run_tests():
        try:
            # Test 1: With text query only (no image file required)
            test_input_text_only = ProductInput(
                text_query="red running shoes",
                weights={"text": 1.0}
            )

            result = await run_pipeline(test_input_text_only)
            assert isinstance(result, ProductMatch)
            assert result.confidence > 0
            logger.info("Test 1 (text only) passed successfully")

            # Test 2: With voice query only
            test_input_voice_only = ProductInput(
                voice_query="looking for a winter jacket",
                weights={"voice": 1.0}
            )

            result = await run_pipeline(test_input_voice_only)
            assert isinstance(result, ProductMatch)
            assert result.confidence > 0
            logger.info("Test 2 (voice only) passed successfully")

            # Test 3: Mixed input (no image file required)
            test_input_mixed = ProductInput(
                text_query="comfortable sneakers",
                voice_query="something for running",
                weights={"text": 0.6, "voice": 0.4}
            )

            result = await run_pipeline(test_input_mixed)
            assert isinstance(result, ProductMatch)
            assert result.confidence > 0
            logger.info("Test 3 (mixed input) passed successfully")

            logger.info("All tests passed successfully!")

        except AssertionError as e:
            logger.error(f"Test assertion failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during testing: {e}")


    # Run tests
    asyncio.run(run_tests())
