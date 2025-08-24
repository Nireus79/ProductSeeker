from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from langgraph.graph import StateGraph, END

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not available - using simplified processing")

try:
    from PIL import Image
    import torch
    from transformers import ViTFeatureExtractor, ViTModel

    ML_DEPENDENCIES_AVAILABLE = True
except ImportError:
    ML_DEPENDENCIES_AVAILABLE = False
    logger.warning("ML dependencies not available - using mock processing")


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

        if not ML_DEPENDENCIES_AVAILABLE:
            logger.info("ML dependencies not available, using mock image processing")
            # Mock processing - generate dummy features
            state.image_features = np.random.random((1, 768))
            return state

        # Check if image file exists
        import os
        if not os.path.exists(state.image_path):
            logger.warning(f"Image file not found: {state.image_path}")
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
    """Creates and configures the product matching pipeline using LangGraph or simplified version"""
    try:
        if not LANGGRAPH_AVAILABLE:
            # Return a simple pipeline function
            def simple_pipeline(state: ProductSearchState) -> ProductSearchState:
                state = process_image(state)
                state = process_text(state)
                state = process_voice(state)
                state = match_products(state)
                return state

            return simple_pipeline

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


def run_pipeline_sync(input_data: ProductInput) -> ProductMatch:
    """Synchronous version of run_pipeline for launcher compatibility"""
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
        if LANGGRAPH_AVAILABLE:
            result = app.invoke(initial_state)
        else:
            result = app(initial_state)  # Simple function call

        if result.final_result is None:
            raise ValueError("Pipeline failed to produce a result")

        logger.info(f"Pipeline completed successfully")
        return result.final_result

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


# Main function for launcher compatibility
def main():
    """Main execution function - launcher entry point"""
    print("=== Socratic Generation Product Seeker ===")
    print("This tool performs multimodal product matching using image, text, and voice inputs.")
    print()

    # Check dependencies
    print("Dependency Status:")
    print(f"  LangGraph: {'✓ Available' if LANGGRAPH_AVAILABLE else '✗ Not available (using simplified processing)'}")
    print(
        f"  ML Libraries: {'✓ Available' if ML_DEPENDENCIES_AVAILABLE else '✗ Not available (using mock processing)'}")
    print()

    try:
        # Interactive mode
        print("Enter your product search details:")

        # Get text query
        text_query = input("Text description (or press Enter to skip): ").strip()
        if not text_query:
            text_query = None

        # Get voice query
        voice_query = input("Voice description (or press Enter to skip): ").strip()
        if not voice_query:
            voice_query = None

        # Get image path
        image_path = input("Image path (or press Enter to skip): ").strip()
        if not image_path:
            image_path = None

        # Validate at least one input
        if not any([text_query, voice_query, image_path]):
            print("Error: At least one input (text, voice, or image) is required.")
            return

        # Get weights (optional)
        print("\nUsing default weights: image=0.6, text=0.3, voice=0.1")
        use_custom = input("Use custom weights? (y/n): ").lower().startswith('y')

        weights = {"image": 0.6, "text": 0.3, "voice": 0.1}
        if use_custom:
            try:
                weights["image"] = float(input("Image weight (0-1): ") or "0.6")
                weights["text"] = float(input("Text weight (0-1): ") or "0.3")
                weights["voice"] = float(input("Voice weight (0-1): ") or "0.1")
            except ValueError:
                print("Invalid weight values, using defaults.")
                weights = {"image": 0.6, "text": 0.3, "voice": 0.1}

        # Create input
        test_input = ProductInput(
            image_path=image_path,
            text_query=text_query,
            voice_query=voice_query,
            weights=weights
        )

        print("\nProcessing...")

        # Execute pipeline
        result = run_pipeline_sync(test_input)

        # Display results
        print("\n=== RESULTS ===")
        print(f"Best Match: {result.product_id}")
        print(f"Confidence: {result.confidence:.2f}")
        print("\nSimilarity Scores:")
        for modality, score in result.similarity_scores.items():
            print(f"  {modality}: {score:.2f}")

        print("\nAlternative Products:")
        for i, alt in enumerate(result.alternatives, 1):
            print(f"  {i}. {alt['product_id']} (confidence: {alt['confidence']:.2f})")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"Error: {e}")


# Test function for development
def run_tests():
    """Test function for development and validation"""
    try:
        print("Running tests...")

        # Test 1: With text query only
        test_input_text_only = ProductInput(
            text_query="red running shoes",
            weights={"text": 1.0, "image": 0.0, "voice": 0.0}
        )

        result = run_pipeline_sync(test_input_text_only)
        assert isinstance(result, ProductMatch)
        assert result.confidence > 0
        print("✓ Test 1 (text only) passed")

        # Test 2: With voice query only
        test_input_voice_only = ProductInput(
            voice_query="looking for a winter jacket",
            weights={"voice": 1.0, "image": 0.0, "text": 0.0}
        )

        result = run_pipeline_sync(test_input_voice_only)
        assert isinstance(result, ProductMatch)
        assert result.confidence > 0
        print("✓ Test 2 (voice only) passed")

        # Test 3: Mixed input
        test_input_mixed = ProductInput(
            text_query="comfortable sneakers",
            voice_query="something for running",
            weights={"text": 0.6, "voice": 0.4, "image": 0.0}
        )

        result = run_pipeline_sync(test_input_mixed)
        assert isinstance(result, ProductMatch)
        assert result.confidence > 0
        print("✓ Test 3 (mixed input) passed")

        print("All tests passed successfully!")
        return True

    except AssertionError as e:
        print(f"✗ Test assertion failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during testing: {e}")
        return False


# Entry point for both launcher and standalone execution
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        main()