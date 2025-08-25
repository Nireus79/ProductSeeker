#!/usr/bin/env python3
"""
Advanced Agentic Product Search System
=====================================

A comprehensive multi-modal product search system with:
- Voice input processing
- Image search capabilities
- Text search with NLP
- Agentic workflow with specialized agents
- Learning and personalization
- GUI interface

Author: AI Assistant
"""

import logging
import asyncio
import json
import time
import threading
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict, Annotated, Union, Literal, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from collections import defaultdict, deque
import warnings

# GUI imports
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    GUI_AVAILABLE = True
except ImportError:
    print("Warning: tkinter not available. GUI will be disabled.")
    GUI_AVAILABLE = False

# Audio processing imports
try:
    import speech_recognition as sr
    import pyttsx3
    import pyaudio
    import wave

    AUDIO_AVAILABLE = True
except ImportError:
    print("Warning: Audio libraries not available. Voice features will be disabled.")
    AUDIO_AVAILABLE = False

# Image processing imports
try:
    from PIL import Image, ImageTk
    import numpy as np
    import cv2

    IMAGE_AVAILABLE = True
except ImportError:
    print("Warning: Image libraries not available. Image features will be disabled.")
    IMAGE_AVAILABLE = False

# LangGraph imports (using mock if not available)
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.tools import tool

    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("Warning: LangGraph not available. Using simplified workflow.")
    LANGGRAPH_AVAILABLE = False


    # Mock classes for when LangGraph is not available
    class StateGraph:
        def __init__(self, state_type):
            self.nodes = {}
            self.edges = {}

        def add_node(self, name, func):
            self.nodes[name] = func

        def add_edge(self, from_node, to_node):
            pass

        def add_conditional_edges(self, node, condition, mapping):
            pass

        def set_entry_point(self, node):
            self.entry = node

        def compile(self):
            return SimplifiedGraph(self.nodes)


    class SimplifiedGraph:
        def __init__(self, nodes):
            self.nodes = nodes

        def invoke(self, state):
            # Simple linear execution
            for name, func in self.nodes.items():
                try:
                    state = func(state)
                except Exception as e:
                    logging.error(f"Node {name} failed: {e}")
            return state


    END = "END"


    class BaseMessage:
        def __init__(self, content=""):
            self.content = content


    class HumanMessage(BaseMessage):
        pass


    class AIMessage(BaseMessage):
        pass


    def add_messages(messages):
        return messages

# Database imports with enhanced path handling
try:
    # Add the Vector module path to Python path if it exists
    vector_path = Path("D:/Vector")
    if vector_path.exists() and str(vector_path) not in sys.path:
        sys.path.insert(0, str(vector_path))

    from Vector import ProductSeekerVectorDB
    from Integrater import IntegratedProductScraper

    DATABASE_AVAILABLE = True
    print("✅ Database modules loaded successfully")
except ImportError as e:
    print(f"Warning: Database modules not found ({e}). Using mock implementations.")
    DATABASE_AVAILABLE = False


    class ProductSeekerVectorDB:
        def __init__(self, *args, **kwargs):
            self.mock_data = True
            self.db_path = kwargs.get('db_path', 'mock_db')
            print(f"🔄 Using mock database for demonstration (target path: {self.db_path})")

        def search_by_text(self, query, n_results=10):
            # Generate mock results
            results = []
            for i in range(min(n_results, 5)):
                results.append({
                    'id': f'product_{i}',
                    'similarity': max(0.5, 0.95 - (i * 0.1)),
                    'metadata': {
                        'title': f'Premium {query.title()} Model {i + 1}',
                        'price': f'${(i + 1) * 99 + 50}.99',
                        'category': 'Electronics',
                        'brand': f'TechBrand{i + 1}',
                        'description': f'High-quality {query} with advanced features and excellent performance. Perfect for both professional and personal use.',
                        'rating': round(4.0 + (i * 0.2), 1),
                        'reviews': 100 + (i * 50)
                    }
                })
            return {'results': results}

        def search_by_image(self, image_path, n_results=10):
            return self.search_by_text(f"product from {Path(image_path).name}", n_results)

        def get_database_stats(self):
            return {
                "total_products": 10000,
                "collections": 1,
                "last_updated": datetime.now().isoformat(),
                "database_path": self.db_path,
                "status": "mock"
            }


    class IntegratedProductScraper:
        def __init__(self, *args, **kwargs):
            pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration classes
@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    name: str
    max_retries: int = 3
    timeout: int = 30
    priority: int = 1
    enable_learning: bool = True


@dataclass
class SystemConfig:
    """Enhanced system configuration"""
    max_results: int = 20
    min_similarity_threshold: float = 0.5
    max_refinements: int = 3
    enable_caching: bool = True
    cache_ttl: int = 600
    enable_parallel_search: bool = True
    search_timeout: int = 30
    voice_enabled: bool = AUDIO_AVAILABLE
    image_enabled: bool = IMAGE_AVAILABLE
    learning_enabled: bool = True
    explanation_enabled: bool = True
    proactive_suggestions: bool = True
    voice_language: str = "en-US"
    tts_rate: int = 200
    tts_voice: str = "default"
    # Database configuration
    database_path: str = "D:/Vector/ProductSeeker_data"
    collection_name: str = "products"
    model_name: str = "all-MiniLM-L6-v2"


@dataclass
class UserProfile:
    """Dynamic user profile that learns from interactions"""
    user_id: str = "default"
    preferences: Dict[str, float] = field(default_factory=dict)
    search_history: List[Dict] = field(default_factory=list)
    interaction_patterns: Dict[str, int] = field(default_factory=dict)
    preferred_brands: List[str] = field(default_factory=list)
    price_range: Dict[str, float] = field(default_factory=dict)
    categories_of_interest: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


class AgenticSearchState(TypedDict):
    """Enhanced state for agentic search system"""
    messages: List[BaseMessage]
    original_query: str
    current_query: str
    search_type: Literal["text", "voice", "image", "hybrid"]
    input_data: Dict[str, Any]
    processed_input: Dict[str, Any]
    search_results: List[Dict[str, Any]]
    user_intent: Dict[str, Any]
    user_profile: UserProfile
    performance_metrics: Dict[str, float]
    explanations: List[str]
    suggestions: List[str]
    refinement_count: int
    session_id: str
    timestamp: datetime


# Base Agent Class
class BaseAgent:
    """Base class for all specialized agents"""

    def __init__(self, name: str, config: AgentConfig):
        self.name = name
        self.config = config
        self.memory = deque(maxlen=100)
        self.performance_stats = defaultdict(float)
        self.learning_data = defaultdict(list)

    def execute(self, state: AgenticSearchState) -> AgenticSearchState:
        """Execute agent's main functionality"""
        raise NotImplementedError

    def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Learn from user feedback"""
        if self.config.enable_learning:
            self.learning_data['feedback'].append({
                'timestamp': datetime.now(),
                'data': feedback
            })

    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        return dict(self.performance_stats)


# Specialized Agents
class VoiceProcessingAgent(BaseAgent):
    """Specialized agent for voice input processing"""

    def __init__(self, config: AgentConfig, system_config: SystemConfig):
        super().__init__("VoiceProcessor", config)
        self.system_config = system_config

        if AUDIO_AVAILABLE and system_config.voice_enabled:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', system_config.tts_rate)

                # Calibrate microphone
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)

                self.available = True
                logger.info("🎤 Voice processing initialized")
            except Exception as e:
                logger.warning(f"Voice initialization failed: {e}")
                self.available = False
        else:
            self.available = False

    def execute(self, state: AgenticSearchState) -> AgenticSearchState:
        """Process voice input"""
        start_time = time.time()

        if not self.available:
            state["processed_input"]["voice_error"] = "Voice processing not available"
            return state

        try:
            if state["search_type"] == "voice" or "audio_data" in state["input_data"]:
                if "audio_data" in state["input_data"]:
                    text = self._process_audio_data(state["input_data"]["audio_data"])
                else:
                    text = "Voice input processed"  # Simplified for demo

                state["processed_input"]["voice_text"] = text
                state["current_query"] = text
                state["explanations"].append(f"🎤 Processed voice input: '{text}'")

        except Exception as e:
            logger.error(f"Voice processing failed: {e}")
            state["processed_input"]["voice_error"] = str(e)

        self.performance_stats["processing_time"] += time.time() - start_time
        return state

    def _process_audio_data(self, audio_data: bytes) -> str:
        """Process provided audio data"""
        try:
            # Simplified audio processing
            return "laptop gaming high performance"  # Mock result
        except Exception as e:
            return f"Audio processing error: {e}"

    def speak(self, text: str) -> None:
        """Convert text to speech"""
        if self.available and self.system_config.voice_enabled:
            try:
                # Clean text for TTS
                clean_text = text.replace('*', '').replace('#', '').replace('`', '')
                # Limit length
                if len(clean_text) > 200:
                    clean_text = clean_text[:200] + "..."

                self.tts_engine.say(clean_text)
                self.tts_engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS error: {e}")


class ImageProcessingAgent(BaseAgent):
    """Specialized agent for image input processing"""

    def __init__(self, config: AgentConfig):
        super().__init__("ImageProcessor", config)
        self.available = IMAGE_AVAILABLE

    def execute(self, state: AgenticSearchState) -> AgenticSearchState:
        """Process image input"""
        start_time = time.time()

        if not self.available:
            state["processed_input"]["image_error"] = "Image processing not available"
            return state

        try:
            if state["search_type"] == "image" or "image_path" in state["input_data"]:
                image_path = state["input_data"].get("image_path")
                if image_path and Path(image_path).exists():
                    image_info = self._analyze_image(image_path)
                    state["processed_input"]["image_features"] = image_info

                    description = self._generate_image_description(image_info)
                    state["current_query"] = description
                    state["explanations"].append(f"📸 Analyzed image: {description}")

        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            state["processed_input"]["image_error"] = str(e)

        self.performance_stats["processing_time"] += time.time() - start_time
        return state

    def _analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image and extract features"""
        try:
            if IMAGE_AVAILABLE:
                image = cv2.imread(image_path)
                if image is None:
                    return {"error": "Could not load image"}

                height, width = image.shape[:2]
                return {
                    "dimensions": (width, height),
                    "file_size": Path(image_path).stat().st_size,
                    "format": Path(image_path).suffix.lower()
                }
            else:
                return {"mock": True, "description": "Mock image analysis"}
        except Exception as e:
            return {"error": str(e)}

    def _generate_image_description(self, image_info: Dict[str, Any]) -> str:
        """Generate searchable description from image analysis"""
        if "error" in image_info:
            return "product from image"
        return "electronic device from image"


class IntentAnalysisAgent(BaseAgent):
    """Agent for analyzing user intent and context"""

    def __init__(self, config: AgentConfig):
        super().__init__("IntentAnalyzer", config)
        self.intent_patterns = {
            "browse": ["show me", "what do you have", "browse", "look at", "display"],
            "compare": ["compare", "difference", "vs", "versus", "better", "which"],
            "buy": ["buy", "purchase", "order", "get", "want", "need"],
            "research": ["reviews", "specs", "specifications", "details", "info", "about"],
            "budget": ["cheap", "affordable", "budget", "under", "less than", "low cost"],
            "premium": ["best", "premium", "high-end", "expensive", "luxury", "top"]
        }

    def execute(self, state: AgenticSearchState) -> AgenticSearchState:
        """Analyze user intent"""
        start_time = time.time()

        try:
            query = state["current_query"].lower()

            # Analyze intent
            intents = self._analyze_intent(query)
            state["user_intent"] = intents

            # Extract entities
            entities = self._extract_entities(query)
            state["user_intent"]["entities"] = entities

            # Update user profile
            self._update_user_profile(state)

            primary_intent = intents.get("primary", "search")
            state["explanations"].append(f"🎯 Detected intent: {primary_intent}")

        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            state["user_intent"] = {"primary": "search", "confidence": 0.5}

        self.performance_stats["processing_time"] += time.time() - start_time
        return state

    def _analyze_intent(self, query: str) -> Dict[str, Any]:
        """Analyze user intent from query"""
        intent_scores = {}

        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query)
            if score > 0:
                intent_scores[intent] = score

        if not intent_scores:
            return {"primary": "search", "confidence": 0.8}

        primary_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
        confidence = min(intent_scores[primary_intent] / len(query.split()), 1.0)

        return {
            "primary": primary_intent,
            "confidence": confidence,
            "all_scores": intent_scores
        }

    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from query"""
        entities = {}

        # Price extraction
        words = query.split()
        for i, word in enumerate(words):
            if word in ["under", "below", "less"] and i + 1 < len(words):
                try:
                    next_word = words[i + 1].replace('$', '').replace(',', '')
                    entities["max_price"] = float(next_word)
                    break
                except ValueError:
                    pass

        # Brand extraction
        common_brands = ["apple", "samsung", "sony", "microsoft", "google", "amazon", "dell", "hp"]
        for brand in common_brands:
            if brand in query:
                entities["brand"] = brand
                break

        # Category detection
        categories = {
            "laptop": ["laptop", "notebook", "computer"],
            "phone": ["phone", "smartphone", "mobile"],
            "headphones": ["headphone", "earphone", "headset"],
            "tv": ["tv", "television", "display", "monitor"]
        }

        for category, keywords in categories.items():
            if any(keyword in query for keyword in keywords):
                entities["category"] = category
                break

        return entities

    def _update_user_profile(self, state: AgenticSearchState) -> None:
        """Update user profile based on current interaction"""
        profile = state["user_profile"]
        intent = state["user_intent"]

        # Update interaction patterns
        primary_intent = intent.get("primary", "search")
        profile.interaction_patterns[primary_intent] = profile.interaction_patterns.get(primary_intent, 0) + 1

        # Update preferences based on entities
        entities = intent.get("entities", {})
        if "brand" in entities:
            brand = entities["brand"]
            if brand not in profile.preferred_brands:
                profile.preferred_brands.append(brand)

        if "category" in entities:
            category = entities["category"]
            if category not in profile.categories_of_interest:
                profile.categories_of_interest.append(category)

        profile.last_updated = datetime.now()


class SearchExecutionAgent(BaseAgent):
    """Agent responsible for executing searches"""

    def __init__(self, config: AgentConfig, db: ProductSeekerVectorDB, system_config: SystemConfig):
        super().__init__("SearchExecutor", config)
        self.db = db
        self.system_config = system_config

    def execute(self, state: AgenticSearchState) -> AgenticSearchState:
        """Execute search based on processed input"""
        start_time = time.time()

        try:
            search_type = state["search_type"]
            query = state["current_query"]

            if search_type == "image" and "image_path" in state["input_data"]:
                results = self._search_by_image(state["input_data"]["image_path"])
            else:
                results = self._search_by_text(query)

            # Enhance results with user profile
            enhanced_results = self._enhance_with_profile(results, state["user_profile"])

            state["search_results"] = enhanced_results
            state["explanations"].append(f"🔍 Retrieved {len(enhanced_results)} relevant products")

        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            state["search_results"] = []

        self.performance_stats["search_time"] += time.time() - start_time
        return state

    def _search_by_text(self, query: str) -> List[Dict[str, Any]]:
        """Execute text-based search"""
        try:
            result = self.db.search_by_text(query, n_results=self.system_config.max_results)
            return result.get('results', [])
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

    def _search_by_image(self, image_path: str) -> List[Dict[str, Any]]:
        """Execute image-based search"""
        try:
            result = self.db.search_by_image(image_path, n_results=self.system_config.max_results)
            return result.get('results', [])
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return []

    def _enhance_with_profile(self, results: List[Dict], profile: UserProfile) -> List[Dict]:
        """Enhance results based on user profile"""
        for result in results:
            metadata = result.get('metadata', {})

            # Calculate personalization score
            personalization_score = 0.0

            # Brand preference boost
            brand = metadata.get('brand', '').lower()
            if brand in [b.lower() for b in profile.preferred_brands]:
                personalization_score += 0.2

            # Category interest boost
            category = metadata.get('category', '').lower()
            if category in [c.lower() for c in profile.categories_of_interest]:
                personalization_score += 0.1

            result['personalization_score'] = personalization_score
            result['total_score'] = result.get('similarity', 0) + personalization_score

        # Sort by total score
        results.sort(key=lambda x: x.get('total_score', 0), reverse=True)
        return results


class RecommendationAgent(BaseAgent):
    """Agent for generating proactive recommendations"""

    def __init__(self, config: AgentConfig):
        super().__init__("RecommendationEngine", config)

    def execute(self, state: AgenticSearchState) -> AgenticSearchState:
        """Generate recommendations and suggestions"""
        start_time = time.time()

        try:
            suggestions = self._generate_suggestions(state)
            state["suggestions"].extend(suggestions)

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")

        self.performance_stats["processing_time"] += time.time() - start_time
        return state

    def _generate_suggestions(self, state: AgenticSearchState) -> List[str]:
        """Generate contextual suggestions"""
        suggestions = []
        intent = state["user_intent"]
        primary_intent = intent.get("primary", "search")

        if primary_intent == "browse":
            suggestions.extend([
                "💡 Try: 'Show me trending electronics'",
                "🔥 Try: 'What's popular in laptops'",
                "✨ Try: 'New arrivals this week'"
            ])
        elif primary_intent == "budget":
            suggestions.extend([
                "💰 Try: 'Best deals under $200'",
                "🏷️ Try: 'Budget gaming setup'",
                "📈 Try: 'Value for money products'"
            ])
        elif primary_intent == "compare":
            suggestions.extend([
                "⚖️ Try: 'Compare top 3 laptops'",
                "🔍 Try: 'Show alternatives'",
                "📊 Try: 'Feature comparison'"
            ])

        return suggestions[:3]


class ResponseFormattingAgent(BaseAgent):
    """Agent for formatting responses with explanations"""

    def __init__(self, config: AgentConfig, system_config: SystemConfig):
        super().__init__("ResponseFormatter", config)
        self.system_config = system_config

    def execute(self, state: AgenticSearchState) -> AgenticSearchState:
        """Format the final response"""
        start_time = time.time()

        try:
            results = state["search_results"]

            if not results:
                response = self._format_no_results_response(state)
            else:
                response = self._format_success_response(state)

            ai_message = AIMessage(content=response)
            if "messages" not in state:
                state["messages"] = []
            state["messages"].append(ai_message)

        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            if "messages" not in state:
                state["messages"] = []
            state["messages"].append(AIMessage(content=f"Error formatting response: {e}"))

        self.performance_stats["processing_time"] += time.time() - start_time
        return state

    def _format_no_results_response(self, state: AgenticSearchState) -> str:
        """Format response when no results found"""
        query = state["current_query"]
        return f"""😔 No products found for '{query}'.

🔧 **Let me help you differently:**
• Try broader search terms
• Use voice search with the microphone button
• Upload a product image for visual search
• Ask me to browse categories

🎯 **Quick suggestions:**
• Try "show me electronics"
• Upload an image for visual search
• Say "find me budget laptops"
"""

    def _format_success_response(self, state: AgenticSearchState) -> str:
        """Format successful search response"""
        results = state["search_results"]
        query = state["current_query"]
        search_type = state["search_type"]

        type_emoji = {
            "text": "📝",
            "voice": "🎤",
            "image": "📸",
            "hybrid": "🔀"
        }.get(search_type, "🔍")

        response = f"{type_emoji} Found {len(results)} products for '{query}'\n\n"

        for i, result in enumerate(results[:5], 1):
            metadata = result.get('metadata', {})
            similarity = result.get('similarity', 0)
            personalization = result.get('personalization_score', 0)

            response += f"**{i}. {metadata.get('title', 'Unknown Product')}**\n"
            response += f"   💰 Price: {metadata.get('price', 'N/A')}\n"
            response += f"   📂 Category: {metadata.get('category', 'N/A')}\n"
            response += f"   🏷️ Brand: {metadata.get('brand', 'N/A')}\n"
            response += f"   📊 Match: {similarity:.2f}"

            if personalization > 0:
                response += f" (👤 +{personalization:.1f} personal)"
            response += "\n"

            if metadata.get('description'):
                desc = metadata['description'][:100] + "..." if len(metadata['description']) > 100 else metadata[
                    'description']
                response += f"   📝 {desc}\n"

            response += "\n"

        response += "🎯 **What's next?**\n"
        response += "• Click on any product for details\n"
        response += "• Try voice: 'Show me cheaper alternatives'\n"
        response += "• Upload an image for visual search\n"
        response += "• Ask: 'Compare options 1 and 2'"

        return response


# Main Agentic System
class AgenticProductSearchSystem:
    """
    Advanced agentic product search system with multi-modal input
    """

    def __init__(self,
                 db_path: str = None,
                 collection_name: str = None,
                 model_name: str = None,
                 config: Optional[SystemConfig] = None):

        self.config = config or SystemConfig()

        # Use config defaults if parameters not provided
        db_path = db_path or self.config.database_path
        collection_name = collection_name or self.config.collection_name
        model_name = model_name or self.config.model_name

        self.session_id = f"session_{int(time.time())}"

        # Validate database path exists
        if not Path(db_path).exists():
            logger.warning(f"Database path does not exist: {db_path}")
            logger.info("Creating directory structure...")
            try:
                Path(db_path).mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created database directory: {db_path}")
            except Exception as e:
                logger.error(f"Failed to create database directory: {e}")

        # Initialize database
        logger.info(f"🔌 Connecting to database at: {db_path}")
        self.db = ProductSeekerVectorDB(
            db_path=db_path,
            collection_name=collection_name,
            model_name=model_name
        )

        # Initialize all agents
        self._initialize_agents()

        # User profile storage
        self.user_profiles = {}
        self.current_user = "default"

        # Build the agentic graph
        self.graph = self._build_agentic_graph()

        # Performance tracking
        self.system_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "average_response_time": 0.0,
            "user_satisfaction": 4.0,
            "database_path": db_path,
            "database_available": DATABASE_AVAILABLE
        }

        logger.info("🤖 Advanced Agentic Product Search System initialized")
        logger.info(f"📁 Database path: {db_path}")
        logger.info(f"📊 Collection: {collection_name}")
        logger.info(f"🧠 Model: {model_name}")

    def _initialize_agents(self):
        """Initialize all specialized agents"""
        base_config = AgentConfig(name="base", max_retries=3, enable_learning=True)

        self.voice_agent = VoiceProcessingAgent(
            AgentConfig(name="voice", max_retries=2),
            self.config
        )

        self.image_agent = ImageProcessingAgent(
            AgentConfig(name="image", max_retries=2)
        )

        self.intent_agent = IntentAnalysisAgent(
            AgentConfig(name="intent", max_retries=1, enable_learning=True)
        )

        self.search_agent = SearchExecutionAgent(
            AgentConfig(name="search", max_retries=3),
            self.db,
            self.config
        )

        self.recommendation_agent = RecommendationAgent(
            AgentConfig(name="recommendations", enable_learning=True)
        )

        self.response_agent = ResponseFormattingAgent(
            Agent
