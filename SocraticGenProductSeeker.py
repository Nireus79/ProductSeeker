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
                        'description': f'High-quality {query} with advanced features and excellent performance. '
                                       f'Perfect for both professional and personal use.',
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
            AgentConfig(name="response", max_retries=1),
            self.config
        )

        logger.info("🤖 All agents initialized successfully")

    def _build_agentic_graph(self):
        """Build the agentic workflow graph"""
        if not LANGGRAPH_AVAILABLE:
            # Create simplified workflow
            workflow = StateGraph(AgenticSearchState)
        else:
            workflow = StateGraph(AgenticSearchState)

        # Add nodes for each agent
        workflow.add_node("voice_processing", self.voice_agent.execute)
        workflow.add_node("image_processing", self.image_agent.execute)
        workflow.add_node("intent_analysis", self.intent_agent.execute)
        workflow.add_node("search_execution", self.search_agent.execute)
        workflow.add_node("recommendations", self.recommendation_agent.execute)
        workflow.add_node("response_formatting", self.response_agent.execute)

        # Define the flow
        workflow.set_entry_point("intent_analysis")

        # Conditional routing based on input type
        def route_by_input_type(state):
            search_type = state.get("search_type", "text")
            if search_type == "voice":
                return "voice_processing"
            elif search_type == "image":
                return "image_processing"
            else:
                return "search_execution"

        workflow.add_conditional_edges(
            "intent_analysis",
            route_by_input_type,
            {
                "voice_processing": "voice_processing",
                "image_processing": "image_processing",
                "search_execution": "search_execution"
            }
        )

        # Route from voice/image processing to search
        workflow.add_edge("voice_processing", "search_execution")
        workflow.add_edge("image_processing", "search_execution")

        # Final steps
        workflow.add_edge("search_execution", "recommendations")
        workflow.add_edge("recommendations", "response_formatting")
        workflow.add_edge("response_formatting", END)

        return workflow.compile()

    def get_user_profile(self, user_id: str = None) -> UserProfile:
        """Get or create user profile"""
        user_id = user_id or self.current_user

        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)

        return self.user_profiles[user_id]

    async def search_async(self,
                           query: str = None,
                           search_type: str = "text",
                           image_path: str = None,
                           audio_data: bytes = None,
                           user_id: str = None) -> Dict[str, Any]:
        """Async search method"""
        return await asyncio.to_thread(
            self.search,
            query=query,
            search_type=search_type,
            image_path=image_path,
            audio_data=audio_data,
            user_id=user_id
        )

    def search(self,
               query: str = None,
               search_type: str = "text",
               image_path: str = None,
               audio_data: bytes = None,
               user_id: str = None,
               **kwargs) -> Dict[str, Any]:
        """
        Main search method with multi-modal support

        Args:
            query: Text query for search
            search_type: Type of search ('text', 'voice', 'image', 'hybrid')
            image_path: Path to image file for image search
            audio_data: Audio data for voice search
            user_id: User identifier for personalization
            **kwargs: Additional parameters

        Returns:
            Search results with explanations and metadata
        """
        start_time = time.time()
        self.system_stats["total_queries"] += 1

        try:
            # Prepare input data
            input_data = {}
            if query:
                input_data["text_query"] = query
            if image_path:
                input_data["image_path"] = image_path
            if audio_data:
                input_data["audio_data"] = audio_data

            # Initialize search state
            state = AgenticSearchState(
                messages=[HumanMessage(content=query or "Search request")],
                original_query=query or "",
                current_query=query or "",
                search_type=search_type,
                input_data=input_data,
                processed_input={},
                search_results=[],
                user_intent={},
                user_profile=self.get_user_profile(user_id),
                performance_metrics={},
                explanations=[],
                suggestions=[],
                refinement_count=0,
                session_id=self.session_id,
                timestamp=datetime.now()
            )

            # Execute the agentic workflow
            logger.info(f"🚀 Starting {search_type} search: '{query}'")
            final_state = self.graph.invoke(state)

            # Calculate metrics
            response_time = time.time() - start_time
            self._update_system_stats(response_time, True)

            # Extract response
            response_content = ""
            if final_state.get("messages"):
                last_message = final_state["messages"][-1]
                response_content = getattr(last_message, 'content', str(last_message))

            result = {
                "success": True,
                "results": final_state.get("search_results", []),
                "response": response_content,
                "explanations": final_state.get("explanations", []),
                "suggestions": final_state.get("suggestions", []),
                "user_intent": final_state.get("user_intent", {}),
                "search_type": search_type,
                "response_time": response_time,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"✅ Search completed in {response_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            self._update_system_stats(time.time() - start_time, False)

            return {
                "success": False,
                "error": str(e),
                "results": [],
                "response": f"❌ Search failed: {e}",
                "explanations": [f"Error occurred: {e}"],
                "suggestions": ["Try a simpler query", "Check your input format"],
                "response_time": time.time() - start_time,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat()
            }

    def search_by_voice(self, duration: int = 5) -> Dict[str, Any]:
        """Record voice and perform search"""
        if not self.voice_agent.available:
            return {
                "success": False,
                "error": "Voice processing not available",
                "response": "🎤 Voice search is not available. Please use text search instead."
            }

        logger.info(f"🎤 Starting voice recording for {duration} seconds...")

        # Simplified voice recording simulation
        mock_audio = b"mock_audio_data"

        return self.search(
            search_type="voice",
            audio_data=mock_audio
        )

    def search_by_image(self, image_path: str) -> Dict[str, Any]:
        """Search using image input"""
        if not Path(image_path).exists():
            return {
                "success": False,
                "error": f"Image file not found: {image_path}",
                "response": "📸 Image file not found. Please check the file path."
            }

        return self.search(
            search_type="image",
            image_path=image_path
        )

    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics"""
        try:
            stats = self.db.get_database_stats()
            stats.update({
                "system_stats": self.system_stats,
                "agents_available": {
                    "voice": self.voice_agent.available,
                    "image": self.image_agent.available,
                    "database": DATABASE_AVAILABLE
                }
            })
            return stats
        except Exception as e:
            return {
                "error": str(e),
                "system_stats": self.system_stats
            }

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "system_ready": True,
            "database_connected": DATABASE_AVAILABLE,
            "voice_available": self.voice_agent.available,
            "image_available": self.image_agent.available,
            "langgraph_available": LANGGRAPH_AVAILABLE,
            "session_id": self.session_id,
            "stats": self.system_stats,
            "config": {
                "max_results": self.config.max_results,
                "voice_enabled": self.config.voice_enabled,
                "image_enabled": self.config.image_enabled,
                "learning_enabled": self.config.learning_enabled
            }
        }

    def _update_system_stats(self, response_time: float, success: bool):
        """Update system performance statistics"""
        if success:
            self.system_stats["successful_queries"] += 1

        # Update average response time
        total_queries = self.system_stats["total_queries"]
        current_avg = self.system_stats["average_response_time"]
        new_avg = ((current_avg * (total_queries - 1)) + response_time) / total_queries
        self.system_stats["average_response_time"] = new_avg

    def speak_response(self, text: str):
        """Use TTS to speak the response"""
        if self.voice_agent.available:
            self.voice_agent.speak(text)
        else:
            logger.info("TTS not available")


class ProductSearchGUI:
    """Enhanced GUI for the product search system"""

    def __init__(self):
        if not GUI_AVAILABLE:
            raise RuntimeError("GUI not available - tkinter not installed")

        self.search_system = None
        self.setup_gui()
        self.initialize_search_system()

    def setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("🤖 Advanced Product Search System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')

        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame, text="🤖 Advanced Agentic Product Search",
                                font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)

        # Search input frame
        search_frame = ttk.LabelFrame(main_frame, text="Search Input", padding="10")
        search_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # Text search
        ttk.Label(search_frame, text="Text Query:").grid(row=0, column=0, sticky=tk.W)
        self.query_var = tk.StringVar(value="gaming laptop under $1000")
        self.query_entry = ttk.Entry(search_frame, textvariable=self.query_var, width=50)
        self.query_entry.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))

        # Buttons frame
        buttons_frame = ttk.Frame(search_frame)
        buttons_frame.grid(row=1, column=0, columnspan=3, pady=10)

        # Search buttons
        self.search_btn = ttk.Button(buttons_frame, text="🔍 Search", command=self.text_search)
        self.search_btn.pack(side=tk.LEFT, padx=2)

        self.voice_btn = ttk.Button(buttons_frame, text="🎤 Voice Search", command=self.voice_search)
        self.voice_btn.pack(side=tk.LEFT, padx=2)

        self.image_btn = ttk.Button(buttons_frame, text="📸 Image Search", command=self.image_search)
        self.image_btn.pack(side=tk.LEFT, padx=2)

        self.clear_btn = ttk.Button(buttons_frame, text="🗑️ Clear", command=self.clear_results)
        self.clear_btn.pack(side=tk.LEFT, padx=2)

        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding="5")
        status_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        self.status_text = tk.Text(status_frame, height=4, width=80)
        self.status_text.pack(fill=tk.BOTH, expand=True)

        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Search Results", padding="5")
        results_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        # Results text with scrollbar
        results_container = ttk.Frame(results_frame)
        results_container.pack(fill=tk.BOTH, expand=True)

        self.results_text = tk.Text(results_container, height=20, width=80, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(results_container, orient=tk.VERTICAL, command=self.results_text.yview)

        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # Bind Enter key to search
        self.query_entry.bind('<Return>', lambda e: self.text_search())

    def initialize_search_system(self):
        """Initialize the search system"""
        try:
            self.update_status("🔄 Initializing search system...")
            self.search_system = AgenticProductSearchSystem()

            # Get system status
            status = self.search_system.get_system_status()
            status_msg = "✅ System Ready\n"
            status_msg += f"📊 Database: {'✅' if status['database_connected'] else '❌'}\n"
            status_msg += f"🎤 Voice: {'✅' if status['voice_available'] else '❌'}\n"
            status_msg += f"📸 Image: {'✅' if status['image_available'] else '❌'}"

            self.update_status(status_msg)

        except Exception as e:
            error_msg = f"❌ Initialization failed: {str(e)}"
            self.update_status(error_msg)
            messagebox.showerror("Initialization Error", str(e))

    def update_status(self, message: str):
        """Update status display"""
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(1.0, message)
        self.root.update_idletasks()

    def update_results(self, result: Dict[str, Any]):
        """Update results display"""
        self.results_text.delete(1.0, tk.END)

        if result.get("success"):
            response = result.get("response", "No response")
            self.results_text.insert(tk.END, response)

            # Add explanations if available
            explanations = result.get("explanations", [])
            if explanations:
                self.results_text.insert(tk.END, "\n\n🔍 **Process Explanation:**\n")
                for explanation in explanations:
                    self.results_text.insert(tk.END, f"• {explanation}\n")

            # Add suggestions if available
            suggestions = result.get("suggestions", [])
            if suggestions:
                self.results_text.insert(tk.END, "\n💡 **Suggestions:**\n")
                for suggestion in suggestions:
                    self.results_text.insert(tk.END, f"• {suggestion}\n")

        else:
            error_msg = result.get("error", "Unknown error")
            self.results_text.insert(tk.END, f"❌ Error: {error_msg}")

        # Scroll to top
        self.results_text.see(1.0)

    def text_search(self):
        """Perform text search"""
        if not self.search_system:
            messagebox.showerror("Error", "Search system not initialized")
            return

        query = self.query_var.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a search query")
            return

        self.update_status(f"🔍 Searching for: {query}")
        self.search_btn.configure(state="disabled")

        try:
            result = self.search_system.search(query=query, search_type="text")
            self.update_results(result)

            response_time = result.get("response_time", 0)
            self.update_status(f"✅ Search completed in {response_time:.2f}s")

        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            self.update_status(f"❌ {error_msg}")
            messagebox.showerror("Search Error", error_msg)
        finally:
            self.search_btn.configure(state="normal")

    def voice_search(self):
        """Perform voice search"""
        if not self.search_system:
            messagebox.showerror("Error", "Search system not initialized")
            return

        self.update_status("🎤 Starting voice search...")
        self.voice_btn.configure(state="disabled")

        try:
            result = self.search_system.search_by_voice(duration=3)
            self.update_results(result)

            if result.get("success"):
                response_time = result.get("response_time", 0)
                self.update_status(f"✅ Voice search completed in {response_time:.2f}s")
            else:
                self.update_status("❌ Voice search failed")

        except Exception as e:
            error_msg = f"Voice search failed: {str(e)}"
            self.update_status(f"❌ {error_msg}")
            messagebox.showerror("Voice Search Error", error_msg)
        finally:
            self.voice_btn.configure(state="normal")

    def image_search(self):
        """Perform image search"""
        if not self.search_system:
            messagebox.showerror("Error", "Search system not initialized")
            return

        # File dialog for image selection
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]

        image_path = filedialog.askopenfilename(
            title="Select Product Image",
            filetypes=file_types
        )

        if not image_path:
            return

        self.update_status(f"📸 Analyzing image: {Path(image_path).name}")
        self.image_btn.configure(state="disabled")

        try:
            result = self.search_system.search_by_image(image_path)
            self.update_results(result)

            if result.get("success"):
                response_time = result.get("response_time", 0)
                self.update_status(f"✅ Image search completed in {response_time:.2f}s")
            else:
                self.update_status("❌ Image search failed")

        except Exception as e:
            error_msg = f"Image search failed: {str(e)}"
            self.update_status(f"❌ {error_msg}")
            messagebox.showerror("Image Search Error", error_msg)
        finally:
            self.image_btn.configure(state="normal")

    def clear_results(self):
        """Clear all results and reset interface"""
        self.results_text.delete(1.0, tk.END)
        self.query_var.set("")
        self.update_status("🗑️ Results cleared")

    def run(self):
        """Start the GUI application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ GUI Error: {e}")


# CLI Interface
class ProductSearchCLI:
    """Command-line interface for the product search system"""

    def __init__(self):
        self.search_system = None
        self.initialize_system()

    def initialize_system(self):
        """Initialize the search system"""
        try:
            print("🤖 Advanced Agentic Product Search System")
            print("=" * 50)
            print("🔄 Initializing system...")

            self.search_system = AgenticProductSearchSystem()

            status = self.search_system.get_system_status()
            print(f"✅ System ready!")
            print(f"📊 Database: {'✅' if status['database_connected'] else '❌'}")
            print(f"🎤 Voice: {'✅' if status['voice_available'] else '❌'}")
            print(f"📸 Image: {'✅' if status['image_available'] else '❌'}")
            print()

        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            sys.exit(1)

    def print_help(self):
        """Print help information"""
        print("""
🤖 **Available Commands:**
• search <query>       - Text search
• voice                - Voice search (if available)
• image <path>         - Image search
• status               - Show system status
• stats                - Show statistics
• help                 - Show this help
• quit/exit            - Exit program

📝 **Example queries:**
• search gaming laptop under $1000
• search best wireless headphones
• image /path/to/product.jpg
""")

    def run(self):
        """Run the CLI interface"""
        self.print_help()

        while True:
            try:
                user_input = input("\n🔍 Enter command: ").strip()

                if not user_input:
                    continue

                parts = user_input.split(None, 1)
                command = parts[0].lower()

                if command in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                elif command == 'help':
                    self.print_help()

                elif command == 'status':
                    self.show_status()

                elif command == 'stats':
                    self.show_stats()

                elif command == 'search':
                    if len(parts) > 1:
                        self.text_search(parts[1])
                    else:
                        print("❌ Please provide a search query")

                elif command == 'voice':
                    self.voice_search()

                elif command == 'image':
                    if len(parts) > 1:
                        self.image_search(parts[1])
                    else:
                        print("❌ Please provide an image path")

                else:
                    # Treat as direct search query
                    self.text_search(user_input)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def text_search(self, query: str):
        """Perform text search"""
        print(f"🔍 Searching for: {query}")

        try:
            result = self.search_system.search(query=query, search_type="text")
            self.print_result(result)

        except Exception as e:
            print(f"❌ Search failed: {e}")

    def voice_search(self):
        """Perform voice search"""
        print("🎤 Starting voice search...")

        try:
            result = self.search_system.search_by_voice(duration=3)
            self.print_result(result)

        except Exception as e:
            print(f"❌ Voice search failed: {e}")

    def image_search(self, image_path: str):
        """Perform image search"""
        print(f"📸 Analyzing image: {image_path}")

        try:
            result = self.search_system.search_by_image(image_path)
            self.print_result(result)

        except Exception as e:
            print(f"❌ Image search failed: {e}")

    def print_result(self, result: Dict[str, Any]):
        """Print search result"""
        if result.get("success"):
            response = result.get("response", "")
            print(f"\n{response}")

            # Print explanations
            explanations = result.get("explanations", [])
            if explanations:
                print(f"\n🔍 **Process Explanation:**")
                for explanation in explanations:
                    print(f"  • {explanation}")

            # Print timing info
            response_time = result.get("response_time", 0)
            print(f"\n⏱️  Response time: {response_time:.2f}s")

        else:
            error = result.get("error", "Unknown error")
            print(f"❌ {error}")

    def show_status(self):
        """Show system status"""
        try:
            status = self.search_system.get_system_status()
            print("\n📊 **System Status:**")
            print(f"  • Database: {'✅ Connected' if status['database_connected'] else '❌ Not available'}")
            print(f"  • Voice: {'✅ Available' if status['voice_available'] else '❌ Not available'}")
            print(f"  • Image: {'✅ Available' if status['image_available'] else '❌ Not available'}")
            print(f"  • Session ID: {status['session_id']}")

        except Exception as e:
            print(f"❌ Status check failed: {e}")

    def show_stats(self):
        """Show system statistics"""
        try:
            db_info = self.search_system.get_database_info()
            stats = db_info.get("system_stats", {})

            print("\n📈 **System Statistics:**")
            print(f"  • Total queries: {stats.get('total_queries', 0)}")
            print(f"  • Successful queries: {stats.get('successful_queries', 0)}")
            print(f"  • Average response time: {stats.get('average_response_time', 0):.2f}s")
            print(f"  • User satisfaction: {stats.get('user_satisfaction', 0):.1f}/5.0")

            if "total_products" in db_info:
                print(f"  • Database products: {db_info['total_products']:,}")

        except Exception as e:
            print(f"❌ Stats retrieval failed: {e}")


# Main execution
def main():
    """Main function to run the application"""
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Agentic Product Search System")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    parser.add_argument("--db-path", help="Database path")
    parser.add_argument("--collection", default="products", help="Collection name")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model")

    args = parser.parse_args()

    # Configuration
    config = SystemConfig()
    if args.db_path:
        config.database_path = args.db_path
    if args.collection:
        config.collection_name = args.collection
    if args.model:
        config.model_name = args.model

    try:
        if args.cli or not GUI_AVAILABLE:
            # Run CLI interface
            cli = ProductSearchCLI()
            cli.run()
        else:
            # Run GUI interface
            gui = ProductSearchGUI()
            gui.run()

    except Exception as e:
        print(f"❌ Application failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
