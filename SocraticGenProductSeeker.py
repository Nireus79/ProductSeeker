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

# Database imports (using mock if not available)
try:
    from Vector import ProductSeekerVectorDB
    from Integrater import IntegratedProductScraper

    DATABASE_AVAILABLE = True
except ImportError:
    print("Warning: Database modules not found. Using mock implementations.")
    DATABASE_AVAILABLE = False


    class ProductSeekerVectorDB:
        def __init__(self, *args, **kwargs):
            self.mock_data = True
            print("🔄 Using mock database for demonstration")

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
                "last_updated": datetime.now().isoformat()
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
                 db_path: str = "mock_db",
                 collection_name: str = "products",
                 model_name: str = "mock_model",
                 config: Optional[SystemConfig] = None):

        self.config = config or SystemConfig()
        self.session_id = f"session_{int(time.time())}"

        # Initialize database
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
            "user_satisfaction": 4.0
        }

        logger.info("🤖 Advanced Agentic Product Search System initialized")

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
            AgentConfig(name="formatter"),
            self.config
        )

    def _build_agentic_graph(self):
        """Build the agentic workflow graph"""
        if LANGGRAPH_AVAILABLE:
            return self._build_langgraph_workflow()
        else:
            return self._build_simple_workflow()

    def _build_langgraph_workflow(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(AgenticSearchState)

        # Add nodes
        workflow.add_node("voice_processing", self.voice_agent.execute)
        workflow.add_node("image_processing", self.image_agent.execute)
        workflow.add_node("intent_analysis", self.intent_agent.execute)
        workflow.add_node("search_execution", self.search_agent.execute)
        workflow.add_node("recommendations", self.recommendation_agent.execute)
        workflow.add_node("response_formatting", self.response_agent.execute)

        # Set entry point
        workflow.set_entry_point("voice_processing")

        # Add edges
        workflow.add_edge("voice_processing", "image_processing")
        workflow.add_edge("image_processing", "intent_analysis")
        workflow.add_edge("intent_analysis", "search_execution")
        workflow.add_edge("search_execution", "recommendations")
        workflow.add_edge("recommendations", "response_formatting")
        workflow.add_edge("response_formatting", END)

        return workflow.compile()

    def _build_simple_workflow(self):
        """Build simplified workflow for when LangGraph is not available"""
        workflow = StateGraph(AgenticSearchState)

        workflow.add_node("voice_processing", self.voice_agent.execute)
        workflow.add_node("image_processing", self.image_agent.execute)
        workflow.add_node("intent_analysis", self.intent_agent.execute)
        workflow.add_node("search_execution", self.search_agent.execute)
        workflow.add_node("recommendations", self.recommendation_agent.execute)
        workflow.add_node("response_formatting", self.response_agent.execute)

        return workflow.compile()

    def get_user_profile(self, user_id: str = None) -> UserProfile:
        """Get or create user profile"""
        user_id = user_id or self.current_user
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        return self.user_profiles[user_id]

    def search(self,
               query: str = "",
               search_type: Literal["text", "voice", "image", "hybrid"] = "text",
               image_path: Optional[str] = None,
               audio_data: Optional[bytes] = None,
               user_id: str = None) -> Dict[str, Any]:
        """
        Main search method with agentic processing
        """
        start_time = time.time()

        try:
            # Initialize state
            state = AgenticSearchState(
                messages=[HumanMessage(content=query)],
                original_query=query,
                current_query=query,
                search_type=search_type,
                input_data={},
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

            # Add input data based on search type
            if image_path:
                state["input_data"]["image_path"] = image_path
            if audio_data:
                state["input_data"]["audio_data"] = audio_data

            # Execute agentic workflow
            final_state = self.graph.invoke(state)

            # Extract response
            response_content = ""
            if final_state.get("messages"):
                last_message = final_state["messages"][-1]
                if hasattr(last_message, 'content'):
                    response_content = last_message.content
                else:
                    response_content = str(last_message)

            # Update system stats
            processing_time = time.time() - start_time
            self.system_stats["total_queries"] += 1
            if final_state["search_results"]:
                self.system_stats["successful_queries"] += 1

            # Update average response time
            current_avg = self.system_stats["average_response_time"]
            total_queries = self.system_stats["total_queries"]
            self.system_stats["average_response_time"] = (
                    (current_avg * (total_queries - 1) + processing_time) / total_queries
            )

            return {
                "response": response_content,
                "results": final_state["search_results"],
                "explanations": final_state["explanations"],
                "suggestions": final_state["suggestions"],
                "user_intent": final_state["user_intent"],
                "processing_time": processing_time,
                "search_type": search_type,
                "success": len(final_state["search_results"]) > 0
            }

        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return {
                "response": f"❌ Search failed: {str(e)}",
                "results": [],
                "explanations": [f"Error: {str(e)}"],
                "suggestions": ["Try a different search term", "Check your input format"],
                "user_intent": {},
                "processing_time": time.time() - start_time,
                "search_type": search_type,
                "success": False
            }

    def voice_search(self, user_id: str = None) -> Dict[str, Any]:
        """Perform voice-based search"""
        if not self.voice_agent.available:
            return {
                "response": "🎤 Voice search is not available on this system",
                "results": [],
                "success": False
            }

        try:
            logger.info("🎤 Starting voice search...")
            # In a real implementation, this would capture audio
            # For demo purposes, we'll use a mock voice input
            mock_query = "gaming laptop under 1000 dollars"
            return self.search(query=mock_query, search_type="voice", user_id=user_id)
        except Exception as e:
            return {
                "response": f"🎤 Voice search failed: {str(e)}",
                "results": [],
                "success": False
            }

    def image_search(self, image_path: str, user_id: str = None) -> Dict[str, Any]:
        """Perform image-based search"""
        if not self.image_agent.available:
            return {
                "response": "📸 Image search is not available on this system",
                "results": [],
                "success": False
            }

        if not Path(image_path).exists():
            return {
                "response": f"📸 Image file not found: {image_path}",
                "results": [],
                "success": False
            }

        return self.search(query="", search_type="image", image_path=image_path, user_id=user_id)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = self.system_stats.copy()
        stats.update({
            "database_stats": self.db.get_database_stats(),
            "agent_performance": {
                agent.name: agent.get_stats() for agent in [
                    self.voice_agent, self.image_agent, self.intent_agent,
                    self.search_agent, self.recommendation_agent, self.response_agent
                ]
            },
            "active_users": len(self.user_profiles),
            "session_id": self.session_id
        })
        return stats

    def provide_feedback(self, query: str, rating: int, comments: str = "", user_id: str = None):
        """Collect user feedback for learning"""
        user_profile = self.get_user_profile(user_id)

        feedback = {
            "query": query,
            "rating": rating,
            "comments": comments,
            "timestamp": datetime.now(),
            "user_id": user_id or self.current_user
        }

        # Add to user profile
        user_profile.search_history.append(feedback)

        # Update system satisfaction score
        current_satisfaction = self.system_stats["user_satisfaction"]
        self.system_stats["user_satisfaction"] = (current_satisfaction * 0.9) + (rating * 0.1)

        # Distribute feedback to agents for learning
        for agent in [self.intent_agent, self.search_agent, self.recommendation_agent]:
            agent.learn_from_feedback(feedback)

        logger.info(f"📝 Feedback received: {rating}/5 stars")




    def get_user_profile(self, user_id: str = None) -> UserProfile:
        """Get or create user profile"""
        user_id = user_id or self.current_user
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        return self.user_profiles[user_id]


    def search(self,
               query: str = "",
               search_type: Literal["text", "voice", "image", "hybrid"] = "text",
               image_path: Optional[str] = None,
               audio_data: Optional[bytes] = None,
               user_id: str = None) -> Dict[str, Any]:
        """
        Main search method with agentic processing
        """
        start_time = time.time()

        try:
            # Initialize state
            state = AgenticSearchState(
                messages=[HumanMessage(content=query)],
                original_query=query,
                current_query=query,
                search_type=search_type,
                input_data={},
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

            # Add input data based on search type
            if image_path:
                state["input_data"]["image_path"] = image_path
            if audio_data:
                state["input_data"]["audio_data"] = audio_data

            # Execute agentic workflow
            final_state = self.graph.invoke(state)

            # Extract response
            response_content = ""
            if final_state.get("messages"):
                last_message = final_state["messages"][-1]
                if hasattr(last_message, 'content'):
                    response_content = last_message.content
                else:
                    response_content = str(last_message)

            # Update system stats
            processing_time = time.time() - start_time
            self.system_stats["total_queries"] += 1
            if final_state["search_results"]:
                self.system_stats["successful_queries"] += 1

            # Update average response time
            current_avg = self.system_stats["average_response_time"]
            total_queries = self.system_stats["total_queries"]
            self.system_stats["average_response_time"] = (
                    (current_avg * (total_queries - 1) + processing_time) / total_queries
            )

            return {
                "response": response_content,
                "results": final_state["search_results"],
                "explanations": final_state["explanations"],
                "suggestions": final_state["suggestions"],
                "user_intent": final_state["user_intent"],
                "processing_time": processing_time,
                "search_type": search_type,
                "success": len(final_state["search_results"]) > 0
            }

        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return {
                "response": f"❌ Search failed: {str(e)}",
                "results": [],
                "explanations": [f"Error: {str(e)}"],
                "suggestions": ["Try a different search term", "Check your input format"],
                "user_intent": {},
                "processing_time": time.time() - start_time,
                "search_type": search_type,
                "success": False
            }


    def voice_search(self, user_id: str = None) -> Dict[str, Any]:
        """Perform voice-based search"""
        if not self.voice_agent.available:
            return {
                "response": "🎤 Voice search is not available on this system",
                "results": [],
                "success": False
            }

        try:
            logger.info("🎤 Starting voice search...")
            # In a real implementation, this would capture audio
            # For demo purposes, we'll use a mock voice input
            mock_query = "gaming laptop under 1000 dollars"
            return self.search(query=mock_query, search_type="voice", user_id=user_id)
        except Exception as e:
            return {
                "response": f"🎤 Voice search failed: {str(e)}",
                "results": [],
                "success": False
            }


    def image_search(self, image_path: str, user_id: str = None) -> Dict[str, Any]:
        """Perform image-based search"""
        if not self.image_agent.available:
            return {
                "response": "📸 Image search is not available on this system",
                "results": [],
                "success": False
            }

        if not Path(image_path).exists():
            return {
                "response": f"📸 Image file not found: {image_path}",
                "results": [],
                "success": False
            }

        return self.search(query="", search_type="image", image_path=image_path, user_id=user_id)


    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = self.system_stats.copy()
        stats.update({
            "database_stats": self.db.get_database_stats(),
            "agent_performance": {
                agent.name: agent.get_stats() for agent in [
                    self.voice_agent, self.image_agent, self.intent_agent,
                    self.search_agent, self.recommendation_agent, self.response_agent
                ]
            },
            "active_users": len(self.user_profiles),
            "session_id": self.session_id
        })
        return stats


    def provide_feedback(self, query: str, rating: int, comments: str = "", user_id: str = None):
        """Collect user feedback for learning"""
        user_profile = self.get_user_profile(user_id)

        feedback = {
            "query": query,
            "rating": rating,
            "comments": comments,
            "timestamp": datetime.now(),
            "user_id": user_id or self.current_user
        }

        # Add to user profile
        user_profile.search_history.append(feedback)

        # Update system satisfaction score
        current_satisfaction = self.system_stats["user_satisfaction"]
        self.system_stats["user_satisfaction"] = (current_satisfaction * 0.9) + (rating * 0.1)

        # Distribute feedback to agents for learning
        for agent in [self.intent_agent, self.search_agent, self.recommendation_agent]:
            agent.learn_from_feedback(feedback)

        logger.info(f"📝 Feedback received: {rating}/5 stars")


# GUI Application Class
class ProductSearchGUI:
    """Enhanced GUI for the agentic product search system"""

    def __init__(self, search_system: AgenticProductSearchSystem):
        if not GUI_AVAILABLE:
            raise ImportError("tkinter not available for GUI")

        self.search_system = search_system
        self.root = tk.Tk()
        self.setup_gui()
        self.current_results = []

    def setup_gui(self):
        """Setup the GUI interface"""
        self.root.title("🤖 Agentic Product Search System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')

        # Create main frames
        self.create_header_frame()
        self.create_search_frame()
        self.create_results_frame()
        self.create_status_frame()

    def create_header_frame(self):
        """Create header with title and stats"""
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill='x', padx=10, pady=5)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="🤖 Agentic Product Search System",
            font=('Arial', 16, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(pady=20)

    def create_search_frame(self):
        """Create search input frame"""
        search_frame = tk.LabelFrame(self.root, text="Search Options", font=('Arial', 12, 'bold'))
        search_frame.pack(fill='x', padx=10, pady=5)

        # Text search
        text_frame = tk.Frame(search_frame)
        text_frame.pack(fill='x', padx=10, pady=5)

        tk.Label(text_frame, text="Search Query:", font=('Arial', 10)).pack(anchor='w')
        self.search_entry = tk.Entry(text_frame, font=('Arial', 11), width=50)
        self.search_entry.pack(side='left', padx=(0, 10), fill='x', expand=True)
        self.search_entry.bind('<Return>', lambda e: self.perform_text_search())

        tk.Button(
            text_frame,
            text="🔍 Search",
            command=self.perform_text_search,
            bg='#3498db',
            fg='white',
            font=('Arial', 10, 'bold')
        ).pack(side='right')

        # Voice and Image search buttons
        button_frame = tk.Frame(search_frame)
        button_frame.pack(fill='x', padx=10, pady=5)

        tk.Button(
            button_frame,
            text="🎤 Voice Search",
            command=self.perform_voice_search,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 10),
            width=15
        ).pack(side='left', padx=(0, 10))

        tk.Button(
            button_frame,
            text="📸 Image Search",
            command=self.perform_image_search,
            bg='#f39c12',
            fg='white',
            font=('Arial', 10),
            width=15
        ).pack(side='left', padx=(0, 10))

        tk.Button(
            button_frame,
            text="📊 System Stats",
            command=self.show_stats,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 10),
            width=15
        ).pack(side='right')

    def create_results_frame(self):
        """Create results display frame"""
        results_frame = tk.LabelFrame(self.root, text="Search Results", font=('Arial', 12, 'bold'))
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Create text widget with scrollbar
        text_frame = tk.Frame(results_frame)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.results_text = tk.Text(
            text_frame,
            font=('Arial', 10),
            wrap='word',
            bg='white',
            fg='black'
        )

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side='right', fill='y')

        self.results_text.pack(side='left', fill='both', expand=True)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)

        # Initial welcome message
        self.display_welcome_message()

    def create_status_frame(self):
        """Create status bar"""
        status_frame = tk.Frame(self.root, bg='#34495e', height=30)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(
            status_frame,
            text="Ready for search...",
            bg='#34495e',
            fg='white',
            font=('Arial', 9)
        )
        self.status_label.pack(side='left', padx=10, pady=5)

    def display_welcome_message(self):
        """Display welcome message"""
        welcome_text = """🤖 Welcome to the Agentic Product Search System!

🎯 **How to use:**
• Type your search query and press Enter or click 'Search'
• Use voice search by clicking the microphone button
• Upload product images for visual search
• View system statistics and performance metrics

🔥 **Try these examples:**
• "gaming laptop under $1000"
• "wireless headphones with noise cancellation" 
• "smartphone with good camera"
• "budget office chair"

💡 **Advanced features:**
• The system learns from your preferences
• Get personalized recommendations
• Multi-modal search (text, voice, image)
• Intelligent intent analysis

Ready when you are! 🚀
"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, welcome_text)

    def perform_text_search(self):
        """Perform text-based search"""
        query = self.search_entry.get().strip()
        if not query:
            messagebox.showwarning("Input Required", "Please enter a search query")
            return

        self.status_label.config(text="Searching...")
        self.root.update()

        try:
            result = self.search_system.search(query, search_type="text")
            self.display_results(result)
            self.current_results = result
        except Exception as e:
            messagebox.showerror("Search Error", f"Search failed: {str(e)}")
        finally:
            self.status_label.config(text="Search completed")

    def perform_voice_search(self):
        """Perform voice-based search"""
        self.status_label.config(text="Listening...")
        self.root.update()

        try:
            result = self.search_system.voice_search()
            self.display_results(result)
            self.current_results = result
        except Exception as e:
            messagebox.showerror("Voice Search Error", f"Voice search failed: {str(e)}")
        finally:
            self.status_label.config(text="Voice search completed")

    def perform_image_search(self):
        """Perform image-based search"""
        file_path = filedialog.askopenfilename(
            title="Select Product Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        self.status_label.config(text="Analyzing image...")
        self.root.update()

        try:
            result = self.search_system.image_search(file_path)
            self.display_results(result)
            self.current_results = result
        except Exception as e:
            messagebox.showerror("Image Search Error", f"Image search failed: {str(e)}")
        finally:
            self.status_label.config(text="Image search completed")

    def display_results(self, result: Dict[str, Any]):
        """Display search results"""
        self.results_text.delete(1.0, tk.END)

        # Display main response
        response = result.get("response", "No response generated")
        self.results_text.insert(tk.END, response + "\n\n")

        # Display explanations if available
        explanations = result.get("explanations", [])
        if explanations:
            self.results_text.insert(tk.END, "🔍 **Process Explanations:**\n")
            for explanation in explanations:
                self.results_text.insert(tk.END, f"  • {explanation}\n")
            self.results_text.insert(tk.END, "\n")

        # Display suggestions if available
        suggestions = result.get("suggestions", [])
        if suggestions:
            self.results_text.insert(tk.END, "💡 **Suggestions:**\n")
            for suggestion in suggestions:
                self.results_text.insert(tk.END, f"  {suggestion}\n")
            self.results_text.insert(tk.END, "\n")

        # Display performance metrics
        processing_time = result.get("processing_time", 0)
        num_results = len(result.get("results", []))
        self.results_text.insert(tk.END, f"⚡ Processed in {processing_time:.2f}s | Found {num_results} results\n")

    def show_stats(self):
        """Show system statistics"""
        stats = self.search_system.get_system_stats()

        stats_window = tk.Toplevel(self.root)
        stats_window.title("📊 System Statistics")
        stats_window.geometry("600x400")
        stats_window.configure(bg='#f0f0f0')

        # Create text widget for stats
        text_widget = tk.Text(stats_window, font=('Courier', 10), bg='white')
        scrollbar = tk.Scrollbar(stats_window)

        scrollbar.pack(side='right', fill='y')
        text_widget.pack(side='left', fill='both', expand=True)

        text_widget.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=text_widget.yview)

        # Format and display stats
        stats_text = self.format_stats(stats)
        text_widget.insert(tk.END, stats_text)
        text_widget.config(state='disabled')

    def format_stats(self, stats: Dict[str, Any]) -> str:
        """Format statistics for display"""
        formatted = "📊 SYSTEM STATISTICS\n"
        formatted += "=" * 50 + "\n\n"

        # Basic stats
        formatted += "🎯 QUERY STATISTICS:\n"
        formatted += f"  Total Queries: {stats.get('total_queries', 0)}\n"
        formatted += f"  Successful Queries: {stats.get('successful_queries', 0)}\n"

        if stats.get('total_queries', 0) > 0:
            success_rate = (stats.get('successful_queries', 0) / stats.get('total_queries', 1)) * 100
            formatted += f"  Success Rate: {success_rate:.1f}%\n"

        formatted += f"  Average Response Time: {stats.get('average_response_time', 0):.2f}s\n"
        formatted += f"  User Satisfaction: {stats.get('user_satisfaction', 0):.1f}/5.0\n\n"

        # Database stats
        db_stats = stats.get('database_stats', {})
        formatted += "💾 DATABASE STATISTICS:\n"
        formatted += f"  Total Products: {db_stats.get('total_products', 0)}\n"
        formatted += f"  Collections: {db_stats.get('collections', 0)}\n"
        formatted += f"  Last Updated: {db_stats.get('last_updated', 'Unknown')}\n\n"

        # Agent performance
        agent_perf = stats.get('agent_performance', {})
        if agent_perf:
            formatted += "🤖 AGENT PERFORMANCE:\n"
            for agent_name, perf in agent_perf.items():
                formatted += f"  {agent_name}:\n"
                for metric, value in perf.items():
                    if isinstance(value, float):
                        formatted += f"    {metric}: {value:.3f}\n"
                    else:
                        formatted += f"    {metric}: {value}\n"
            formatted += "\n"

        # System info
        formatted += "ℹ️ SYSTEM INFO:\n"
        formatted += f"  Session ID: {stats.get('session_id', 'Unknown')}\n"
        formatted += f"  Active Users: {stats.get('active_users', 0)}\n"
        formatted += f"  Voice Available: {'Yes' if AUDIO_AVAILABLE else 'No'}\n"
        formatted += f"  Image Available: {'Yes' if IMAGE_AVAILABLE else 'No'}\n"
        formatted += f"  LangGraph Available: {'Yes' if LANGGRAPH_AVAILABLE else 'No'}\n"

        return formatted

    def run(self):
        """Start the GUI application"""
        logger.info("🖥️ Starting GUI application")
        self.root.mainloop()


# Command Line Interface
class ProductSearchCLI:
    """Command-line interface for the search system"""

    def __init__(self, search_system: AgenticProductSearchSystem):
        self.search_system = search_system
        self.running = True

    def show_banner(self):
        """Display welcome banner"""
        print("\n" + "=" * 60)
        print("🤖 AGENTIC PRODUCT SEARCH SYSTEM")
        print("=" * 60)
        print("🎯 Multi-modal AI-powered product search")
        print("🔥 Voice, Text, and Image search capabilities")
        print("💡 Intelligent recommendations and learning")
        print("=" * 60)

    def show_help(self):
        """Display help information"""
        help_text = """
📖 AVAILABLE COMMANDS:

🔍 SEARCH COMMANDS:
  search <query>     - Text search for products
  voice              - Start voice search (if available)  
  image <path>       - Search using product image

💡 UTILITY COMMANDS:
  help               - Show this help message
  stats              - Display system statistics
  clear              - Clear screen
  quit/exit          - Exit the application

📝 EXAMPLES:
  search gaming laptop under $1000
  search wireless headphones noise cancellation
  image /path/to/product-image.jpg
  voice
  stats

🎯 TIPS:
  • Be specific with your search queries
  • Use natural language (e.g., "cheap smartphones")
  • Try different search modes for best results
  • The system learns from your interactions
"""
        print(help_text)

    def run(self):
        """Start the CLI application"""
        self.show_banner()

        while self.running:
            try:
                user_input = input("\n🔍 Search> ").strip()

                if not user_input:
                    continue

                self.process_command(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using the Agentic Product Search System!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def process_command(self, user_input: str):
        """Process user command"""
        parts = user_input.split()
        command = parts[0].lower()

        if command in ['quit', 'exit']:
            self.running = False
            print("👋 Goodbye!")

        elif command == 'help':
            self.show_help()

        elif command == 'clear':
            os.system('cls' if os.name == 'nt' else 'clear')
            self.show_banner()

        elif command == 'stats':
            self.show_stats()

        elif command == 'search':
            if len(parts) > 1:
                query = ' '.join(parts[1:])
                self.perform_search(query)
            else:
                print("❌ Please provide a search query")

        elif command == 'voice':
            self.perform_voice_search()

        elif command == 'image':
            if len(parts) > 1:
                image_path = parts[1]
                self.perform_image_search(image_path)
            else:
                print("❌ Please provide image path")

        else:
            # Treat as search query
            self.perform_search(user_input)

    def perform_search(self, query: str):
        """Perform text search"""
        print(f"🔍 Searching for: '{query}'...")

        try:
            result = self.search_system.search(query, search_type="text")
            self.display_result(result)
        except Exception as e:
            print(f"❌ Search failed: {e}")

    def perform_voice_search(self):
        """Perform voice search"""
        print("🎤 Starting voice search...")

        try:
            result = self.search_system.voice_search()
            self.display_result(result)
        except Exception as e:
            print(f"❌ Voice search failed: {e}")

    def perform_image_search(self, image_path: str):
        """Perform image search"""
        print(f"📸 Analyzing image: {image_path}...")

        try:
            result = self.search_system.image_search(image_path)
            self.display_result(result)
        except Exception as e:
            print(f"❌ Image search failed: {e}")

    def display_result(self, result: Dict[str, Any]):
        """Display search result"""
        print("\n" + "=" * 60)

        # Main response
        response = result.get("response", "No response")
        print(response)

        # Processing info
        processing_time = result.get("processing_time", 0)
        num_results = len(result.get("results", []))
        success = result.get("success", False)

        status = "✅ Success" if success else "❌ Failed"
        print(f"\n{status} | {num_results} results | {processing_time:.2f}s")

        print("=" * 60)

    def show_stats(self):
        """Display system statistics"""
        print("\n📊 SYSTEM STATISTICS")
        print("=" * 40)

        stats = self.search_system.get_system_stats()

        print(f"Total Queries: {stats.get('total_queries', 0)}")
        print(f"Successful Queries: {stats.get('successful_queries', 0)}")

        if stats.get('total_queries', 0) > 0:
            success_rate = (stats.get('successful_queries', 0) / stats.get('total_queries', 1)) * 100
            print(f"Success Rate: {success_rate:.1f}%")

        print(f"Avg Response Time: {stats.get('average_response_time', 0):.2f}s")
        print(f"User Satisfaction: {stats.get('user_satisfaction', 0):.1f}/5.0")
        print("=" * 40)


# Main execution and launcher
def main():
    """Main application launcher"""
    import argparse

    parser = argparse.ArgumentParser(
        description="🤖 Agentic Product Search System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python SocraticGenProductSeeker.py --mode gui
  python SocraticGenProductSeeker.py --mode cli  
  python SocraticGenProductSeeker.py --mode gui --voice-enabled
  python SocraticGenProductSeeker.py --help
        """
    )

    parser.add_argument(
        '--mode',
        choices=['gui', 'cli', 'both'],
        default='gui',
        help='Interface mode (default: gui)'
    )

    parser.add_argument(
        '--db-path',
        default='./product_search_db',
        help='Database path (default: ./product_search_db)'
    )

    parser.add_argument(
        '--collection-name',
        default='products',
        help='Collection name (default: products)'
    )

    parser.add_argument(
        '--model-name',
        default='all-MiniLM-L6-v2',
        help='Embedding model name (default: all-MiniLM-L6-v2)'
    )

    parser.add_argument(
        '--voice-enabled',
        action='store_true',
        help='Enable voice features'
    )

    parser.add_argument(
        '--image-enabled',
        action='store_true',
        help='Enable image features'
    )

    parser.add_argument(
        '--max-results',
        type=int,
        default=20,
        help='Maximum search results (default: 20)'
    )

    args = parser.parse_args()

    # Configure system
    config = SystemConfig(
        max_results=args.max_results,
        voice_enabled=args.voice_enabled and AUDIO_AVAILABLE,
        image_enabled=args.image_enabled and IMAGE_AVAILABLE
    )

    # Initialize search system
    try:
        print("🚀 Initializing Agentic Product Search System...")
        search_system = AgenticProductSearchSystem(
            db_path=args.db_path,
            collection_name=args.collection_name,
            model_name=args.model_name,
            config=config
        )
        print("✅ System initialized successfully!")

    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        sys.exit(1)

    # Launch interface(s)
    if args.mode == 'gui' or args.mode == 'both':
        if GUI_AVAILABLE:
            try:
                gui = ProductSearchGUI(search_system)
                if args.mode == 'gui':
                    gui.run()
                else:
                    # Run GUI in separate thread for 'both' mode
                    gui_thread = threading.Thread(target=gui.run, daemon=True)
                    gui_thread.start()
            except Exception as e:
                print(f"❌ GUI failed to start: {e}")
                if args.mode == 'gui':
                    sys.exit(1)
        else:
            print("❌ GUI not available - tkinter not installed")
            if args.mode == 'gui':
                args.mode = 'cli'  # Fallback to CLI

    if args.mode == 'cli' or args.mode == 'both':
        try:
            cli = ProductSearchCLI(search_system)
            cli.run()
        except Exception as e:
            print(f"❌ CLI failed to start: {e}")
            sys.exit(1)


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Launch the application
    main()
