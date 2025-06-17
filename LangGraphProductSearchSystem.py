import logging
import asyncio
from typing import List, Dict, Any, Optional, TypedDict, Annotated, Union, Literal
from pathlib import Path
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time
from functools import lru_cache
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

# Import your existing classes
from Vector import ProductSeekerVectorDB
from Integrater import IntegratedProductScraper

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for search operations"""
    max_results: int = 15
    min_similarity_threshold: float = 0.5
    max_refinements: int = 2
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    enable_parallel_search: bool = True
    search_timeout: int = 30  # seconds


class ProductSearchState(TypedDict):
    """Optimized state for the product search graph"""
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    search_type: Literal["text", "image", "hybrid"]
    search_results: List[Dict[str, Any]]
    current_query: str
    refinement_count: int
    config: SearchConfig
    user_feedback: Optional[str]
    selected_product: Optional[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]
    search_metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]


class LangGraphProductSearcher:
    """
    Highly optimized LangGraph-powered product search system

    Key optimizations:
    - Async operations for better performance
    - Intelligent caching system
    - Parallel search capabilities
    - Enhanced query processing
    - Performance monitoring
    - Better error handling and recovery
    """

    def __init__(self,
                 db_path: str,
                 collection_name: str,
                 model_name: str,
                 config: Optional[SearchConfig] = None):

        self.config = config or SearchConfig()

        # Initialize the vector database with connection pooling
        self.db = ProductSeekerVectorDB(
            db_path=db_path,
            collection_name=collection_name,
            model_name=model_name
        )

        # Cache for search results
        self._search_cache = {}
        self._cache_timestamps = {}

        # Performance tracking
        self._performance_stats = {
            "total_searches": 0,
            "average_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        # Thread pool for parallel operations
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Initialize the optimized graph
        self.graph = self._build_optimized_graph()

        logger.info("🚀 Optimized LangGraph ProductSearcher initialized")

    def _build_optimized_graph(self) -> StateGraph:
        """Build the optimized LangGraph workflow"""

        workflow = StateGraph(ProductSearchState)

        # Add optimized nodes
        workflow.add_node("parse_query", self._parse_query)
        workflow.add_node("check_cache", self._check_cache)
        workflow.add_node("execute_search", self._execute_search)
        workflow.add_node("post_process", self._post_process_results)
        workflow.add_node("evaluate_quality", self._evaluate_quality)
        workflow.add_node("smart_refine", self._smart_refine)
        workflow.add_node("format_response", self._format_response)
        workflow.add_node("update_cache", self._update_cache)

        # Set entry point
        workflow.set_entry_point("parse_query")

        # Optimized flow with conditional routing
        workflow.add_edge("parse_query", "check_cache")

        workflow.add_conditional_edges(
            "check_cache",
            self._cache_decision,
            {
                "hit": "format_response",
                "miss": "execute_search"
            }
        )

        workflow.add_edge("execute_search", "post_process")
        workflow.add_edge("post_process", "evaluate_quality")

        workflow.add_conditional_edges(
            "evaluate_quality",
            self._quality_decision,
            {
                "good": "format_response",
                "refine": "smart_refine",
                "failed": "format_response"
            }
        )

        workflow.add_edge("smart_refine", "execute_search")
        workflow.add_edge("format_response", "update_cache")
        workflow.add_edge("update_cache", END)

        return workflow.compile()

    def _parse_query(self, state: ProductSearchState) -> ProductSearchState:
        """Enhanced query parsing with NLP techniques"""
        start_time = time.time()

        last_message = state["messages"][-1] if state["messages"] else None
        query = last_message.content if isinstance(last_message, HumanMessage) else state.get("query", "")

        # Enhanced query processing
        processed_query = self._preprocess_query(query)

        # Intelligent search type detection
        search_type = self._detect_search_type(query)

        state.update({
            "current_query": processed_query,
            "search_type": search_type,
            "refinement_count": 0,
            "search_metadata": {"original_query": query},
            "performance_metrics": {"parse_time": time.time() - start_time}
        })

        return state

    def _preprocess_query(self, query: str) -> str:
        """Advanced query preprocessing"""
        if not query:
            return ""

        # Remove common stop words that don't help in product search
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}

        # Basic cleaning
        query = query.strip().lower()

        # Split and filter
        words = [word for word in query.split() if word not in stop_words and len(word) > 1]

        # Rejoin
        return " ".join(words)

    def _detect_search_type(self, query: str) -> Literal["text", "image", "hybrid"]:
        """Intelligent search type detection"""
        if not query:
            return "text"

        # Check for image file
        if Path(query).exists() and Path(query).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            return "image"

        # Check for hybrid indicators (could be enhanced with ML)
        hybrid_keywords = ["like this", "similar to", "looks like", "design like"]
        if any(keyword in query.lower() for keyword in hybrid_keywords):
            return "hybrid"

        return "text"

    def _check_cache(self, state: ProductSearchState) -> ProductSearchState:
        """Check if we have cached results"""
        if not self.config.enable_caching:
            state["cache_status"] = "disabled"
            return state

        cache_key = self._generate_cache_key(state["current_query"], state["search_type"])

        if cache_key in self._search_cache:
            cache_time = self._cache_timestamps.get(cache_key, 0)
            if time.time() - cache_time < self.config.cache_ttl:
                state["search_results"] = self._search_cache[cache_key]
                state["cache_status"] = "hit"
                self._performance_stats["cache_hits"] += 1
                return state

        state["cache_status"] = "miss"
        self._performance_stats["cache_misses"] += 1
        return state

    def _cache_decision(self, state: ProductSearchState) -> str:
        """Decide whether to use cache or execute search"""
        return state.get("cache_status", "miss")

    def _execute_search(self, state: ProductSearchState) -> ProductSearchState:
        """Execute optimized search with parallel capabilities"""
        start_time = time.time()

        query = state["current_query"]
        search_type = state["search_type"]

        try:
            if self.config.enable_parallel_search and search_type == "hybrid":
                # Parallel text and image search for hybrid
                results = self._parallel_hybrid_search(query)
            else:
                # Single search
                results = self._single_search(query, search_type)

            state["search_results"] = results
            state["performance_metrics"]["search_time"] = time.time() - start_time

        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            state["search_results"] = []
            state["search_error"] = str(e)

        return state

    def _single_search(self, query: str, search_type: str) -> List[Dict[str, Any]]:
        """Execute single search operation"""
        try:
            if search_type == "image":
                results = self.db.search_by_image(query, n_results=self.config.max_results)
            else:
                results = self.db.search_by_text(query, n_results=self.config.max_results)

            if results.get('error'):
                logger.error(f"Database search failed: {results['error']}")
                return []

            return results.get('results', [])

        except Exception as e:
            logger.error(f"Single search failed: {e}")
            return []

    def _parallel_hybrid_search(self, query: str) -> List[Dict[str, Any]]:
        """Execute parallel searches for hybrid mode"""
        try:
            # Submit both searches in parallel
            text_future = self._executor.submit(self._single_search, query, "text")
            # For hybrid, we might search for similar products
            image_future = self._executor.submit(self._single_search, query, "text")  # Placeholder

            # Get results with timeout
            text_results = text_future.result(timeout=self.config.search_timeout)
            image_results = image_future.result(timeout=self.config.search_timeout)

            # Merge and deduplicate results
            return self._merge_results(text_results, image_results)

        except Exception as e:
            logger.error(f"Parallel search failed: {e}")
            return self._single_search(query, "text")  # Fallback

    def _merge_results(self, text_results: List[Dict], image_results: List[Dict]) -> List[Dict]:
        """Merge and deduplicate search results"""
        seen_ids = set()
        merged = []

        # Prioritize text results
        for result in text_results:
            result_id = result.get('id') or result.get('metadata', {}).get('id')
            if result_id and result_id not in seen_ids:
                seen_ids.add(result_id)
                merged.append(result)

        # Add unique image results
        for result in image_results:
            result_id = result.get('id') or result.get('metadata', {}).get('id')
            if result_id and result_id not in seen_ids:
                seen_ids.add(result_id)
                merged.append(result)

        return merged[:self.config.max_results]

    def _post_process_results(self, state: ProductSearchState) -> ProductSearchState:
        """Post-process search results for better quality"""
        results = state["search_results"]

        if not results:
            return state

        # Filter by similarity threshold
        filtered_results = [
            result for result in results
            if result.get('similarity', 0) >= self.config.min_similarity_threshold
        ]

        # Sort by similarity score
        filtered_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        # Enhance metadata
        for result in filtered_results:
            self._enhance_result_metadata(result)

        state["search_results"] = filtered_results
        return state

    def _enhance_result_metadata(self, result: Dict[str, Any]) -> None:
        """Enhance individual result with additional metadata"""
        metadata = result.get('metadata', {})

        # Add computed fields
        if 'price' in metadata:
            try:
                price_str = metadata['price'].replace('$', '').replace(',', '')
                metadata['price_numeric'] = float(price_str)
            except (ValueError, AttributeError):
                pass

        # Add relevance score (combination of similarity and other factors)
        similarity = result.get('similarity', 0)
        metadata['relevance_score'] = similarity  # Can be enhanced with more factors

    def _evaluate_quality(self, state: ProductSearchState) -> ProductSearchState:
        """Evaluate search result quality"""
        results = state["search_results"]

        if not results:
            state["quality_score"] = 0.0
            state["quality_status"] = "no_results"
        elif len(results) < 3:
            state["quality_score"] = 0.3
            state["quality_status"] = "few_results"
        else:
            # Calculate average similarity
            avg_similarity = sum(r.get('similarity', 0) for r in results) / len(results)
            state["quality_score"] = avg_similarity

            if avg_similarity > 0.7:
                state["quality_status"] = "good"
            elif avg_similarity > 0.5:
                state["quality_status"] = "acceptable"
            else:
                state["quality_status"] = "poor"

        return state

    def _quality_decision(self, state: ProductSearchState) -> str:
        """Decide whether to refine search based on quality"""
        quality_status = state.get("quality_status", "good")
        refinement_count = state.get("refinement_count", 0)

        if (quality_status in ["no_results", "few_results", "poor"] and
                refinement_count < self.config.max_refinements):
            return "refine"
        elif quality_status in ["good", "acceptable"]:
            return "good"
        else:
            return "failed"

    def _smart_refine(self, state: ProductSearchState) -> ProductSearchState:
        """Intelligent query refinement"""
        current_query = state["current_query"]
        refinement_count = state.get("refinement_count", 0)
        quality_status = state.get("quality_status", "")

        refined_query = self._apply_refinement_strategy(current_query, refinement_count, quality_status)

        state.update({
            "current_query": refined_query,
            "refinement_count": refinement_count + 1
        })

        logger.info(f"🔄 Smart refinement #{refinement_count + 1}: '{current_query}' → '{refined_query}'")
        return state

    def _apply_refinement_strategy(self, query: str, refinement_count: int, quality_status: str) -> str:
        """Apply intelligent refinement strategies"""
        words = query.split()

        if refinement_count == 0:
            if quality_status == "no_results":
                # Make query broader
                return words[0] if words else query
            else:
                # Try removing last word
                return " ".join(words[:-1]) if len(words) > 1 else query

        elif refinement_count == 1:
            # Try category-based search
            category_terms = ["electronics", "computer", "phone", "headphone", "laptop"]
            for term in category_terms:
                if term in query.lower():
                    return term
            return words[0] if words else query

        return query  # No more refinements

    def _format_response(self, state: ProductSearchState) -> ProductSearchState:
        """Format the final response with rich information"""
        results = state["search_results"]
        query = state.get("current_query", "")
        quality_score = state.get("quality_score", 0.0)

        if not results:
            response = self._format_no_results_response(query)
        else:
            response = self._format_success_response(results, query, quality_score)

        # Add performance info in debug mode
        if logger.isEnabledFor(logging.DEBUG):
            metrics = state.get("performance_metrics", {})
            response += f"\n\n🔍 Performance: {metrics}"

        ai_message = AIMessage(content=response)
        state["messages"].append(ai_message)

        return state

    def _format_no_results_response(self, query: str) -> str:
        """Format response when no results found"""
        return f"""😔 No products found for '{query}'.

💡 **Suggestions:**
• Try broader search terms
• Check spelling
• Use different keywords
• Upload a product image for visual search

🔄 Would you like to try a different search?"""

    def _format_success_response(self, results: List[Dict], query: str, quality_score: float) -> str:
        """Format successful search response"""
        response = f"🎯 Found {len(results)} products for '{query}' (Quality: {quality_score:.1f}/1.0)\n\n"

        for i, result in enumerate(results[:5], 1):
            metadata = result.get('metadata', {})
            similarity = result.get('similarity', 0)

            response += f"**{i}. {metadata.get('title', 'Unknown Product')}**\n"
            response += f"   💰 Price: {metadata.get('price', 'N/A')}\n"
            response += f"   📂 Category: {metadata.get('category', 'N/A')}\n"
            response += f"   🏷️ Brand: {metadata.get('brand', 'N/A')}\n"
            response += f"   📊 Match: {similarity:.2f}\n"

            if metadata.get('description'):
                desc = metadata['description'][:80] + "..." if len(metadata['description']) > 80 else metadata[
                    'description']
                response += f"   📝 {desc}\n"

            response += "\n"

        response += "💡 **Next steps:**\n"
        response += "• Type a number (1-5) for product details\n"
        response += "• Try a new search\n"
        response += "• Upload an image for visual search"

        return response

    def _update_cache(self, state: ProductSearchState) -> ProductSearchState:
        """Update search cache"""
        if not self.config.enable_caching:
            return state

        cache_key = self._generate_cache_key(state["current_query"], state["search_type"])
        self._search_cache[cache_key] = state["search_results"]
        self._cache_timestamps[cache_key] = time.time()

        # Cleanup old cache entries
        self._cleanup_cache()

        return state

    def _generate_cache_key(self, query: str, search_type: str) -> str:
        """Generate cache key for search"""
        return f"{search_type}:{query.lower().strip()}"

    def _cleanup_cache(self) -> None:
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._cache_timestamps.items()
            if current_time - timestamp > self.config.cache_ttl
        ]

        for key in expired_keys:
            self._search_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)

    @lru_cache(maxsize=100)
    def _cached_db_stats(self) -> Dict[str, Any]:
        """Cached database statistics"""
        return self.db.get_database_stats()

    def search(self,
               query: str,
               search_type: str = "auto",
               config: Optional[SearchConfig] = None) -> Dict[str, Any]:
        """
        Optimized main search interface

        Args:
            query: Search query (text or image path)
            search_type: "text", "image", "hybrid", or "auto"
            config: Optional custom configuration

        Returns:
            Enhanced search results and metadata
        """
        start_time = time.time()
        self._performance_stats["total_searches"] += 1

        # Use custom config if provided
        if config:
            original_config = self.config
            self.config = config

        try:
            # Determine search type if auto
            if search_type == "auto":
                search_type = self._detect_search_type(query)

            # Initial state
            initial_state = ProductSearchState(
                messages=[HumanMessage(content=query)],
                query=query,
                search_type=search_type,
                search_results=[],
                current_query=query,
                refinement_count=0,
                config=self.config,
                user_feedback=None,
                selected_product=None,
                conversation_history=[],
                search_metadata={},
                performance_metrics={}
            )

            # Run the optimized graph
            final_state = self.graph.invoke(initial_state)

            # Calculate total time
            total_time = time.time() - start_time
            self._update_performance_stats(total_time)

            return {
                "success": True,
                "results": final_state.get("search_results", []),
                "messages": final_state.get("messages", []),
                "metadata": {
                    "refinement_count": final_state.get("refinement_count", 0),
                    "search_type": search_type,
                    "original_query": query,
                    "final_query": final_state.get("current_query", query),
                    "quality_score": final_state.get("quality_score", 0.0),
                    "total_time": total_time,
                    "cache_status": final_state.get("cache_status", "unknown")
                },
                "performance_stats": self._performance_stats.copy()
            }

        except Exception as e:
            logger.error(f"Optimized search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "messages": [AIMessage(content=f"Search failed: {str(e)}")],
                "metadata": {"total_time": time.time() - start_time}
            }

        finally:
            # Restore original config if custom was used
            if config:
                self.config = original_config

    def _update_performance_stats(self, response_time: float) -> None:
        """Update performance statistics"""
        total_searches = self._performance_stats["total_searches"]
        current_avg = self._performance_stats["average_response_time"]

        # Calculate new average
        new_avg = ((current_avg * (total_searches - 1)) + response_time) / total_searches
        self._performance_stats["average_response_time"] = new_avg

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        stats = self._performance_stats.copy()
        stats.update({
            "cache_size": len(self._search_cache),
            "cache_hit_rate": (stats["cache_hits"] / max(stats["cache_hits"] + stats["cache_misses"], 1)) * 100
        })
        return stats

    def clear_cache(self) -> None:
        """Clear search cache"""
        self._search_cache.clear()
        self._cache_timestamps.clear()
        logger.info("🧹 Search cache cleared")

    def get_database_stats(self) -> Dict[str, Any]:
        """Get cached database statistics"""
        return self._cached_db_stats()

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)


# Example usage with configuration
def main_example():
    """Example of using the optimized searcher"""

    # Custom configuration for high-performance searching
    config = SearchConfig(
        max_results=15,
        min_similarity_threshold=0.6,
        max_refinements=2,
        enable_caching=True,
        cache_ttl=600,  # 10 minutes
        enable_parallel_search=True,
        search_timeout=20
    )

    # Initialize optimized searcher
    searcher = LangGraphProductSearcher(
        db_path="D:/Vector/ProductSeeker_data",
        collection_name="ecommerce_test",
        config=config,
        model_name="clip-ViT-B-32"
    )

    # Test searches
    test_queries = [
        "gaming laptop high performance",
        "wireless bluetooth headphones",
        "smartphone android 5g",
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"🔍 Optimized Search: {query}")
        print('=' * 60)

        result = searcher.search(query)

        if result["success"]:
            metadata = result["metadata"]
            print(f"✅ Success! Found {len(result['results'])} results")
            print(f"⏱️  Total time: {metadata['total_time']:.2f}s")
            print(f"🎯 Quality score: {metadata['quality_score']:.2f}")
            print(f"🔄 Refinements: {metadata['refinement_count']}")
            print(f"💾 Cache: {metadata['cache_status']}")

            # Show performance stats
            stats = result["performance_stats"]
            print(f"📊 Avg response time: {stats['average_response_time']:.2f}s")
            print(f"📊 Cache hit rate: {searcher.get_performance_stats()['cache_hit_rate']:.1f}%")
        else:
            print(f"❌ Search failed: {result['error']}")

    # Show final performance statistics
    print(f"\n{'=' * 60}")
    print("📊 Final Performance Statistics")
    print('=' * 60)
    final_stats = searcher.get_performance_stats()
    for key, value in final_stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main_example()

