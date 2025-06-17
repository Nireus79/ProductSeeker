import logging
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from pathlib import Path
import json
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

# Import your existing classes
from Vector import ProductSeekerVectorDB
from Integrater import IntegratedProductScraper

logger = logging.getLogger(__name__)


class ProductSearchState(TypedDict):
    """State for the product search graph"""
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    search_type: str  # "text", "image", or "hybrid"
    search_results: List[Dict[str, Any]]
    current_query: str
    refinement_count: int
    max_refinements: int
    user_feedback: Optional[str]
    selected_product: Optional[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]


class LangGraphProductSearcher:
    """
    LangGraph-powered product search system that integrates with your vector database
    """

    def __init__(self,
                 db_path: str,
                 collection_name: str = "ecommerce_test",
                 model_name: str = "clip-ViT-B-32"):

        # Initialize the vector database
        self.db = ProductSeekerVectorDB(
            db_path=db_path,
            collection_name=collection_name,
            model_name=model_name
        )

        # Initialize the graph
        self.graph = self._build_graph()

        logger.info("🤖 LangGraph ProductSearcher initialized")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""

        # Define the graph
        workflow = StateGraph(ProductSearchState)

        # Add nodes
        workflow.add_node("understand_query", self._understand_query)
        workflow.add_node("search_products", self._search_products)
        workflow.add_node("evaluate_results", self._evaluate_results)
        workflow.add_node("refine_search", self._refine_search)
        workflow.add_node("present_results", self._present_results)
        workflow.add_node("handle_feedback", self._handle_feedback)

        # Define the flow
        workflow.set_entry_point("understand_query")

        # Add edges
        workflow.add_edge("understand_query", "search_products")
        workflow.add_edge("search_products", "evaluate_results")

        # Conditional edges from evaluate_results
        workflow.add_conditional_edges(
            "evaluate_results",
            self._should_refine,
            {
                "refine": "refine_search",
                "present": "present_results"
            }
        )

        workflow.add_edge("refine_search", "search_products")
        workflow.add_edge("present_results", "handle_feedback")

        # Conditional edges from handle_feedback
        workflow.add_conditional_edges(
            "handle_feedback",
            self._handle_feedback_decision,
            {
                "continue": "understand_query",
                "end": END
            }
        )

        return workflow.compile()

    def _understand_query(self, state: ProductSearchState) -> ProductSearchState:
        """Analyze and understand the user's query"""

        last_message = state["messages"][-1] if state["messages"] else None

        if isinstance(last_message, HumanMessage):
            query = last_message.content
        else:
            query = state.get("query", "")

        # Determine search type
        search_type = "text"  # Default

        # Check if it's an image path
        if query and Path(query).exists() and Path(query).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            search_type = "image"

        # Enhanced query processing could go here
        processed_query = query.strip()

        state.update({
            "current_query": processed_query,
            "search_type": search_type,
            "refinement_count": state.get("refinement_count", 0),
            "max_refinements": state.get("max_refinements", 3)
        })

        logger.info(f"🧠 Query understood: '{processed_query}' (type: {search_type})")

        return state

    def _search_products(self, state: ProductSearchState) -> ProductSearchState:
        """Search for products in the vector database"""

        query = state["current_query"]
        search_type = state["search_type"]

        try:
            if search_type == "image":
                results = self.db.search_by_image(query, n_results=10)
            else:
                results = self.db.search_by_text(query, n_results=10)

            if results.get('error'):
                logger.error(f"Search failed: {results['error']}")
                search_results = []
            else:
                search_results = results.get('results', [])

            state["search_results"] = search_results

            logger.info(f"🔍 Found {len(search_results)} products for query: '{query}'")

        except Exception as e:
            logger.error(f"Search error: {e}")
            state["search_results"] = []

        return state

    def _evaluate_results(self, state: ProductSearchState) -> ProductSearchState:
        """Evaluate the quality of search results"""

        results = state["search_results"]

        # Simple evaluation criteria
        if not results:
            state["evaluation"] = "no_results"
        elif len(results) < 3:
            state["evaluation"] = "few_results"
        elif all(result.get('similarity', 0) < 0.7 for result in results):
            state["evaluation"] = "low_confidence"
        else:
            state["evaluation"] = "good_results"

        return state

    def _should_refine(self, state: ProductSearchState) -> str:
        """Decide whether to refine the search or present results"""

        evaluation = state.get("evaluation", "good_results")
        refinement_count = state.get("refinement_count", 0)
        max_refinements = state.get("max_refinements", 3)

        if (evaluation in ["no_results", "few_results", "low_confidence"] and
                refinement_count < max_refinements):
            return "refine"
        else:
            return "present"

    def _refine_search(self, state: ProductSearchState) -> ProductSearchState:
        """Refine the search query"""

        current_query = state["current_query"]
        refinement_count = state.get("refinement_count", 0)

        # Simple refinement strategies
        if refinement_count == 0:
            # Try broader terms
            refined_query = " ".join(current_query.split()[:2])  # Use first 2 words
        elif refinement_count == 1:
            # Try more specific terms
            refined_query = current_query + " popular"
        else:
            # Try category-based search
            refined_query = current_query.split()[0] if current_query.split() else current_query

        state.update({
            "current_query": refined_query,
            "refinement_count": refinement_count + 1
        })

        logger.info(f"🔄 Refining search: '{current_query}' → '{refined_query}'")

        return state

    def _present_results(self, state: ProductSearchState) -> ProductSearchState:
        """Present search results to the user"""

        results = state["search_results"]
        query = state["current_query"]

        if not results:
            response = f"😔 Sorry, I couldn't find any products matching '{query}'. Try a different search term or upload an image!"
        else:
            response = f"🎯 Found {len(results)} products for '{query}':\n\n"

            for i, result in enumerate(results[:5], 1):  # Show top 5
                metadata = result.get('metadata', {})
                similarity = result.get('similarity', 0)

                response += f"{i}. **{metadata.get('title', 'Unknown Product')}**\n"
                response += f"   💰 Price: {metadata.get('price', 'N/A')}\n"
                response += f"   📂 Category: {metadata.get('category', 'N/A')}\n"
                response += f"   🏷️ Brand: {metadata.get('brand', 'N/A')}\n"
                response += f"   📊 Match: {similarity:.2f}\n"

                if metadata.get('description'):
                    desc = metadata['description'][:100] + "..." if len(metadata['description']) > 100 else metadata[
                        'description']
                    response += f"   📝 {desc}\n"

                response += "\n"

            response += "💡 Type a number to get more details, or try a new search!"

        # Add AI message to conversation
        ai_message = AIMessage(content=response)
        state["messages"].append(ai_message)

        return state

    def _handle_feedback(self, state: ProductSearchState) -> ProductSearchState:
        """Handle user feedback and selection"""

        # This would be implemented based on user input
        # For now, just mark as handled
        state["user_feedback"] = "handled"

        return state

    def _handle_feedback_decision(self, state: ProductSearchState) -> str:
        """Decide whether to continue or end based on feedback"""

        # Simple logic - in a real implementation, this would be more sophisticated
        return "end"  # For now, always end after presenting results

    def search(self, query: str, search_type: str = "auto") -> Dict[str, Any]:
        """
        Main search interface

        Args:
            query: Search query (text or image path)
            search_type: "text", "image", or "auto"

        Returns:
            Search results and conversation
        """

        # Determine search type if auto
        if search_type == "auto":
            if Path(query).exists() and Path(query).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                search_type = "image"
            else:
                search_type = "text"

        # Initial state
        initial_state = ProductSearchState(
            messages=[HumanMessage(content=query)],
            query=query,
            search_type=search_type,
            search_results=[],
            current_query=query,
            refinement_count=0,
            max_refinements=3,
            user_feedback=None,
            selected_product=None,
            conversation_history=[]
        )

        # Run the graph
        try:
            final_state = self.graph.invoke(initial_state)

            return {
                "success": True,
                "results": final_state.get("search_results", []),
                "messages": final_state.get("messages", []),
                "refinement_count": final_state.get("refinement_count", 0),
                "search_type": search_type,
                "original_query": query,
                "final_query": final_state.get("current_query", query)
            }

        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "messages": [AIMessage(content=f"Sorry, something went wrong: {str(e)}")]
            }

    def get_product_details(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific product"""

        try:
            # This would query the database for specific product details
            # Implementation depends on your database structure
            return self.db.get_product_by_id(product_id) if hasattr(self.db, 'get_product_by_id') else None
        except Exception as e:
            logger.error(f"Failed to get product details: {e}")
            return None

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return self.db.get_database_stats()


# Example usage function
def main_example():
    """Example of how to use the LangGraph ProductSearcher"""

    # Initialize the searcher
    searcher = LangGraphProductSearcher(
        db_path="D:/Vector/ProductSeeker_data",
        collection_name="ecommerce_test"
    )

    # Example searches
    test_queries = [
        "gaming laptop",
        "smartphone",
        "bluetooth headphones",
        # "path/to/product/image.jpg"  # Image search example
    ]

    for query in test_queries:
        print(f"\n{'=' * 50}")
        print(f"🔍 Searching for: {query}")
        print('=' * 50)

        result = searcher.search(query)

        if result["success"]:
            print(f"✅ Search completed successfully!")
            print(f"📊 Found {len(result['results'])} results")
            print(f"🔄 Refinements made: {result['refinement_count']}")

            # Print the AI response
            for message in result["messages"]:
                if isinstance(message, AIMessage):
                    print(f"\n🤖 Assistant: {message.content}")
        else:
            print(f"❌ Search failed: {result['error']}")


if __name__ == "__main__":
    main_example()
