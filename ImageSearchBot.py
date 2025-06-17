import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import streamlit as st
from PIL import Image
import io
import base64

from Vector import ProductSeekerVectorDB
from langgraph_db_interface import LangGraphProductSearcher

logger = logging.getLogger(__name__)


class ImageSearchBot:
    """
    Advanced image search bot with second chance and text assistance
    """

    def __init__(self, db_path: str, collection_name: str, model_name: str = "clip-ViT-B-32"):
        self.db = ProductSeekerVectorDB(
            db_path=db_path,
            collection_name=collection_name,
            model_name=model_name
        )

        # Initialize LangGraph system for advanced search
        self.langgraph_system = LangGraphProductSearcher(
            db_path=db_path,
            collection_name=collection_name,
            model_name=model_name
        )

        # Session state for conversation
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        if 'current_results' not in st.session_state:
            st.session_state.current_results = []
        if 'user_feedback' not in st.session_state:
            st.session_state.user_feedback = {}

        logger.info("🤖 Image Search Bot initialized")

    def search_by_image(self, image_path: str, n_results: int = 10) -> Dict:
        """Search products by image"""
        try:
            results = self.db.search_by_image(image_path, n_results=n_results)

            # Add search to history
            st.session_state.search_history.append({
                'type': 'image',
                'query': image_path,
                'results': results.get('results', []),
                'timestamp': st.session_state.get('timestamp', 'now')
            })

            return results
        except Exception as e:
            logger.error(f"Image search error: {e}")
            return {'error': str(e), 'results': [], 'count': 0}

    def search_by_text(self, query: str, n_results: int = 10) -> Dict:
        """Search products by text with LangGraph enhancement"""
        try:
            # Use LangGraph system for enhanced text search
            results = self.langgraph_system.search(query, search_type="text", max_results=n_results)

            # Add search to history
            st.session_state.search_history.append({
                'type': 'text',
                'query': query,
                'results': results.get('results', []),
                'timestamp': st.session_state.get('timestamp', 'now')
            })

            return results
        except Exception as e:
            logger.error(f"Text search error: {e}")
            return {'error': str(e), 'results': [], 'count': 0}

    def hybrid_search(self, image_path: str, text_query: str, n_results: int = 10) -> Dict:
        """Combine image and text search for better results"""
        try:
            # Get image results
            image_results = self.search_by_image(image_path, n_results=n_results // 2)

            # Get text results
            text_results = self.search_by_text(text_query, n_results=n_results // 2)

            # Combine and rank results
            combined_results = self._combine_search_results(
                image_results.get('results', []),
                text_results.get('results', [])
            )

            return {
                'results': combined_results[:n_results],
                'count': len(combined_results[:n_results]),
                'search_type': 'hybrid'
            }

        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            return {'error': str(e), 'results': [], 'count': 0}

    def _combine_search_results(self, image_results: List[Dict], text_results: List[Dict]) -> List[Dict]:
        """Combine and rank image and text search results"""
        combined = {}

        # Add image results with weight
        for result in image_results:
            product_id = result.get('id')
            if product_id:
                combined[product_id] = result.copy()
                combined[product_id]['image_similarity'] = result.get('similarity', 0)
                combined[product_id]['text_similarity'] = 0
                combined[product_id]['combined_score'] = result.get('similarity', 0) * 0.6

        # Add text results with weight
        for result in text_results:
            product_id = result.get('id')
            if product_id:
                if product_id in combined:
                    # Update existing result
                    combined[product_id]['text_similarity'] = result.get('similarity', 0)
                    combined[product_id]['combined_score'] = (
                            combined[product_id]['image_similarity'] * 0.6 +
                            result.get('similarity', 0) * 0.4
                    )
                else:
                    # Add new result
                    combined[product_id] = result.copy()
                    combined[product_id]['image_similarity'] = 0
                    combined[product_id]['text_similarity'] = result.get('similarity', 0)
                    combined[product_id]['combined_score'] = result.get('similarity', 0) * 0.4

        # Sort by combined score
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x.get('combined_score', 0),
            reverse=True
        )

        return sorted_results

    def get_product_suggestions(self, category: str, exclude_ids: List[str] = None) -> List[Dict]:
        """Get product suggestions based on category"""
        try:
            results = self.search_by_text(f"products in {category}", n_results=20)
            suggestions = results.get('results', [])

            # Filter out excluded products
            if exclude_ids:
                suggestions = [s for s in suggestions if s.get('id') not in exclude_ids]

            return suggestions[:5]  # Return top 5 suggestions

        except Exception as e:
            logger.error(f"Suggestion error: {e}")
            return []

    def record_user_feedback(self, product_id: str, feedback_type: str, rating: int = None):
        """Record user feedback for improving future searches"""
        if product_id not in st.session_state.user_feedback:
            st.session_state.user_feedback[product_id] = []

        st.session_state.user_feedback[product_id].append({
            'type': feedback_type,  # 'like', 'dislike', 'not_relevant', 'close_match'
            'rating': rating,
            'timestamp': st.session_state.get('timestamp', 'now')
        })

    def run_streamlit_app(self):
        """Run the Streamlit web application"""
        st.set_page_config(
            page_title="🔍 AI Product Search Bot",
            page_icon="🔍",
            layout="wide"
        )

        st.title("🔍 AI Product Search Bot")
        st.markdown("**Upload an image or describe what you're looking for!**")

        # Sidebar for settings
        with st.sidebar:
            st.header("⚙️ Settings")
            max_results = st.slider("Max Results", 3, 20, 10)
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

            st.header("📊 Database Stats")
            try:
                stats = self.db.get_database_stats()
                st.metric("Total Products", stats.get('total_products', 0))
                st.metric("Products with Images", stats.get('products_with_images', 0))
            except Exception as e:
                st.error(f"Failed to load stats: {e}")

        # Main search interface
        col1, col2 = st.columns([1, 1])

        with col1:
            st.header("🖼️ Image Search")
            uploaded_file = st.file_uploader(
                "Upload product image",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
                help="Upload an image of the product you're looking for"
            )

            if uploaded_file:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                # Save image temporarily
                temp_path = f"temp_image_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                if st.button("🔍 Search by Image", key="image_search"):
                    with st.spinner("Searching for similar products..."):
                        results = self.search_by_image(temp_path, n_results=max_results)
                        st.session_state.current_results = results.get('results', [])
                        st.session_state.search_type = 'image'
                        st.session_state.search_query = uploaded_file.name

                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        with col2:
            st.header("📝 Text Search")
            text_query = st.text_input(
                "Describe the product",
                placeholder="e.g., 'red gaming laptop with RGB keyboard'"
            )

            if st.button("🔍 Search by Text", key="text_search"):
                if text_query:
                    with st.spinner("Searching for products..."):
                        results = self.search_by_text(text_query, n_results=max_results)
                        st.session_state.current_results = results.get('results', [])
                        st.session_state.search_type = 'text'
                        st.session_state.search_query = text_query
                else:
                    st.warning("Please enter a search query")

        # Hybrid search option
        st.header("🔄 Hybrid Search")
        if uploaded_file and text_query:
            if st.button("🎯 Combine Image + Text Search", key="hybrid_search"):
                with st.spinner("Performing hybrid search..."):
                    temp_path = f"temp_image_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    results = self.hybrid_search(temp_path, text_query, n_results=max_results)
                    st.session_state.current_results = results.get('results', [])
                    st.session_state.search_type = 'hybrid'
                    st.session_state.search_query = f"Image: {uploaded_file.name} + Text: {text_query}"

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

        # Display results
        if st.session_state.current_results:
            st.header("🎯 Search Results")

            # Filter results by confidence
            filtered_results = [
                r for r in st.session_state.current_results
                if r.get('similarity', 0) >= confidence_threshold
            ]

            if not filtered_results:
                st.warning(f"No results meet the confidence threshold of {confidence_threshold:.2f}")
                st.info("Try lowering the confidence threshold or refining your search")

                # Second chance suggestions
                self._show_second_chance_options()
            else:
                # Display results in a grid
                self._display_results_grid(filtered_results)

                # Second chance if user is not satisfied
                if st.button("🔄 Not what you're looking for? Get suggestions"):
                    self._show_second_chance_options()

    def _display_results_grid(self, results: List[Dict]):
        """Display search results in a grid layout"""
        cols = st.columns(3)

        for idx, result in enumerate(results):
            col = cols[idx % 3]
            metadata = result.get('metadata', {})

            with col:
                st.subheader(f"#{idx + 1}")

                # Product image
                if metadata.get('image_path'):
                    try:
                        image_path = Path(metadata['image_path'])
                        if image_path.exists():
                            image = Image.open(image_path)
                            st.image(image, use_column_width=True)
                    except Exception:
                        st.info("📷 Image not available")
                else:
                    st.info("📷 No image available")

                # Product details
                st.write(f"**{metadata.get('title', 'Unknown Product')}**")
                st.write(f"💰 **Price:** {metadata.get('price', 'N/A')}")
                st.write(f"📂 **Category:** {metadata.get('category', 'N/A')}")
                st.write(f"🏷️ **Brand:** {metadata.get('brand', 'N/A')}")
                st.write(f"📊 **Match:** {result.get('similarity', 0):.3f}")

                if metadata.get('description'):
                    with st.expander("📝 Description"):
                        st.write(metadata['description'])

                # User feedback buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("👍", key=f"like_{idx}"):
                        self.record_user_feedback(result.get('id'), 'like')
                        st.success("Feedback recorded!")

                with col2:
                    if st.button("👎", key=f"dislike_{idx}"):
                        self.record_user_feedback(result.get('id'), 'dislike')
                        st.info("Feedback recorded!")

                with col3:
                    if st.button("❓", key=f"not_relevant_{idx}"):
                        self.record_user_feedback(result.get('id'), 'not_relevant')
                        st.info("Feedback recorded!")

                st.markdown("---")

    def _show_second_chance_options(self):
        """Show second chance options when user is not satisfied"""
        st.header("🔄 Second Chance - Let's Try Again!")

        # Option 1: Different search terms
        st.subheader("1. Try Different Keywords")
        new_query = st.text_input(
            "Try different search terms:",
            placeholder="e.g., 'laptop computer', 'mobile phone', 'wireless earbuds'"
        )

        if st.button("🔍 Search with New Terms", key="second_chance_text"):
            if new_query:
                with st.spinner("Searching with new terms..."):
                    results = self.search_by_text(new_query, n_results=10)
                    st.session_state.current_results = results.get('results', [])
                    st.session_state.search_type = 'text_refined'
                    st.session_state.search_query = new_query
                    st.rerun()

        # Option 2: Browse by category
        st.subheader("2. Browse by Category")

        # Get available categories from database
        try:
            categories = ["Electronics", "Computers", "Phones", "Tablets", "Accessories",
                          "Gaming"]  # Default categories

            selected_category = st.selectbox("Choose a category:", categories)

            if st.button("📂 Browse Category", key="browse_category"):
                with st.spinner(f"Loading {selected_category} products..."):
                    results = self.search_by_text(selected_category, n_results=15)
                    st.session_state.current_results = results.get('results', [])
                    st.session_state.search_type = 'category'
                    st.session_state.search_query = f"Category: {selected_category}"
                    st.rerun()
        except Exception as e:
            st.error(f"Failed to load categories: {e}")

        # Option 3: Describe what you're looking for
        st.subheader("3. Describe What You Need")

        col1, col2 = st.columns(2)
        with col1:
            purpose = st.selectbox(
                "What's it for?",
                ["", "Work", "Gaming", "Entertainment", "Study", "Travel", "Home", "Gift"]
            )

        with col2:
            price_range = st.selectbox(
                "Price range?",
                ["", "Under $100", "$100-500", "$500-1000", "Over $1000"]
            )

        additional_features = st.text_area(
            "Any specific features or requirements?",
            placeholder="e.g., 'needs to be portable', 'high battery life', 'good camera'"
        )

        if st.button("🎯 Find Based on Needs", key="needs_based_search"):
            # Construct search query from needs
            query_parts = []
            if purpose:
                query_parts.append(f"for {purpose.lower()}")
            if additional_features:
                query_parts.append(additional_features)

            needs_query = " ".join(query_parts) if query_parts else "popular products"

            with st.spinner("Finding products that match your needs..."):
                results = self.langgraph_system.search(
                    query=needs_query,
                    search_type="text",
                    max_results=12
                )

                # Filter by price range if specified
                if price_range and results.get('results'):
                    filtered_results = self._filter_by_price_range(results['results'], price_range)
                    results['results'] = filtered_results

                st.session_state.current_results = results.get('results', [])
                st.session_state.search_type = 'needs_based'
                st.session_state.search_query = f"Needs: {needs_query} (Price: {price_range})"
                st.rerun()

        # Option 4: Similar products to what was found
        if st.session_state.current_results:
            st.subheader("4. Find Similar Products")

            # Let user select a product they found interesting
            product_options = []
            for i, result in enumerate(st.session_state.current_results[:5]):
                metadata = result.get('metadata', {})
                title = metadata.get('title', f'Product {i + 1}')
                product_options.append(f"{i}: {title}")

            if product_options:
                selected_product_idx = st.selectbox(
                    "Which product is closest to what you want?",
                    range(len(product_options)),
                    format_func=lambda x: product_options[x]
                )

                if st.button("🔗 Find Similar Products", key="find_similar"):
                    selected_product = st.session_state.current_results[selected_product_idx]
                    metadata = selected_product.get('metadata', {})

                    # Create search query based on selected product
                    category = metadata.get('category', '')
                    brand = metadata.get('brand', '')

                    similar_query = f"{category} {brand}".strip()
                    if not similar_query:
                        similar_query = metadata.get('title', 'similar products')

                    with st.spinner("Finding similar products..."):
                        results = self.search_by_text(similar_query, n_results=15)

                        # Remove the originally selected product
                        filtered_results = [
                            r for r in results.get('results', [])
                            if r.get('id') != selected_product.get('id')
                        ]

                        st.session_state.current_results = filtered_results
                        st.session_state.search_type = 'similar'
                        st.session_state.search_query = f"Similar to: {metadata.get('title', 'selected product')}"
                        st.rerun()

    def _filter_by_price_range(self, results: List[Dict], price_range: str) -> List[Dict]:
        """Filter results by price range"""
        filtered = []

        for result in results:
            metadata = result.get('metadata', {})
            price_str = metadata.get('price', '0')

            try:
                # Extract numeric price (assuming format like "$123.45")
                price_clean = price_str.replace('$', '').replace(',', '').strip()
                price = float(price_clean) if price_clean else 0

                if self._price_matches_range(price, price_range):
                    filtered.append(result)
            except (ValueError, AttributeError):
                # If price parsing fails, include the product
                filtered.append(result)

        return filtered

    def _price_matches_range(self, price: float, price_range: str) -> bool:
        """Check if price matches the specified range"""
        if price_range == "Under $100":
            return price < 100
        elif price_range == "$100-500":
            return 100 <= price <= 500
        elif price_range == "$500-1000":
            return 500 <= price <= 1000
        elif price_range == "Over $1000":
            return price > 1000

        return True  # If no specific range, include all

    def run_console_interface(self):
        """Run console-based interface for testing"""
        print("🤖 AI Product Search Bot - Console Mode")
        print("=" * 50)

        while True:
            print("\nOptions:")
            print("1. Search by image")
            print("2. Search by text")
            print("3. Hybrid search")
            print("4. View search history")
            print("5. Database stats")
            print("6. Exit")

            choice = input("\nSelect option (1-6): ").strip()

            if choice == "1":
                image_path = input("Enter image path: ").strip()
                if os.path.exists(image_path):
                    results = self.search_by_image(image_path)
                    self._print_console_results(results, f"Image: {image_path}")
                else:
                    print("❌ Image file not found")

            elif choice == "2":
                query = input("Enter search query: ").strip()
                if query:
                    results = self.search_by_text(query)
                    self._print_console_results(results, f"Text: {query}")
                else:
                    print("❌ Please enter a search query")

            elif choice == "3":
                image_path = input("Enter image path: ").strip()
                text_query = input("Enter text description: ").strip()

                if os.path.exists(image_path) and text_query:
                    results = self.hybrid_search(image_path, text_query)
                    self._print_console_results(results, f"Hybrid: {image_path} + {text_query}")
                else:
                    print("❌ Please provide both image and text")

            elif choice == "4":
                self._print_search_history()

            elif choice == "5":
                stats = self.db.get_database_stats()
                print(f"\n📊 Database Statistics:")
                print(f"Total Products: {stats.get('total_products', 0)}")
                print(f"Products with Images: {stats.get('products_with_images', 0)}")

            elif choice == "6":
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid option")

    def _print_console_results(self, results: Dict, query: str):
        """Print search results in console"""
        print(f"\n🔍 Results for: {query}")
        print("-" * 50)

        if results.get('error'):
            print(f"❌ Error: {results['error']}")
            return

        search_results = results.get('results', [])
        if not search_results:
            print("No products found")
            return

        for i, result in enumerate(search_results, 1):
            metadata = result.get('metadata', {})
            print(f"{i}. {metadata.get('title', 'Unknown Product')}")
            print(f"   💰 Price: {metadata.get('price', 'N/A')}")
            print(f"   📂 Category: {metadata.get('category', 'N/A')}")
            print(f"   🏷️ Brand: {metadata.get('brand', 'N/A')}")
            print(f"   📊 Similarity: {result.get('similarity', 0):.3f}")
            print()

    def _print_search_history(self):
        """Print search history"""
        print("\n📜 Search History:")
        print("-" * 30)

        if not st.session_state.search_history:
            print("No searches performed yet")
            return

        for i, search in enumerate(st.session_state.search_history, 1):
            print(f"{i}. {search['type'].upper()}: {search['query']}")
            print(f"   Results: {len(search['results'])}")
            print()


def main():
    """Main function to run the bot"""
    import sys

    # Configuration
    DATABASE_PATH = "D:/Vector/ProductSeeker_data"
    COLLECTION_NAME = "ecommerce_test"
    MODEL_NAME = "clip-ViT-B-32"

    # Initialize bot
    bot = ImageSearchBot(
        db_path=DATABASE_PATH,
        collection_name=COLLECTION_NAME,
        model_name=MODEL_NAME
    )

    # Choose interface based on command line argument
    if len(sys.argv) > 1 and sys.argv[1] == "--console":
        bot.run_console_interface()
    else:
        # Run Streamlit app
        bot.run_streamlit_app()


if __name__ == "__main__":
    main()
