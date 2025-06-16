import requests
from bs4 import BeautifulSoup
import json
import time
import os
from urllib.parse import urljoin, urlparse
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EcommerceScraper:
    """
    Scraper for webscraper.io test e-commerce site
    """

    def __init__(self, base_url, output_dir):
        # def __init__(self, base_url: str = "https://webscraper.io/test-sites/e-commerce/allinone",
        #                  output_dir: str = "scraped_data"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.session = requests.Session()

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        # Set headers to avoid blocking
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

        logger.info(f"EcommerceScraper initialized with base URL: {self.base_url}")

    def get_page(self, url: str, retries: int = 3) -> Optional[BeautifulSoup]:
        """Get and parse a web page"""
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')
                return soup

            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch {url} after {retries} attempts")
                    return None

    def download_image(self, image_url: str, product_id: str) -> Optional[str]:
        """Download product image and return local path"""
        try:
            # Create filename from product ID and image URL
            image_ext = Path(urlparse(image_url).path).suffix or '.jpg'
            filename = f"product_{product_id}{image_ext}"
            filepath = self.images_dir / filename

            # Skip if already exists
            if filepath.exists():
                return str(filepath)

            # Download image
            response = self.session.get(image_url, timeout=10)
            response.raise_for_status()

            # Save image
            with open(filepath, 'wb') as f:
                f.write(response.content)

            logger.debug(f"Downloaded image: {filename}")
            return str(filepath)

        except Exception as e:
            logger.warning(f"Failed to download image {image_url}: {e}")
            return None

    def extract_price(self, price_text: str) -> str:
        """Extract and clean price from text"""
        if not price_text:
            return "0.00"

        # Remove extra whitespace and extract price
        price_text = price_text.strip()

        # Find price pattern (e.g., $19.99, €25.50)
        price_match = re.search(r'[\$€£¥]?(\d+(?:\.\d{2})?)', price_text)
        if price_match:
            return price_match.group(0)

        return price_text

    def scrape_product_details(self, product_url: str, product_id: str) -> Dict[str, Any]:
        """Scrape detailed product information"""
        soup = self.get_page(product_url)
        if not soup:
            return {}

        details = {}

        try:
            # Get product title
            title_elem = soup.find('h1', class_='title') or soup.find('h1') or soup.find('title')
            if title_elem:
                details['title'] = title_elem.get_text(strip=True)

            # Get price
            price_elem = soup.find('h4', class_='price') or soup.find('span', class_='price') or soup.find(
                string=re.compile(r'\$\d+'))
            if price_elem:
                price_text = price_elem.get_text(strip=True) if hasattr(price_elem, 'get_text') else str(price_elem)
                details['price'] = self.extract_price(price_text)

            # Get description
            desc_elem = soup.find('div', class_='description') or soup.find('p', class_='description')
            if desc_elem:
                details['description'] = desc_elem.get_text(strip=True)

            # Get main product image
            img_elem = soup.find('img', class_='img-responsive') or soup.find('img')
            if img_elem and img_elem.get('src'):
                image_url = urljoin(product_url, img_elem['src'])
                image_path = self.download_image(image_url, product_id)
                if image_path:
                    details['image_path'] = image_path

        except Exception as e:
            logger.warning(f"Error extracting product details from {product_url}: {e}")

        return details

    def scrape_category_page(self, category_url: str, category_name: str) -> List[Dict[str, Any]]:
        """Scrape all products from a category page"""
        logger.info(f"Scraping category: {category_name} from {category_url}")

        soup = self.get_page(category_url)
        if not soup:
            return []

        products = []

        # Find all product thumbnails/links
        product_links = soup.find_all('a', class_='title') or soup.find_all('a', href=re.compile(r'/product/'))

        for i, link in enumerate(product_links, 1):
            try:
                # Get product URL
                product_url = urljoin(category_url, link['href'])
                product_id = f"{category_name.lower()}_{i}_{int(time.time())}"

                logger.info(f"Scraping product {i}/{len(product_links)}: {link.get_text(strip=True)}")

                # Basic product info from category page
                product = {
                    'id': product_id,
                    'url': product_url,
                    'category': category_name,
                    'scraped_at': datetime.now().isoformat()
                }

                # Get title from link if available
                if link.get_text(strip=True):
                    product['title'] = link.get_text(strip=True)

                # Try to get price from category page
                parent = link.find_parent()
                if parent:
                    price_elem = parent.find('h4', class_='price') or parent.find_next('h4')
                    if price_elem:
                        product['price'] = self.extract_price(price_elem.get_text(strip=True))

                # Scrape detailed product information
                detailed_info = self.scrape_product_details(product_url, product_id)
                product.update(detailed_info)

                # Set default values if missing
                product.setdefault('title', f'Product {i}')
                product.setdefault('price', '0.00')
                product.setdefault('description', '')
                product.setdefault('brand', 'Unknown')

                products.append(product)

                # Be respectful - add delay between requests
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error processing product {i}: {e}")
                continue

        logger.info(f"Scraped {len(products)} products from {category_name}")
        return products

    def get_all_categories(self) -> List[Dict[str, str]]:
        """Get all available categories from the main page"""
        logger.info("Discovering categories...")

        soup = self.get_page(self.base_url)
        if not soup:
            return []

        categories = []

        # Look for category links in different possible structures
        category_selectors = [
            'a[href*="/category/"]',
            '.nav a',
            '.category-link',
            'a:contains("Laptops")',
            'a:contains("Tablets")',
            'a:contains("Phones")'
        ]

        found_links = set()

        for selector in category_selectors:
            try:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    text = link.get_text(strip=True)

                    if href and text and href not in found_links:
                        category_url = urljoin(self.base_url, href)
                        categories.append({
                            'name': text,
                            'url': category_url
                        })
                        found_links.add(href)
            except:
                continue

        # Manual fallback for known categories if auto-discovery fails
        if not categories:
            manual_categories = [
                {'name': 'Laptops', 'url': f'{self.base_url}/laptops'},
                {'name': 'Tablets', 'url': f'{self.base_url}/tablets'},
                {'name': 'Phones', 'url': f'{self.base_url}/phones'}
            ]

            # Test which ones exist
            for cat in manual_categories:
                if self.get_page(cat['url']):
                    categories.append(cat)

        logger.info(f"Found {len(categories)} categories: {[c['name'] for c in categories]}")
        return categories

    def scrape_all_products(self, max_products_per_category: int = 50) -> List[Dict[str, Any]]:
        """Scrape all products from all categories"""
        logger.info("Starting full website scrape...")

        all_products = []
        categories = self.get_all_categories()

        if not categories:
            logger.warning("No categories found. Trying to scrape main page...")
            # Try to scrape from main page
            main_products = self.scrape_category_page(self.base_url, "Main")
            all_products.extend(main_products[:max_products_per_category])
        else:
            for category in categories:
                try:
                    products = self.scrape_category_page(category['url'], category['name'])
                    all_products.extend(products[:max_products_per_category])

                    # Add delay between categories
                    time.sleep(2)

                except Exception as e:
                    logger.error(f"Error scraping category {category['name']}: {e}")
                    continue

        logger.info(f"Scraping completed. Total products: {len(all_products)}")
        return all_products

    def save_products(self, products: List[Dict[str, Any]], filename: str = "products.json") -> str:
        """Save products to JSON file"""
        output_file = self.output_dir / filename

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(products, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(products)} products to {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Error saving products: {e}")
            return ""

    def run_full_scrape(self, max_products_per_category: int = 50) -> str:
        """Run complete scraping process"""
        logger.info("🚀 Starting ProductSeeker scrape...")

        try:
            # Scrape all products
            products = self.scrape_all_products(max_products_per_category)

            if not products:
                logger.error("No products were scraped!")
                return ""

            # Save to JSON
            output_file = self.save_products(products)

            # Print summary
            logger.info("\n" + "=" * 50)
            logger.info("📊 SCRAPING SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Total products scraped: {len(products)}")
            logger.info(f"Categories found: {len(set(p.get('category', 'Unknown') for p in products))}")
            logger.info(f"Products with images: {sum(1 for p in products if p.get('image_path'))}")
            logger.info(f"Output file: {output_file}")
            logger.info(f"Images directory: {self.images_dir}")
            logger.info("=" * 50)

            return output_file

        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return ""


# def main():
#     """Main function to run the scraper"""
#     # Initialize scraper
#     scraper = EcommerceScraper(
#         base_url="https://webscraper.io/test-sites/e-commerce/allinone",
#         output_dir="scraped_data"  # You can change this directory
#     )
#
#     # Run scraping
#     output_file = scraper.run_full_scrape(max_products_per_category=20)
#
#     if output_file:
#         logger.info(f"✅ Scraping completed successfully!")
#         logger.info(f"📁 Products saved to: {output_file}")
#         logger.info("\n🔥 Ready to use with ProductSeeker Vector DB!")
#
#         # Show example of how to use with ProductSeeker
#         print("\n" + "=" * 60)
#         print("💡 NEXT STEPS - Use with ProductSeeker:")
#         print("=" * 60)
#         print("from product_seeker import ProductSeekerVectorDB")
#         print("import json")
#         print()
#         print("# Load scraped products")
#         print(f"with open('{output_file}', 'r', encoding='utf-8') as f:")
#         print("    products = json.load(f)")
#         print()
#         print("# Initialize ProductSeeker with custom database path")
#         print("db = ProductSeekerVectorDB(")
#         print("    db_path='./my_product_database',  # Your custom path")
#         print("    collection_name='webscraper_products'")
#         print(")")
#         print()
#         print("# Add products to vector database")
#         print("stats = db.add_products(products)")
#         print("print(f'Added {stats[\"added\"]} products to database')")
#         print()
#         print("# Search products")
#         print("results = db.search_by_text('laptop gaming', n_results=5)")
#         print("for result in results['results']:")
#         print("    print(f\"- {result['metadata']['title']}\")")
#         print("=" * 60)
#     else:
#         logger.error("❌ Scraping failed!")
#
#
# if __name__ == "__main__":
#     main()
