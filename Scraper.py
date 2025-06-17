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
    Enhanced scraper with better debugging and flexibility
    """

    def __init__(self, base_url, output_dir):
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

    def debug_page_structure(self, url: str) -> None:
        """Debug method to inspect page structure"""
        logger.info(f"🔍 DEBUGGING PAGE STRUCTURE: {url}")
        soup = self.get_page(url)
        if not soup:
            logger.error("Could not fetch page for debugging")
            return

        # Log basic page info
        title = soup.find('title')
        logger.info(f"Page title: {title.get_text() if title else 'No title found'}")

        # Look for common navigation elements
        nav_elements = soup.find_all(['nav', 'ul', 'div'], class_=re.compile(r'nav|menu|category|sidebar', re.I))
        logger.info(f"Found {len(nav_elements)} potential navigation elements")

        # Look for all links
        all_links = soup.find_all('a', href=True)
        logger.info(f"Found {len(all_links)} total links")

        # Show first 10 links for inspection
        logger.info("First 10 links:")
        for i, link in enumerate(all_links[:10]):
            href = link.get('href')
            text = link.get_text(strip=True)[:50]  # Limit text length
            logger.info(f"  {i + 1}. {text} -> {href}")

        # Look for product-like elements
        product_elements = soup.find_all(['div', 'article', 'li'], class_=re.compile(r'product|item|card', re.I))
        logger.info(f"Found {len(product_elements)} potential product elements")

        # Look for price elements
        price_elements = soup.find_all(string=re.compile(r'[\$€£¥]\d+|price', re.I))
        logger.info(f"Found {len(price_elements)} potential price elements")

    def find_product_links_generic(self, soup: BeautifulSoup, base_url: str = None) -> List[Dict[str, str]]:
        """Generic method to find product links using multiple strategies"""
        if base_url is None:
            base_url = self.base_url

        product_links = []

        # Strategy 1: Look for links with product-related keywords in href
        product_href_patterns = [
            r'/product/',
            r'/item/',
            r'/p/',
            r'product=',
            r'item=',
            r'/products/',
            r'/shop/',
            r'/buy/'
        ]

        for pattern in product_href_patterns:
            links = soup.find_all('a', href=re.compile(pattern, re.I))
            for link in links:
                href = link.get('href')
                text = link.get_text(strip=True)
                if href and text:
                    product_links.append({
                        'url': urljoin(base_url, href),
                        'title': text,
                        'source': f'href_pattern_{pattern}'
                    })

        # Strategy 2: Look for links within product containers
        product_containers = soup.find_all(['div', 'article', 'li'], class_=re.compile(r'product|item|card', re.I))
        for container in product_containers:
            links = container.find_all('a', href=True)
            for link in links:
                href = link.get('href')
                text = link.get_text(strip=True)
                if href and text:
                    product_links.append({
                        'url': urljoin(base_url, href),
                        'title': text,
                        'source': 'product_container'
                    })

        # Strategy 3: Look for links with product-related classes
        product_link_classes = [
            'product-link',
            'item-link',
            'product-title',
            'product-name',
            'title'
        ]

        for class_name in product_link_classes:
            links = soup.find_all('a', class_=re.compile(class_name, re.I))
            for link in links:
                href = link.get('href')
                text = link.get_text(strip=True)
                if href and text:
                    product_links.append({
                        'url': urljoin(base_url, href),
                        'title': text,
                        'source': f'class_{class_name}'
                    })

        # Remove duplicates based on URL
        seen_urls = set()
        unique_links = []
        for link in product_links:
            if link['url'] not in seen_urls:
                seen_urls.add(link['url'])
                unique_links.append(link)

        logger.info(f"Found {len(unique_links)} unique product links using generic strategies")
        return unique_links

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
            # Get product title - try multiple selectors
            title_selectors = [
                'h1.title',
                'h1',
                '.product-title',
                '.product-name',
                'title'
            ]

            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    details['title'] = title_elem.get_text(strip=True)
                    break

            # Get price - try multiple selectors
            price_selectors = [
                'h4.price',
                '.price',
                '.product-price',
                '[class*="price"]'
            ]

            for selector in price_selectors:
                price_elem = soup.select_one(selector)
                if price_elem:
                    details['price'] = self.extract_price(price_elem.get_text(strip=True))
                    break

            # If no price found, search for price in text
            if 'price' not in details:
                price_text = soup.find(string=re.compile(r'\$\d+|\€\d+|£\d+|¥\d+'))
                if price_text:
                    details['price'] = self.extract_price(str(price_text))

            # Get description
            desc_selectors = [
                '.description',
                '.product-description',
                '.product-details',
                'p'
            ]

            for selector in desc_selectors:
                desc_elem = soup.select_one(selector)
                if desc_elem:
                    desc_text = desc_elem.get_text(strip=True)
                    if len(desc_text) > 20:  # Only use if it's substantial
                        details['description'] = desc_text
                        break

            # Get main product image
            img_selectors = [
                'img.img-responsive',
                '.product-image img',
                '.main-image img',
                'img'
            ]

            for selector in img_selectors:
                img_elem = soup.select_one(selector)
                if img_elem and img_elem.get('src'):
                    image_url = urljoin(product_url, img_elem['src'])
                    image_path = self.download_image(image_url, product_id)
                    if image_path:
                        details['image_path'] = image_path
                        break

        except Exception as e:
            logger.warning(f"Error extracting product details from {product_url}: {e}")

        return details

    def scrape_page_products(self, page_url: str, page_name: str = "Unknown") -> List[Dict[str, Any]]:
        """Scrape all products from any page using generic strategies"""
        logger.info(f"Scraping products from: {page_name} ({page_url})")

        soup = self.get_page(page_url)
        if not soup:
            return []

        # Find product links using generic strategies
        product_links = self.find_product_links_generic(soup, page_url)

        if not product_links:
            logger.warning(f"No product links found on {page_name}")
            return []

        logger.info(f"Found {len(product_links)} product links on {page_name}")

        products = []
        for i, link_info in enumerate(product_links, 1):
            try:
                product_url = link_info['url']
                product_id = f"{page_name.lower().replace(' ', '_')}_{i}_{int(time.time())}"

                logger.info(f"Scraping product {i}/{len(product_links)}: {link_info['title'][:50]}...")

                # Basic product info
                product = {
                    'id': product_id,
                    'url': product_url,
                    'category': page_name,
                    'scraped_at': datetime.now().isoformat(),
                    'title': link_info['title'],
                    'discovery_method': link_info['source']
                }

                # Scrape detailed product information
                detailed_info = self.scrape_product_details(product_url, product_id)
                product.update(detailed_info)

                # Set default values if missing
                product.setdefault('price', '0.00')
                product.setdefault('description', '')
                product.setdefault('brand', 'Unknown')

                products.append(product)

                # Be respectful - add delay between requests
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error processing product {i}: {e}")
                continue

        logger.info(f"Successfully scraped {len(products)} products from {page_name}")
        return products

    def get_all_categories(self) -> List[Dict[str, str]]:
        """Get all available categories from the main page"""
        logger.info("Discovering categories...")

        soup = self.get_page(self.base_url)
        if not soup:
            return []

        categories = []

        # Enhanced category discovery
        category_patterns = [
            # Direct category links
            ('a[href*="/category/"]', 'category_direct'),
            ('a[href*="/categories/"]', 'categories_direct'),
            ('a[href*="/cat/"]', 'cat_direct'),

            # Navigation elements
            ('nav a', 'nav_links'),
            ('.nav a', 'nav_class'),
            ('.navbar a', 'navbar'),
            ('.navigation a', 'navigation'),

            # Menu elements
            ('.menu a', 'menu'),
            ('.main-menu a', 'main_menu'),
            ('.category-menu a', 'category_menu'),

            # Sidebar elements
            ('.sidebar a', 'sidebar'),
            ('.categories a', 'categories_sidebar'),

            # Product type links
            ('a:contains("Laptop")', 'contains_laptop'),
            ('a:contains("Tablet")', 'contains_tablet'),
            ('a:contains("Phone")', 'contains_phone'),
            ('a:contains("Computer")', 'contains_computer'),
            ('a:contains("Electronics")', 'contains_electronics'),
        ]

        found_links = set()

        for selector, method in category_patterns:
            try:
                if selector.startswith('a:contains'):
                    # Handle pseudo-selector manually
                    keyword = selector.split('"')[1]
                    links = soup.find_all('a', string=re.compile(keyword, re.I))
                else:
                    links = soup.select(selector)

                for link in links:
                    href = link.get('href')
                    text = link.get_text(strip=True)

                    if href and text and href not in found_links and len(text) > 0:
                        category_url = urljoin(self.base_url, href)
                        # Filter out unwanted links
                        if not any(skip in href.lower() for skip in ['javascript:', 'mailto:', '#', 'tel:']):
                            categories.append({
                                'name': text,
                                'url': category_url,
                                'discovery_method': method
                            })
                            found_links.add(href)
            except Exception as e:
                logger.debug(f"Error with selector {selector}: {e}")
                continue

        logger.info(f"Found {len(categories)} potential categories")
        for cat in categories[:10]:  # Show first 10
            logger.info(f"  - {cat['name']} ({cat['discovery_method']})")

        return categories

    def scrape_all_products(self, max_products_per_category: int = 50) -> List[Dict[str, Any]]:
        """Scrape all products from all categories"""
        logger.info("Starting comprehensive product scrape...")

        # First, debug the main page structure
        self.debug_page_structure(self.base_url)

        all_products = []
        categories = self.get_all_categories()

        if not categories:
            logger.warning("No categories found. Trying to scrape main page directly...")
            # Try to scrape from main page
            main_products = self.scrape_page_products(self.base_url, "Main_Page")
            all_products.extend(main_products[:max_products_per_category])
        else:
            for category in categories:
                try:
                    products = self.scrape_page_products(category['url'], category['name'])
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
        """Run complete scraping process with enhanced debugging"""
        logger.info("🚀 Starting Enhanced ProductSeeker scrape...")

        try:
            # Scrape all products
            products = self.scrape_all_products(max_products_per_category)

            if not products:
                logger.error("No products were scraped!")
                logger.info("💡 Try running debug_page_structure() manually to inspect the website")
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
            logger.info(f"Discovery methods used: {set(p.get('discovery_method', 'unknown') for p in products)}")
            logger.info(f"Output file: {output_file}")
            logger.info(f"Images directory: {self.images_dir}")
            logger.info("=" * 50)

            return output_file

        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return ""


# Example usage and testing
def test_scraper(url: str, output_dir: str = "scraped_data"):
    """Test the scraper with debugging"""
    scraper = EcommerceScraper(url, output_dir)

    # First, debug the page structure
    scraper.debug_page_structure(url)

    # Then try to scrape
    output_file = scraper.run_full_scrape(max_products_per_category=10)

    return output_file

# Uncomment to test with your URL
# if __name__ == "__main__":
#     url = "https://your-website.com"  # Replace with your URL
#     test_scraper(url)
