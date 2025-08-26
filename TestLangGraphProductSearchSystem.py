#!/usr/bin/env python3
"""
Comprehensive Test Suite for LangGraphProductSearchSystem

This test script includes:
- Unit tests for individual components
- Integration tests for the full workflow
- Performance benchmarks
- Mock data generation for testing
- Error handling validation
"""

import unittest
import asyncio
import time
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the classes to test
try:
    from LangGraphProductSearchSystem import (
        LangGraphProductSearcher,
        SearchConfig,
        ProductSearchState
    )
    from Vector import ProductSeekerVectorDB
    from Integrater import IntegratedProductScraper
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Creating mock classes for testing...")


# Create mock classes if imports fail
class MockProductSeekerVectorDB:
        def __init__(self, db_path, collection_name, model_name):
            self.db_path = db_path
            self.collection_name = collection_name
            self.model_name = model_name

        def search_by_text(self, query, n_results=10):
            return {
                'results': [
                    {
                        'id': f'product_{i}',
                        'similarity': 0.8 - (i * 0.1),
                        'metadata': {
                            'title': f'Test Product {i}',
                            'price': f'${100 + i * 10}',
                            'category': 'Electronics',
                            'brand': f'Brand{i}',
                            'description': f'Description for test product {i}'
                        }
                    }
                    for i in range(min(n_results, 5))
                ]
            }

        def search_by_image(self, image_path, n_results=10):
            return self.search_by_text("image search", n_results)

        def get_database_stats(self):
            return {
                'total_products': 1000,
                'collections': ['ecommerce_test'],
                'model': self.model_name
            }


class MockIntegratedProductScraper:
    pass


class TestSearchConfig(unittest.TestCase):
    """Test the SearchConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = SearchConfig()

        self.assertEqual(config.max_results, 15)
        self.assertEqual(config.min_similarity_threshold, 0.5)
        self.assertEqual(config.max_refinements, 2)
        self.assertTrue(config.enable_caching)
        self.assertEqual(config.cache_ttl, 300)
        self.assertTrue(config.enable_parallel_search)
        self.assertEqual(config.search_timeout, 30)

    def test_custom_config(self):
        """Test custom configuration values"""
        config = SearchConfig(
            max_results=20,
            min_similarity_threshold=0.7,
            max_refinements=3,
            enable_caching=False,
            cache_ttl=600,
            enable_parallel_search=False,
            search_timeout=60
        )

        self.assertEqual(config.max_results, 20)
        self.assertEqual(config.min_similarity_threshold, 0.7)
        self.assertEqual(config.max_refinements, 3)
        self.assertFalse(config.enable_caching)
        self.assertEqual(config.cache_ttl, 600)
        self.assertFalse(config.enable_parallel_search)
        self.assertEqual(config.search_timeout, 60)


class TestLangGraphProductSearcher(unittest.TestCase):
    """Test the main LangGraphProductSearcher class"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_db_path = "test_db"
        self.test_collection = "test_collection"
        self.test_model = "test_model"

        # Mock the vector database
        with patch('LangGraphProductSearchSystem.ProductSeekerVectorDB', MockProductSeekerVectorDB):
            self.searcher = LangGraphProductSearcher(
                db_path=self.test_db_path,
                collection_name=self.test_collection,
                model_name=self.test_model
            )

    def test_initialization(self):
        """Test proper initialization"""
        self.assertIsNotNone(self.searcher)
        self.assertIsNotNone(self.searcher.config)
        self.assertIsNotNone(self.searcher.graph)
        self.assertEqual(self.searcher._performance_stats["total_searches"], 0)

    def test_preprocess_query(self):
        """Test query preprocessing"""
        test_cases = [
            ("gaming laptop", "gaming laptop"),
            ("THE best gaming laptop", "best gaming laptop"),
            ("a great smartphone with camera", "great smartphone camera"),
            ("", ""),
            ("   wireless headphones   ", "wireless headphones")
        ]

        for input_query, expected in test_cases:
            result = self.searcher._preprocess_query(input_query)
            self.assertEqual(result, expected, f"Failed for input: '{input_query}'")

    def test_detect_search_type(self):
        """Test search type detection"""
        test_cases = [
            ("gaming laptop", "text"),
            ("wireless headphones", "text"),
            ("similar to this phone", "hybrid"),
            ("looks like this design", "hybrid"),
            ("", "text")
        ]

        for query, expected_type in test_cases:
            result = self.searcher._detect_search_type(query)
            self.assertEqual(result, expected_type, f"Failed for query: '{query}'")

    def test_cache_key_generation(self):
        """Test cache key generation"""
        test_cases = [
            ("Gaming Laptop", "text", "text:gaming laptop"),
            ("WIRELESS HEADPHONES", "image", "image:wireless headphones"),
            ("  Mixed Case Query  ", "hybrid", "hybrid:mixed case query")
        ]

        for query, search_type, expected_key in test_cases:
            result = self.searcher._generate_cache_key(query, search_type)
            self.assertEqual(result, expected_key)

    def test_merge_results(self):
        """Test result merging functionality"""
        text_results = [
            {'id': '1', 'metadata': {'title': 'Product 1'}, 'similarity': 0.9},
            {'id': '2', 'metadata': {'title': 'Product 2'}, 'similarity': 0.8}
        ]

        image_results = [
            {'id': '2', 'metadata': {'title': 'Product 2'}, 'similarity': 0.7},  # Duplicate
            {'id': '3', 'metadata': {'title': 'Product 3'}, 'similarity': 0.6}
        ]

        merged = self.searcher._merge_results(text_results, image_results)

        self.assertEqual(len(merged), 3)  # Should deduplicate
        self.assertEqual(merged[0]['id'], '1')  # Text results should be prioritized
        self.assertEqual(merged[1]['id'], '2')
        self.assertEqual(merged[2]['id'], '3')

    def test_enhance_result_metadata(self):
        """Test result metadata enhancement"""
        result = {
            'metadata': {
                'title': 'Test Product',
                'price': '$199.99',
                'category': 'Electronics'
            },
            'similarity': 0.85
        }

        self.searcher._enhance_result_metadata(result)

        self.assertEqual(result['metadata']['price_numeric'], 199.99)
        self.assertEqual(result['metadata']['relevance_score'], 0.85)

    def test_refinement_strategy(self):
        """Test query refinement strategies"""
        test_cases = [
            ("gaming laptop computer", 0, "no_results", "gaming laptop"),  # Remove last word
            ("wireless bluetooth headphones", 0, "poor", "wireless bluetooth"),  # Remove last word
            ("smartphone", 1, "few_results", "electronics"),  # Category fallback
            ("unknown product", 1, "poor", "unknown"),  # First word fallback
        ]

        for query, refinement_count, quality_status, expected_contains in test_cases:
            result = self.searcher._apply_refinement_strategy(query, refinement_count, quality_status)
            self.assertIn(expected_contains.split()[0], result.split())


class TestSearchWorkflow(unittest.TestCase):
    """Test the complete search workflow"""

    def setUp(self):
        """Set up test fixtures"""
        with patch('LangGraphProductSearchSystem.ProductSeekerVectorDB', MockProductSeekerVectorDB):
            self.searcher = LangGraphProductSearcher(
                db_path="test_db",
                collection_name="test_collection",
                model_name="test_model"
            )

    def test_successful_search(self):
        """Test successful search workflow"""
        result = self.searcher.search("gaming laptop")

        self.assertTrue(result["success"])
        self.assertGreater(len(result["results"]), 0)
        self.assertIn("messages", result)
        self.assertIn("metadata", result)

        # Check metadata
        metadata = result["metadata"]
        self.assertEqual(metadata["original_query"], "gaming laptop")
        self.assertIn("total_time", metadata)
        self.assertGreaterEqual(metadata["quality_score"], 0.0)

    def test_empty_query_search(self):
        """Test search with empty query"""
        result = self.searcher.search("")

        # Should still work but may return fewer/no results
        self.assertIn("success", result)
        self.assertIn("results", result)

    def test_search_with_custom_config(self):
        """Test search with custom configuration"""
        custom_config = SearchConfig(
            max_results=5,
            min_similarity_threshold=0.8,
            enable_caching=False
        )

        result = self.searcher.search("wireless headphones", config=custom_config)

        self.assertTrue(result["success"])
        # Results should be filtered by custom threshold
        for result_item in result["results"]:
            self.assertGreaterEqual(result_item.get("similarity", 0), 0.8)

    def test_search_type_detection(self):
        """Test automatic search type detection"""
        test_cases = [
            ("gaming laptop", "text"),
            ("similar to this phone", "hybrid"),
        ]

        for query, expected_type in test_cases:
            result = self.searcher.search(query, search_type="auto")
            self.assertEqual(result["metadata"]["search_type"], expected_type)


class TestCacheSystem(unittest.TestCase):
    """Test the caching system"""

    def setUp(self):
        """Set up test fixtures"""
        config = SearchConfig(enable_caching=True, cache_ttl=1)  # Short TTL for testing

        with patch('LangGraphProductSearchSystem.ProductSeekerVectorDB', MockProductSeekerVectorDB):
            self.searcher = LangGraphProductSearcher(
                db_path="test_db",
                collection_name="test_collection",
                model_name="test_model",
                config=config
            )

    def test_cache_hit_miss(self):
        """Test cache hit and miss behavior"""
        query = "test query"

        # First search should be a cache miss
        result1 = self.searcher.search(query)
        self.assertEqual(result1["metadata"]["cache_status"], "miss")

        # Second search should be a cache hit
        result2 = self.searcher.search(query)
        self.assertEqual(result2["metadata"]["cache_status"], "hit")

        # Results should be identical
        self.assertEqual(len(result1["results"]), len(result2["results"]))

    def test_cache_expiration(self):
        """Test cache expiration"""
        query = "expiration test"

        # First search
        self.searcher.search(query)

        # Wait for cache to expire
        time.sleep(1.5)

        # Search again - should be cache miss due to expiration
        result = self.searcher.search(query)
        self.assertEqual(result["metadata"]["cache_status"], "miss")

    def test_cache_disabled(self):
        """Test behavior when cache is disabled"""
        config = SearchConfig(enable_caching=False)

        with patch('LangGraphProductSearchSystem.ProductSeekerVectorDB', MockProductSeekerVectorDB):
            searcher = LangGraphProductSearcher(
                db_path="test_db",
                collection_name="test_collection",
                model_name="test_model",
                config=config
            )

        result = searcher.search("test query")
        self.assertEqual(result["metadata"]["cache_status"], "disabled")

    def test_cache_cleanup(self):
        """Test cache cleanup functionality"""
        # Add some entries to cache
        self.searcher.search("query1")
        self.searcher.search("query2")

        initial_cache_size = len(self.searcher._search_cache)
        self.assertGreater(initial_cache_size, 0)

        # Clear cache
        self.searcher.clear_cache()
        self.assertEqual(len(self.searcher._search_cache), 0)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance monitoring and metrics"""

    def setUp(self):
        """Set up test fixtures"""
        with patch('LangGraphProductSearchSystem.ProductSeekerVectorDB', MockProductSeekerVectorDB):
            self.searcher = LangGraphProductSearcher(
                db_path="test_db",
                collection_name="test_collection",
                model_name="test_model"
            )

    def test_performance_stats_initialization(self):
        """Test initial performance stats"""
        stats = self.searcher.get_performance_stats()

        expected_keys = [
            "total_searches", "average_response_time",
            "cache_hits", "cache_misses", "cache_size", "cache_hit_rate"
        ]

        for key in expected_keys:
            self.assertIn(key, stats)

        self.assertEqual(stats["total_searches"], 0)
        self.assertEqual(stats["average_response_time"], 0.0)

    def test_performance_stats_update(self):
        """Test performance stats updates after searches"""
        initial_stats = self.searcher.get_performance_stats()

        # Perform some searches
        self.searcher.search("query1")
        self.searcher.search("query2")
        self.searcher.search("query1")  # This should be a cache hit

        updated_stats = self.searcher.get_performance_stats()

        self.assertEqual(updated_stats["total_searches"], 3)
        self.assertGreater(updated_stats["average_response_time"], 0)
        self.assertEqual(updated_stats["cache_hits"], 1)
        self.assertEqual(updated_stats["cache_misses"], 2)
        self.assertGreater(updated_stats["cache_hit_rate"], 0)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""

    def setUp(self):
        """Set up test fixtures"""
        with patch('LangGraphProductSearchSystem.ProductSeekerVectorDB', MockProductSeekerVectorDB):
            self.searcher = LangGraphProductSearcher(
                db_path="test_db",
                collection_name="test_collection",
                model_name="test_model"
            )

    def test_database_error_handling(self):
        """Test handling of database errors"""
        # Mock database to raise an exception
        with patch.object(self.searcher.db, 'search_by_text', side_effect=Exception("DB Error")):
            result = self.searcher.search("test query")

            # Should handle error gracefully
            self.assertIn("success", result)
            self.assertIn("results", result)

    def test_invalid_similarity_threshold(self):
        """Test handling of invalid similarity thresholds"""
        config = SearchConfig(min_similarity_threshold=1.5)  # Invalid threshold

        result = self.searcher.search("test query", config=config)

        # Should still work (threshold will be ignored or clamped)
        self.assertIn("success", result)

    def test_extreme_max_results(self):
        """Test handling of extreme max_results values"""
        config = SearchConfig(max_results=0)

        result = self.searcher.search("test query", config=config)

        # Should handle gracefully
        self.assertIn("success", result)


class PerformanceBenchmark:
    """Performance benchmark suite"""

    def __init__(self):
        with patch('LangGraphProductSearchSystem.ProductSeekerVectorDB', MockProductSeekerVectorDB):
            self.searcher = LangGraphProductSearcher(
                db_path="benchmark_db",
                collection_name="benchmark_collection",
                model_name="benchmark_model"
            )

    def benchmark_search_speed(self, num_searches=100):
        """Benchmark search speed"""
        queries = [
            "gaming laptop", "wireless headphones", "smartphone android",
            "bluetooth speaker", "tablet computer", "smart watch",
            "digital camera", "external hard drive", "usb cable",
            "power bank"
        ]

        start_time = time.time()

        for i in range(num_searches):
            query = queries[i % len(queries)]
            self.searcher.search(f"{query} {i}")  # Make each query unique

        total_time = time.time() - start_time
        avg_time = total_time / num_searches

        return {
            "total_searches": num_searches,
            "total_time": total_time,
            "average_time": avg_time,
            "searches_per_second": num_searches / total_time
        }

    def benchmark_cache_performance(self, num_searches=50):
        """Benchmark cache performance"""
        repeated_query = "benchmark cache query"

        # First search (cache miss)
        start_time = time.time()
        self.searcher.search(repeated_query)
        first_search_time = time.time() - start_time

        # Subsequent searches (cache hits)
        start_time = time.time()
        for _ in range(num_searches - 1):
            self.searcher.search(repeated_query)
        cached_searches_time = time.time() - start_time

        avg_cached_time = cached_searches_time / (num_searches - 1)

        return {
            "first_search_time": first_search_time,
            "average_cached_time": avg_cached_time,
            "speedup_ratio": first_search_time / avg_cached_time if avg_cached_time > 0 else float('inf')
        }


def run_test_suite():
    """Run the complete test suite"""
    print("🧪 Running LangGraph Product Search Test Suite")
    print("=" * 60)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestSearchConfig,
        TestLangGraphProductSearcher,
        TestSearchWorkflow,
        TestCacheSystem,
        TestPerformanceMetrics,
        TestErrorHandling
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    print(f"\n📊 Test Results:")
    print(f"✅ Tests run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"🚫 Errors: {len(result.errors)}")

    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\n')[0]}")

    if result.errors:
        print(f"\n🚫 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\n')[-2]}")

    return result.wasSuccessful()


def run_performance_benchmarks():
    """Run performance benchmarks"""
    print("\n🚀 Running Performance Benchmarks")
    print("=" * 60)

    try:
        benchmark = PerformanceBenchmark()

        # Search speed benchmark
        print("📈 Search Speed Benchmark...")
        speed_results = benchmark.benchmark_search_speed(50)  # Reduced for demo

        print(f"  Total searches: {speed_results['total_searches']}")
        print(f"  Total time: {speed_results['total_time']:.2f}s")
        print(f"  Average time per search: {speed_results['average_time']:.3f}s")
        print(f"  Searches per second: {speed_results['searches_per_second']:.1f}")

        # Cache performance benchmark
        print("\n💾 Cache Performance Benchmark...")
        cache_results = benchmark.benchmark_cache_performance(20)  # Reduced for demo

        print(f"  First search time: {cache_results['first_search_time']:.3f}s")
        print(f"  Average cached search time: {cache_results['average_cached_time']:.3f}s")
        print(f"  Cache speedup ratio: {cache_results['speedup_ratio']:.1f}x")

        return True

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False


def run_integration_tests():
    """Run integration tests with real-world scenarios"""
    print("\n🔧 Running Integration Tests")
    print("=" * 60)

    try:
        with patch('LangGraphProductSearchSystem.ProductSeekerVectorDB', MockProductSeekerVectorDB):
            searcher = LangGraphProductSearcher(
                db_path="integration_test_db",
                collection_name="integration_test_collection",
                model_name="integration_test_model"
            )

        # Test various search scenarios
        test_scenarios = [
            {
                "name": "Basic Text Search",
                "query": "gaming laptop high performance",
                "search_type": "text"
            },
            {
                "name": "Hybrid Search",
                "query": "similar to wireless headphones",
                "search_type": "hybrid"
            },
            {
                "name": "Auto Detection",
                "query": "smartphone android latest",
                "search_type": "auto"
            },
            {
                "name": "Single Word Query",
                "query": "tablet",
                "search_type": "auto"
            },
            {
                "name": "Long Query",
                "query": "high quality bluetooth wireless noise cancelling headphones with microphone",
                "search_type": "text"
            }
        ]

        all_passed = True

        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n{i}. {scenario['name']}:")
            print(f"   Query: '{scenario['query']}'")

            start_time = time.time()
            result = searcher.search(scenario['query'], scenario['search_type'])
            duration = time.time() - start_time

            if result.get('success', False):
                print(f"   ✅ Success ({duration:.2f}s)")
                print(f"   📊 Results: {len(result.get('results', []))}")
                print(f"   🎯 Quality: {result.get('metadata', {}).get('quality_score', 0):.2f}")
                print(f"   🔄 Refinements: {result.get('metadata', {}).get('refinement_count', 0)}")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                all_passed = False

        # Test performance statistics
        print(f"\n📊 Final Performance Stats:")
        stats = searcher.get_performance_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")

        return all_passed

    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
        return False


def main():
    """Main test runner"""
    print("🎯 LangGraph Product Search System - Comprehensive Test Suite")
    print("=" * 80)

    all_passed = True

    # Run unit tests
    unit_tests_passed = run_test_suite()
    all_passed = all_passed and unit_tests_passed

    # Run integration tests
    integration_tests_passed = run_integration_tests()
    all_passed = all_passed and integration_tests_passed

    # Run performance benchmarks
    benchmarks_passed = run_performance_benchmarks()

    # Final summary
    print(f"\n🏁 Test Suite Summary")
    print("=" * 60)
    print(f"✅ Unit Tests: {'PASSED' if unit_tests_passed else 'FAILED'}")
    print(f"✅ Integration Tests: {'PASSED' if integration_tests_passed else 'FAILED'}")
    print(f"✅ Performance Benchmarks: {'PASSED' if benchmarks_passed else 'FAILED'}")
    print(f"\n🎯 Overall Status: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)


# C:\Users\themi\AppData\Local\Programs\Python\Python313\python.exe "C:/Program Files/JetBrains/PyCharm Community Edition 2024.1.2/plugins/python-ce/helpers/pycharm/_jb_unittest_runner.py" --path C:\Users\themi\PycharmProjects\ProductSeeker\TestLangGraphProductSearchSystem.py
# Testing started at 2:27 PM ...
# Launching unittests with arguments python -m unittest C:\Users\themi\PycharmProjects\ProductSeeker\TestLangGraphProductSearchSystem.py in C:\Users\themi\PycharmProjects\ProductSeeker
#
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
# INFO:LangGraphProductSearchSystem:🧹 Search cache cleared
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
#
#
# disabled != unknown
#
# Expected :unknown
# Actual   :disabled
# <Click to see difference>
#
# Traceback (most recent call last):
#   File "C:\Users\themi\PycharmProjects\ProductSeeker\TestLangGraphProductSearchSystem.py", line 350, in test_cache_disabled
#     self.assertEqual(result["metadata"]["cache_status"], "disabled")
#     ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# AssertionError: 'unknown' != 'disabled'
# - unknown
# + disabled
#
#
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
#
#
# miss != unknown
#
# Expected :unknown
# Actual   :miss
# <Click to see difference>
#
# Traceback (most recent call last):
#   File "C:\Users\themi\PycharmProjects\ProductSeeker\TestLangGraphProductSearchSystem.py", line 335, in test_cache_expiration
#     self.assertEqual(result["metadata"]["cache_status"], "miss")
#     ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# AssertionError: 'unknown' != 'miss'
# - unknown
# + miss
#
#
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
#
#
# miss != unknown
#
# Expected :unknown
# Actual   :miss
# <Click to see difference>
#
# Traceback (most recent call last):
#   File "C:\Users\themi\PycharmProjects\ProductSeeker\TestLangGraphProductSearchSystem.py", line 314, in test_cache_hit_miss
#     self.assertEqual(result1["metadata"]["cache_status"], "miss")
#     ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# AssertionError: 'unknown' != 'miss'
# - unknown
# + miss
#
#
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
# ERROR:LangGraphProductSearchSystem:Single search failed: DB Error
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
#
# Failure
# Traceback (most recent call last):
#   File "C:\Users\themi\PycharmProjects\ProductSeeker\TestLangGraphProductSearchSystem.py", line 228, in test_refinement_strategy
#     self.assertIn(expected_contains.split()[0], result.split())
#     ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# AssertionError: 'electronics' not found in ['phone']
#
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
# INFO:LangGraphProductSearchSystem:🚀 Optimized LangGraph ProductSearcher initialized
#
#
# Ran 22 tests in 1.946s
#
# FAILED (failures=4)
#
# Process finished with exit code 1
