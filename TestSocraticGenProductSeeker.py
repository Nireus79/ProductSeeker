#!/usr/bin/env python3
"""
SocraticGenProductSeeker Test Script
===================================

Comprehensive test suite for the Advanced Agentic Product Search System.
Tests all components, agents, and interfaces to ensure proper functionality.

Usage:
    python test_socratic_gen.py
    python test_socratic_gen.py --verbose
    python test_socratic_gen.py --quick
    python test_socratic_gen.py --component search
"""

import unittest
import sys
import os
import tempfile
import time
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import warnings

# Add the current directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from SocraticGenProductSeeker import (
        AgenticProductSearchSystem,
        SystemConfig,
        UserProfile,
        AgenticSearchState,
        VoiceProcessingAgent,
        ImageProcessingAgent,
        IntentAnalysisAgent,
        SearchExecutionAgent,
        RecommendationAgent,
        ResponseFormattingAgent,
        ProductSearchCLI,
        AgentConfig
    )

    MODULE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import SocraticGenProductSeeker: {e}")
    MODULE_AVAILABLE = False


class TestSystemConfig(unittest.TestCase):
    """Test SystemConfig class"""

    def test_default_config(self):
        """Test default configuration values"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        config = SystemConfig()

        self.assertEqual(config.max_results, 20)
        self.assertEqual(config.min_similarity_threshold, 0.5)
        self.assertEqual(config.max_refinements, 3)
        self.assertTrue(config.enable_caching)
        self.assertEqual(config.search_timeout, 30)
        self.assertEqual(config.voice_language, "en-US")
        self.assertEqual(config.tts_rate, 200)
        self.assertEqual(config.database_path, "D:/Vector/ProductSeeker_data")
        self.assertEqual(config.collection_name, 'ecommerce_test')
        self.assertEqual(config.model_name, 'clip-ViT-B-32')

    def test_custom_config(self):
        """Test custom configuration"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        config = SystemConfig(
            max_results=50,
            min_similarity_threshold=0.7,
            voice_language="en-GB",
            database_path="/custom/path"
        )

        self.assertEqual(config.max_results, 50)
        self.assertEqual(config.min_similarity_threshold, 0.7)
        self.assertEqual(config.voice_language, "en-GB")
        self.assertEqual(config.database_path, "/custom/path")


class TestUserProfile(unittest.TestCase):
    """Test UserProfile class"""

    def test_default_profile(self):
        """Test default user profile"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        profile = UserProfile()

        self.assertEqual(profile.user_id, "default")
        self.assertEqual(profile.preferences, {})
        self.assertEqual(profile.search_history, [])
        self.assertEqual(profile.interaction_patterns, {})
        self.assertEqual(profile.preferred_brands, [])
        self.assertEqual(profile.price_range, {})
        self.assertEqual(profile.categories_of_interest, [])
        self.assertIsNotNone(profile.last_updated)

    def test_custom_profile(self):
        """Test custom user profile"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        profile = UserProfile(
            user_id="test_user",
            preferred_brands=["Apple", "Sony"],
            categories_of_interest=["Electronics", "Gaming"]
        )

        self.assertEqual(profile.user_id, "test_user")
        self.assertEqual(profile.preferred_brands, ["Apple", "Sony"])
        self.assertEqual(profile.categories_of_interest, ["Electronics", "Gaming"])


class TestAgentConfig(unittest.TestCase):
    """Test AgentConfig class"""

    def test_default_agent_config(self):
        """Test default agent configuration"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        config = AgentConfig(name="test_agent")

        self.assertEqual(config.name, "test_agent")
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.timeout, 30)
        self.assertEqual(config.priority, 1)
        self.assertTrue(config.enable_learning)

    def test_custom_agent_config(self):
        """Test custom agent configuration"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        config = AgentConfig(
            name="custom_agent",
            max_retries=5,
            timeout=60,
            priority=2,
            enable_learning=False
        )

        self.assertEqual(config.name, "custom_agent")
        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.timeout, 60)
        self.assertEqual(config.priority, 2)
        self.assertFalse(config.enable_learning)


class TestAgents(unittest.TestCase):
    """Test individual agent functionality"""

    def setUp(self):
        """Set up test fixtures"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        self.config = AgentConfig(name="test")
        self.system_config = SystemConfig()

        # Create mock state
        self.mock_state = {
            "messages": [],
            "original_query": "test query",
            "current_query": "test query",
            "search_type": "text",
            "input_data": {"text_query": "test query"},
            "processed_input": {},
            "search_results": [],
            "user_intent": {},
            "user_profile": UserProfile(),
            "performance_metrics": {},
            "explanations": [],
            "suggestions": [],
            "refinement_count": 0,
            "session_id": "test_session",
            "timestamp": time.time()
        }

    def test_voice_agent_initialization(self):
        """Test voice processing agent initialization"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        agent = VoiceProcessingAgent(self.config, self.system_config)
        self.assertEqual(agent.name, "VoiceProcessor")
        self.assertIsNotNone(agent.available)

    def test_voice_agent_execute(self):
        """Test voice processing agent execution"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        agent = VoiceProcessingAgent(self.config, self.system_config)

        # Test with voice search type
        state = self.mock_state.copy()
        state["search_type"] = "voice"

        result_state = agent.execute(state)

        self.assertIsInstance(result_state, dict)
        self.assertIn("processed_input", result_state)

    def test_image_agent_initialization(self):
        """Test image processing agent initialization"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        agent = ImageProcessingAgent(self.config)
        self.assertEqual(agent.name, "ImageProcessor")
        self.assertIsNotNone(agent.available)

    def test_image_agent_execute(self):
        """Test image processing agent execution"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        agent = ImageProcessingAgent(self.config)

        # Test with image search type
        state = self.mock_state.copy()
        state["search_type"] = "image"
        state["input_data"]["image_path"] = "/fake/path/image.jpg"

        result_state = agent.execute(state)

        self.assertIsInstance(result_state, dict)
        self.assertIn("processed_input", result_state)

    def test_intent_agent_initialization(self):
        """Test intent analysis agent initialization"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        agent = IntentAnalysisAgent(self.config)
        self.assertEqual(agent.name, "IntentAnalyzer")
        self.assertIsNotNone(agent.intent_patterns)

    def test_intent_agent_execute(self):
        """Test intent analysis agent execution"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        agent = IntentAnalysisAgent(self.config)

        state = self.mock_state.copy()
        state["current_query"] = "buy gaming laptop under $1000"

        result_state = agent.execute(state)

        self.assertIsInstance(result_state, dict)
        self.assertIn("user_intent", result_state)

        # Check intent detection
        intent = result_state["user_intent"]
        self.assertIn("primary", intent)
        self.assertIn("confidence", intent)

    def test_recommendation_agent_initialization(self):
        """Test recommendation agent initialization"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        agent = RecommendationAgent(self.config)
        self.assertEqual(agent.name, "RecommendationEngine")

    def test_recommendation_agent_execute(self):
        """Test recommendation agent execution"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        agent = RecommendationAgent(self.config)

        state = self.mock_state.copy()
        state["user_intent"] = {"primary": "browse", "confidence": 0.8}

        result_state = agent.execute(state)

        self.assertIsInstance(result_state, dict)
        self.assertIn("suggestions", result_state)

    def test_response_agent_initialization(self):
        """Test response formatting agent initialization"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        agent = ResponseFormattingAgent(self.config, self.system_config)
        self.assertEqual(agent.name, "ResponseFormatter")

    def test_response_agent_execute_no_results(self):
        """Test response formatting with no results"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        agent = ResponseFormattingAgent(self.config, self.system_config)

        state = self.mock_state.copy()
        state["search_results"] = []

        result_state = agent.execute(state)

        self.assertIsInstance(result_state, dict)
        self.assertIn("messages", result_state)
        self.assertTrue(len(result_state["messages"]) > 0)

    def test_response_agent_execute_with_results(self):
        """Test response formatting with results"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        agent = ResponseFormattingAgent(self.config, self.system_config)

        # Mock search results
        mock_results = [
            {
                "similarity": 0.9,
                "metadata": {
                    "title": "Gaming Laptop Pro",
                    "price": "$999.99",
                    "category": "Electronics",
                    "brand": "TechBrand",
                    "description": "High-performance gaming laptop"
                }
            }
        ]

        state = self.mock_state.copy()
        state["search_results"] = mock_results

        result_state = agent.execute(state)

        self.assertIsInstance(result_state, dict)
        self.assertIn("messages", result_state)
        self.assertTrue(len(result_state["messages"]) > 0)


class TestAgenticProductSearchSystem(unittest.TestCase):
    """Test main search system"""

    @patch('SocraticGenProductSeeker.ProductSeekerVectorDB')
    def test_system_initialization(self, mock_db):
        """Test system initialization"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        # Mock database
        mock_db_instance = Mock()
        mock_db_instance.get_database_stats.return_value = {
            "total_products": 1000,
            "collections": 1,
            "last_updated": "2024-01-01",
            "status": "active"
        }
        mock_db.return_value = mock_db_instance

        # Initialize system
        with patch('pathlib.Path.exists', return_value=True):
            system = AgenticProductSearchSystem()

        self.assertIsNotNone(system.db)
        self.assertIsNotNone(system.graph)
        self.assertEqual(len(system.user_profiles), 0)
        self.assertIn("total_queries", system.system_stats)

    @patch('SocraticGenProductSeeker.ProductSeekerVectorDB')
    def test_text_search(self, mock_db):
        """Test text search functionality"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        # Mock database responses
        mock_db_instance = Mock()
        mock_results = {
            'results': [
                {
                    'id': 'product_1',
                    'similarity': 0.9,
                    'metadata': {
                        'title': 'Gaming Laptop',
                        'price': '$999.99',
                        'category': 'Electronics',
                        'brand': 'TechBrand'
                    }
                }
            ]
        }
        mock_db_instance.search_by_text.return_value = mock_results
        mock_db_instance.get_database_stats.return_value = {"total_products": 1000}
        mock_db.return_value = mock_db_instance

        # Initialize and test
        with patch('pathlib.Path.exists', return_value=True):
            system = AgenticProductSearchSystem()

        result = system.search(query="gaming laptop", search_type="text")

        self.assertTrue(result["success"])
        self.assertEqual(len(result["results"]), 1)
        self.assertIn("response", result)
        self.assertIn("explanations", result)
        self.assertIn("suggestions", result)
        self.assertEqual(result["search_type"], "text")

    @patch('SocraticGenProductSeeker.ProductSeekerVectorDB')
    def test_image_search(self, mock_db):
        """Test image search functionality"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        # Mock database
        mock_db_instance = Mock()
        mock_results = {'results': []}
        mock_db_instance.search_by_image.return_value = mock_results
        mock_db_instance.get_database_stats.return_value = {"total_products": 1000}
        mock_db.return_value = mock_db_instance

        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(b'fake image data')
            temp_image_path = tmp_file.name

        try:
            with patch('pathlib.Path.exists', return_value=True):
                system = AgenticProductSearchSystem()

            result = system.search_by_image(temp_image_path)

            self.assertIsInstance(result, dict)
            self.assertIn("success", result)
            self.assertEqual(result.get("search_type", "image"), "image")

        finally:
            # Cleanup
            os.unlink(temp_image_path)

    @patch('SocraticGenProductSeeker.ProductSeekerVectorDB')
    def test_voice_search(self, mock_db):
        """Test voice search functionality"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        # Mock database
        mock_db_instance = Mock()
        mock_db_instance.search_by_text.return_value = {'results': []}
        mock_db_instance.get_database_stats.return_value = {"total_products": 1000}
        mock_db.return_value = mock_db_instance

        with patch('pathlib.Path.exists', return_value=True):
            system = AgenticProductSearchSystem()

        result = system.search_by_voice(duration=1)

        self.assertIsInstance(result, dict)
        self.assertIn("success", result)

    @patch('SocraticGenProductSeeker.ProductSeekerVectorDB')
    def test_user_profile_management(self, mock_db):
        """Test user profile management"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        # Mock database
        mock_db_instance = Mock()
        mock_db_instance.get_database_stats.return_value = {"total_products": 1000}
        mock_db.return_value = mock_db_instance

        with patch('pathlib.Path.exists', return_value=True):
            system = AgenticProductSearchSystem()

        # Test profile creation
        profile = system.get_user_profile("test_user")
        self.assertIsInstance(profile, UserProfile)
        self.assertEqual(profile.user_id, "test_user")

        # Test profile retrieval
        same_profile = system.get_user_profile("test_user")
        self.assertEqual(profile, same_profile)

    @patch('SocraticGenProductSeeker.ProductSeekerVectorDB')
    def test_system_status(self, mock_db):
        """Test system status functionality"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        # Mock database
        mock_db_instance = Mock()
        mock_db_instance.get_database_stats.return_value = {"total_products": 1000}
        mock_db.return_value = mock_db_instance

        with patch('pathlib.Path.exists', return_value=True):
            system = AgenticProductSearchSystem()

        status = system.get_system_status()

        self.assertIsInstance(status, dict)
        self.assertIn("system_ready", status)
        self.assertIn("database_connected", status)
        self.assertIn("voice_available", status)
        self.assertIn("image_available", status)
        self.assertIn("session_id", status)
        self.assertIn("stats", status)
        self.assertIn("config", status)

    @patch('SocraticGenProductSeeker.ProductSeekerVectorDB')
    def test_database_info(self, mock_db):
        """Test database information retrieval"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        # Mock database
        mock_db_instance = Mock()
        mock_stats = {
            "total_products": 1000,
            "collections": 1,
            "last_updated": "2024-01-01"
        }
        mock_db_instance.get_database_stats.return_value = mock_stats
        mock_db.return_value = mock_db_instance

        with patch('pathlib.Path.exists', return_value=True):
            system = AgenticProductSearchSystem()

        db_info = system.get_database_info()

        self.assertIsInstance(db_info, dict)
        self.assertIn("system_stats", db_info)
        self.assertIn("agents_available", db_info)

    @patch('SocraticGenProductSeeker.ProductSeekerVectorDB')
    def test_async_search(self, mock_db):
        """Test async search functionality"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        # Mock database
        mock_db_instance = Mock()
        mock_results = {'results': []}
        mock_db_instance.search_by_text.return_value = mock_results
        mock_db_instance.get_database_stats.return_value = {"total_products": 1000}
        mock_db.return_value = mock_db_instance

        async def run_async_test():
            with patch('pathlib.Path.exists', return_value=True):
                system = AgenticProductSearchSystem()

            result = await system.search_async(query="test", search_type="text")
            self.assertIsInstance(result, dict)
            self.assertIn("success", result)

        # Run async test
        asyncio.run(run_async_test())


class TestCLIInterface(unittest.TestCase):
    """Test CLI interface"""

    @patch('SocraticGenProductSeeker.ProductSeekerVectorDB')
    @patch('sys.stdout', new_callable=StringIO)
    def test_cli_initialization(self, mock_stdout, mock_db):
        """Test CLI initialization"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        # Mock database
        mock_db_instance = Mock()
        mock_db_instance.get_database_stats.return_value = {"total_products": 1000}
        mock_db.return_value = mock_db_instance

        with patch('pathlib.Path.exists', return_value=True):
            cli = ProductSearchCLI()

        self.assertIsNotNone(cli.search_system)

        # Check initialization output
        output = mock_stdout.getvalue()
        self.assertIn("Advanced Agentic Product Search System", output)

    @patch('SocraticGenProductSeeker.ProductSeekerVectorDB')
    def test_cli_text_search(self, mock_db):
        """Test CLI text search"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        # Mock database
        mock_db_instance = Mock()
        mock_results = {'results': []}
        mock_db_instance.search_by_text.return_value = mock_results
        mock_db_instance.get_database_stats.return_value = {"total_products": 1000}
        mock_db.return_value = mock_db_instance

        with patch('pathlib.Path.exists', return_value=True):
            cli = ProductSearchCLI()

        # Mock the search result
        with patch.object(cli.search_system, 'search') as mock_search:
            mock_search.return_value = {
                "success": True,
                "results": [],
                "response": "Test response",
                "explanations": [],
                "response_time": 0.1
            }

            # Capture print output
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                cli.text_search("test query")
                output = mock_stdout.getvalue()

            self.assertIn("Test response", output)


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios"""

    def test_missing_image_file(self):
        """Test handling of missing image files"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        with patch('SocraticGenProductSeeker.ProductSeekerVectorDB'):
            with patch('pathlib.Path.exists', return_value=True):
                system = AgenticProductSearchSystem()

        result = system.search_by_image("/nonexistent/path/image.jpg")

        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("Image file not found", result["response"])

    @patch('SocraticGenProductSeeker.ProductSeekerVectorDB')
    def test_database_error_handling(self, mock_db):
        """Test database error handling"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        # Mock database that raises an exception
        mock_db_instance = Mock()
        mock_db_instance.search_by_text.side_effect = Exception("Database error")
        mock_db_instance.get_database_stats.return_value = {"total_products": 1000}
        mock_db.return_value = mock_db_instance

        with patch('pathlib.Path.exists', return_value=True):
            system = AgenticProductSearchSystem()

        result = system.search(query="test", search_type="text")

        # Should handle error gracefully
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)

    @patch('SocraticGenProductSeeker.ProductSeekerVectorDB')
    def test_agent_failure_handling(self, mock_db):
        """Test agent failure handling"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        # Mock database
        mock_db_instance = Mock()
        mock_db_instance.search_by_text.return_value = {'results': []}
        mock_db_instance.get_database_stats.return_value = {"total_products": 1000}
        mock_db.return_value = mock_db_instance

        with patch('pathlib.Path.exists', return_value=True):
            system = AgenticProductSearchSystem()

        # Mock agent failure
        with patch.object(system.intent_agent, 'execute', side_effect=Exception("Agent failed")):
            result = system.search(query="test", search_type="text")

            # Should still return a result
            self.assertIsInstance(result, dict)


class TestPerformance(unittest.TestCase):
    """Test performance aspects"""

    @patch('SocraticGenProductSeeker.ProductSeekerVectorDB')
    def test_response_time_tracking(self, mock_db):
        """Test response time tracking"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        # Mock database
        mock_db_instance = Mock()
        mock_db_instance.search_by_text.return_value = {'results': []}
        mock_db_instance.get_database_stats.return_value = {"total_products": 1000}
        mock_db.return_value = mock_db_instance

        with patch('pathlib.Path.exists', return_value=True):
            system = AgenticProductSearchSystem()

        initial_stats = system.system_stats.copy()

        result = system.search(query="test", search_type="text")

        # Check that stats were updated
        self.assertGreater(system.system_stats["total_queries"], initial_stats["total_queries"])
        self.assertIn("response_time", result)
        self.assertIsInstance(result["response_time"], float)

    @patch('SocraticGenProductSeeker.ProductSeekerVectorDB')
    def test_multiple_searches(self, mock_db):
        """Test multiple search operations"""
        if not MODULE_AVAILABLE:
            self.skipTest("Module not available")

        # Mock database
        mock_db_instance = Mock()
        mock_db_instance.search_by_text.return_value = {'results': []}
        mock_db_instance.get_database_stats.return_value = {"total_products": 1000}
        mock_db.return_value = mock_db_instance

        with patch('pathlib.Path.exists', return_value=True):
            system = AgenticProductSearchSystem()

        # Perform multiple searches
        queries = ["laptop", "phone", "headphones", "mouse", "keyboard"]
        results = []

        for query in queries:
            result = system.search(query=query, search_type="text")
            results.append(result)

        # Check all searches completed
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIn("success", result)

        # Check stats
        self.assertEqual(system.system_stats["total_queries"], 5)


def create_test_suite(test_type="all"):
    """Create test suite based on type"""
    suite = unittest.TestSuite()

    if test_type == "all" or test_type == "config":
        suite.addTest(unittest.makeSuite(TestSystemConfig))
        suite.addTest(unittest.makeSuite(TestUserProfile))
        suite.addTest(unittest.makeSuite(TestAgentConfig))

    if test_type == "all" or test_type == "agents":
        suite.addTest(unittest.makeSuite(TestAgents))

    if test_type == "all" or test_type == "search":
        suite.addTest(unittest.makeSuite(TestAgenticProductSearchSystem))

    if test_type == "all" or test_type == "cli":
        suite.addTest(unittest.makeSuite(TestCLIInterface))

    if test_type == "all" or test_type == "error":
        suite.addTest(unittest.makeSuite(TestErrorHandling))

    if test_type == "all" or test_type == "performance":
        suite.addTest(unittest.makeSuite(TestPerformance))

    return suite


def run_tests(verbose=False, test_type="all", quick=False):
    """Run the test suite"""
    print("🧪 SocraticGenProductSeeker Test Suite")
    print("=" * 50)

    if not MODULE_AVAILABLE:
        print("❌ Cannot run tests - SocraticGenProductSeeker module not available")
        print("   Please ensure the module is in the Python path")
        return False

    # Suppress warnings if not verbose
    if not verbose:
        warnings.filterwarnings("ignore")

    # Create test suite
    suite = create_test_suite(test_type)

    # Configure test runner
    runner = unittest.TextTestRunner(
        verbosity=2 if verbose else 1,
        stream=sys.stdout
    )

    if quick:
        print("🏃 Running quick tests (skipping performance tests)")
        # Remove performance tests for quick run
        suite = create_test_suite("config")

    print(f"🎯 Running {test_type} tests...")
    print()

    # Run tests
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()

    # Print summary
    print()
    print("📊 Test Summary")
    print("-" * 30)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"Duration: {end_time - start_time:.2f}s")
    print(f"Duration: {end_time - start_time:.2f}s")

    # Print detailed failure information if verbose
    if verbose and (result.failures or result.errors):
        print("\n💥 Detailed Failure Information")
        print("-" * 40)

        for test, traceback in result.failures:
            print(f"\n❌ FAILURE: {test}")
            print(traceback)

        for test, traceback in result.errors:
            print(f"\n💀 ERROR: {test}")
            print(traceback)

    # Return success status
    success = len(result.failures) == 0 and len(result.errors) == 0

    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")

    return success


def print_help():
    """Print help information"""
    print("""
SocraticGenProductSeeker Test Suite
==================================

Usage:
    python TestSocraticGenProductSeeker.py [options]

Options:
    --help, -h          Show this help message
    --verbose, -v       Run tests with verbose output
    --quick, -q         Run quick tests only (skip performance tests)
    --component TYPE    Run tests for specific component

Component Types:
    all         Run all tests (default)
    config      Test configuration classes
    agents      Test individual agents
    search      Test main search system
    cli         Test CLI interface
    error       Test error handling
    performance Test performance aspects

Examples:
    python TestSocraticGenProductSeeker.py
    python TestSocraticGenProductSeeker.py --verbose
    python TestSocraticGenProductSeeker.py --quick
    python TestSocraticGenProductSeeker.py --component agents
    python TestSocraticGenProductSeeker.py --component search --verbose
    """)


def main():
    """Main function to run tests with command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test suite for SocraticGenProductSeeker",
        add_help=False
    )

    parser.add_argument(
        '--help', '-h',
        action='store_true',
        help='Show help message'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Run with verbose output'
    )

    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run quick tests only'
    )

    parser.add_argument(
        '--component', '-c',
        choices=['all', 'config', 'agents', 'search', 'cli', 'error', 'performance'],
        default='all',
        help='Component to test'
    )

    # Parse arguments
    args = parser.parse_args()

    # Show help if requested
    if args.help:
        print_help()
        return

    # Check if module is available before proceeding
    if not MODULE_AVAILABLE:
        print("❌ SocraticGenProductSeeker module not found!")
        print("   Please ensure the module is in the same directory or Python path.")
        sys.exit(1)

    # Run tests
    try:
        success = run_tests(
            verbose=args.verbose,
            test_type=args.component,
            quick=args.quick
        )

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\n💥 Unexpected error running tests: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# C:\Users\themi\AppData\Local\Programs\Python\Python313\python.exe "C:/Program Files/JetBrains/PyCharm Community Edition 2024.1.2/plugins/python-ce/helpers/pycharm/_jb_unittest_runner.py" --path C:\Users\themi\PycharmProjects\ProductSeeker\TestSocraticGenProductSeeker.py
# Testing started at 10:56 AM ...
# Launching unittests with arguments python -m unittest C:\Users\themi\PycharmProjects\ProductSeeker\TestSocraticGenProductSeeker.py in C:\Users\themi\PycharmProjects\ProductSeeker
#
# ✅ Database modules loaded successfully
# INFO:SocraticGenProductSeeker:🔌 Connecting to database at: D:/Vector/ProductSeeker_data
# WARNING:SocraticGenProductSeeker:Voice initialization failed: No Default Input Device Available
# INFO:SocraticGenProductSeeker:🤖 All agents initialized successfully
# INFO:SocraticGenProductSeeker:🤖 Advanced Agentic Product Search System initialized
# INFO:SocraticGenProductSeeker:📁 Database path: D:/Vector/ProductSeeker_data
# INFO:SocraticGenProductSeeker:📊 Collection: ecommerce_test
# INFO:SocraticGenProductSeeker:🧠 Model: clip-ViT-B-32
# INFO:SocraticGenProductSeeker:🚀 Starting text search: 'test'
# INFO:SocraticGenProductSeeker:✅ Search completed in 0.01s
# INFO:SocraticGenProductSeeker:🔌 Connecting to database at: D:/Vector/ProductSeeker_data
# WARNING:SocraticGenProductSeeker:Voice initialization failed: No Default Input Device Available
# INFO:SocraticGenProductSeeker:🤖 All agents initialized successfully
# INFO:SocraticGenProductSeeker:🤖 Advanced Agentic Product Search System initialized
# INFO:SocraticGenProductSeeker:📁 Database path: D:/Vector/ProductSeeker_data
# INFO:SocraticGenProductSeeker:📊 Collection: ecommerce_test
# INFO:SocraticGenProductSeeker:🧠 Model: clip-ViT-B-32
# INFO:SocraticGenProductSeeker:🔌 Connecting to database at: D:/Vector/ProductSeeker_data
# WARNING:SocraticGenProductSeeker:Voice initialization failed: No Default Input Device Available
# INFO:SocraticGenProductSeeker:🤖 All agents initialized successfully
# INFO:SocraticGenProductSeeker:🤖 Advanced Agentic Product Search System initialized
# INFO:SocraticGenProductSeeker:📁 Database path: D:/Vector/ProductSeeker_data
# INFO:SocraticGenProductSeeker:📊 Collection: ecommerce_test
# INFO:SocraticGenProductSeeker:🧠 Model: clip-ViT-B-32
# INFO:SocraticGenProductSeeker:🚀 Starting image search: 'None'
# INFO:SocraticGenProductSeeker:✅ Search completed in 0.00s
# INFO:SocraticGenProductSeeker:🔌 Connecting to database at: D:/Vector/ProductSeeker_data
# WARNING:SocraticGenProductSeeker:Voice initialization failed: No Default Input Device Available
# INFO:SocraticGenProductSeeker:🤖 All agents initialized successfully
# INFO:SocraticGenProductSeeker:🤖 Advanced Agentic Product Search System initialized
# INFO:SocraticGenProductSeeker:📁 Database path: D:/Vector/ProductSeeker_data
# INFO:SocraticGenProductSeeker:📊 Collection: ecommerce_test
# INFO:SocraticGenProductSeeker:🧠 Model: clip-ViT-B-32
# INFO:SocraticGenProductSeeker:🔌 Connecting to database at: D:/Vector/ProductSeeker_data
# WARNING:SocraticGenProductSeeker:Voice initialization failed: No Default Input Device Available
# INFO:SocraticGenProductSeeker:🤖 All agents initialized successfully
# INFO:SocraticGenProductSeeker:🤖 Advanced Agentic Product Search System initialized
# INFO:SocraticGenProductSeeker:📁 Database path: D:/Vector/ProductSeeker_data
# INFO:SocraticGenProductSeeker:📊 Collection: ecommerce_test
# INFO:SocraticGenProductSeeker:🧠 Model: clip-ViT-B-32
# INFO:SocraticGenProductSeeker:🔌 Connecting to database at: D:/Vector/ProductSeeker_data
# WARNING:SocraticGenProductSeeker:Voice initialization failed: No Default Input Device Available
# INFO:SocraticGenProductSeeker:🤖 All agents initialized successfully
# INFO:SocraticGenProductSeeker:🤖 Advanced Agentic Product Search System initialized
# INFO:SocraticGenProductSeeker:📁 Database path: D:/Vector/ProductSeeker_data
# INFO:SocraticGenProductSeeker:📊 Collection: ecommerce_test
# INFO:SocraticGenProductSeeker:🧠 Model: clip-ViT-B-32
# INFO:SocraticGenProductSeeker:🚀 Starting text search: 'gaming laptop'
# INFO:SocraticGenProductSeeker:✅ Search completed in 0.00s
# INFO:SocraticGenProductSeeker:🔌 Connecting to database at: D:/Vector/ProductSeeker_data
# WARNING:SocraticGenProductSeeker:Voice initialization failed: No Default Input Device Available
# INFO:SocraticGenProductSeeker:🤖 All agents initialized successfully
# INFO:SocraticGenProductSeeker:🤖 Advanced Agentic Product Search System initialized
# INFO:SocraticGenProductSeeker:📁 Database path: D:/Vector/ProductSeeker_data
# INFO:SocraticGenProductSeeker:📊 Collection: ecommerce_test
# INFO:SocraticGenProductSeeker:🧠 Model: clip-ViT-B-32
# INFO:SocraticGenProductSeeker:🔌 Connecting to database at: D:/Vector/ProductSeeker_data
# WARNING:SocraticGenProductSeeker:Voice initialization failed: No Default Input Device Available
# INFO:SocraticGenProductSeeker:🤖 All agents initialized successfully
# INFO:SocraticGenProductSeeker:🤖 Advanced Agentic Product Search System initialized
# INFO:SocraticGenProductSeeker:📁 Database path: D:/Vector/ProductSeeker_data
# INFO:SocraticGenProductSeeker:📊 Collection: ecommerce_test
# INFO:SocraticGenProductSeeker:🧠 Model: clip-ViT-B-32
# WARNING:SocraticGenProductSeeker:Voice initialization failed: No Default Input Device Available
# WARNING:SocraticGenProductSeeker:Voice initialization failed: No Default Input Device Available
# INFO:SocraticGenProductSeeker:🔌 Connecting to database at: D:/Vector/ProductSeeker_data
# WARNING:SocraticGenProductSeeker:Voice initialization failed: No Default Input Device Available
# INFO:SocraticGenProductSeeker:🤖 All agents initialized successfully
# 🤖 Advanced Agentic Product Search System
# ==================================================
# INFO:SocraticGenProductSeeker:🤖 Advanced Agentic Product Search System initialized
# INFO:SocraticGenProductSeeker:📁 Database path: D:/Vector/ProductSeeker_data
# INFO:SocraticGenProductSeeker:📊 Collection: ecommerce_test
# INFO:SocraticGenProductSeeker:🧠 Model: clip-ViT-B-32
# INFO:SocraticGenProductSeeker:🔌 Connecting to database at: D:/Vector/ProductSeeker_data
# 🔄 Initializing system...
# WARNING:SocraticGenProductSeeker:Voice initialization failed: No Default Input Device Available
# INFO:SocraticGenProductSeeker:🤖 All agents initialized successfully
# INFO:SocraticGenProductSeeker:🤖 Advanced Agentic Product Search System initialized
# INFO:SocraticGenProductSeeker:📁 Database path: D:/Vector/ProductSeeker_data
# INFO:SocraticGenProductSeeker:📊 Collection: ecommerce_test
# INFO:SocraticGenProductSeeker:🧠 Model: clip-ViT-B-32
# INFO:SocraticGenProductSeeker:🔌 Connecting to database at: D:/Vector/ProductSeeker_data
# ✅ System ready!
# 📊 Database: ✅
# 🎤 Voice: ❌
# 📸 Image: ✅
#
# WARNING:SocraticGenProductSeeker:Voice initialization failed: No Default Input Device Available
# INFO:SocraticGenProductSeeker:🤖 All agents initialized successfully
# INFO:SocraticGenProductSeeker:🤖 Advanced Agentic Product Search System initialized
# INFO:SocraticGenProductSeeker:📁 Database path: D:/Vector/ProductSeeker_data
# INFO:SocraticGenProductSeeker:📊 Collection: ecommerce_test
# INFO:SocraticGenProductSeeker:🧠 Model: clip-ViT-B-32
# INFO:SocraticGenProductSeeker:🚀 Starting text search: 'test'
# INFO:SocraticGenProductSeeker:✅ Search completed in 0.00s
# INFO:SocraticGenProductSeeker:🔌 Connecting to database at: D:/Vector/ProductSeeker_data
# WARNING:SocraticGenProductSeeker:Voice initialization failed: No Default Input Device Available
# INFO:SocraticGenProductSeeker:🤖 All agents initialized successfully
# INFO:SocraticGenProductSeeker:🤖 Advanced Agentic Product Search System initialized
# INFO:SocraticGenProductSeeker:📁 Database path: D:/Vector/ProductSeeker_data
# INFO:SocraticGenProductSeeker:📊 Collection: ecommerce_test
# INFO:SocraticGenProductSeeker:🧠 Model: clip-ViT-B-32
# INFO:SocraticGenProductSeeker:🚀 Starting text search: 'test'
# ERROR:SocraticGenProductSeeker:Text search failed: Database error
# INFO:SocraticGenProductSeeker:✅ Search completed in 0.00s
# INFO:SocraticGenProductSeeker:🔌 Connecting to database at: D:/Vector/ProductSeeker_data
# WARNING:SocraticGenProductSeeker:Voice initialization failed: No Default Input Device Available
# INFO:SocraticGenProductSeeker:🤖 All agents initialized successfully
# INFO:SocraticGenProductSeeker:🤖 Advanced Agentic Product Search System initialized
# INFO:SocraticGenProductSeeker:📁 Database path: D:/Vector/ProductSeeker_data
# INFO:SocraticGenProductSeeker:📊 Collection: ecommerce_test
# INFO:SocraticGenProductSeeker:🧠 Model: clip-ViT-B-32
# INFO:SocraticGenProductSeeker:🔌 Connecting to database at: D:/Vector/ProductSeeker_data
# WARNING:SocraticGenProductSeeker:Voice initialization failed: No Default Input Device Available
# INFO:SocraticGenProductSeeker:🤖 All agents initialized successfully
# INFO:SocraticGenProductSeeker:🤖 Advanced Agentic Product Search System initialized
# INFO:SocraticGenProductSeeker:📁 Database path: D:/Vector/ProductSeeker_data
# INFO:SocraticGenProductSeeker:📊 Collection: ecommerce_test
# INFO:SocraticGenProductSeeker:🧠 Model: clip-ViT-B-32
# INFO:SocraticGenProductSeeker:🚀 Starting text search: 'laptop'
# INFO:SocraticGenProductSeeker:✅ Search completed in 0.00s
# INFO:SocraticGenProductSeeker:🚀 Starting text search: 'phone'
# INFO:SocraticGenProductSeeker:✅ Search completed in 0.00s
# INFO:SocraticGenProductSeeker:🚀 Starting text search: 'headphones'
# INFO:SocraticGenProductSeeker:✅ Search completed in 0.00s
# INFO:SocraticGenProductSeeker:🚀 Starting text search: 'mouse'
# INFO:SocraticGenProductSeeker:✅ Search completed in 0.00s
# INFO:SocraticGenProductSeeker:🚀 Starting text search: 'keyboard'
# INFO:SocraticGenProductSeeker:✅ Search completed in 0.00s
# INFO:SocraticGenProductSeeker:🔌 Connecting to database at: D:/Vector/ProductSeeker_data
# WARNING:SocraticGenProductSeeker:Voice initialization failed: No Default Input Device Available
# INFO:SocraticGenProductSeeker:🤖 All agents initialized successfully
# INFO:SocraticGenProductSeeker:🤖 Advanced Agentic Product Search System initialized
# INFO:SocraticGenProductSeeker:📁 Database path: D:/Vector/ProductSeeker_data
# INFO:SocraticGenProductSeeker:📊 Collection: ecommerce_test
# INFO:SocraticGenProductSeeker:🧠 Model: clip-ViT-B-32
# INFO:SocraticGenProductSeeker:🚀 Starting text search: 'test'
# INFO:SocraticGenProductSeeker:✅ Search completed in 0.00s
#
#
# Ran 32 tests in 0.988s
#
# OK
