#!/usr/bin/env python3
"""
Comprehensive Test Suite for Product Search System Launcher
Tests all components: scraper, database, LangGraph system, and image bot
"""

import unittest
import logging
import sys
import os
import tempfile
import shutil
import subprocess
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

"""Test Coverage
1. Unit Tests

Environment Setup Tests: Verify dependency checking and module imports
Scraper Functionality: Test successful scraping, failures, and exception handling
LangGraph System: Test search functionality, query processing, and error handling
Image Bot Tests: Test both Streamlit and console interfaces
Database Operations: Test status checking, empty database handling, and connection errors
Main Function: Test command-line argument processing and workflow orchestration

2. Integration Tests

Full Setup Workflow: Test the complete full-setup command sequence
Component Interaction: Verify that all modules work together correctly
Error Propagation: Test how errors cascade through the system

3. Error Handling Tests

Invalid Commands: Test argparse error handling
Empty Database Scenarios: Test behavior when database is not ready
Exception Recovery: Test graceful handling of various failure modes

Key Features
Mocking Strategy

Uses unittest.mock to isolate components and avoid external dependencies
Mocks database operations, file I/O, and network calls
Allows testing without actually running scrapers or databases

Comprehensive Reporting

Detailed test results with success rates
Separate integration and unit test reporting
Manual testing checklist for final verification

Realistic Test Scenarios

Tests actual command-line usage patterns
Simulates real-world failure conditions
Validates configuration and setup requirements

Usage
Run the test suite with:
bashpython test_launcher.py
The script will:

✅ Run integration tests to verify component availability
🧪 Execute comprehensive unit tests
📊 Generate detailed test reports
📋 Provide a manual testing checklist

Test Structure
The tests are organized into logical groups:

TestEnvironmentSetup - Environment and dependency tests
TestScraperFunctionality - Web scraping tests
TestLangGraphSystem - AI search system tests
TestImageBot - Image search interface tests
TestDatabaseOperations - Database interaction tests
TestMainFunction - Command-line interface tests
TestIntegrationScenarios - End-to-end workflow tests
TestErrorHandling - Error condition and edge case tests"""


class TestProductSearchSystemLauncher(unittest.TestCase):
    """Main test class for the Product Search System"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = tempfile.mkdtemp()
        cls.test_db_path = os.path.join(cls.test_dir, "test_db")
        cls.test_scraper_output = os.path.join(cls.test_dir, "test_scraper")
        cls.test_collection = "test_collection"
        cls.test_model = "clip-ViT-B-32"

        # Create test directories
        os.makedirs(cls.test_db_path, exist_ok=True)
        os.makedirs(cls.test_scraper_output, exist_ok=True)

        print(f"Test environment created at: {cls.test_dir}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        try:
            shutil.rmtree(cls.test_dir)
            print("Test environment cleaned up")
        except Exception as e:
            print(f"Warning: Failed to clean up test directory: {e}")

    def setUp(self):
        """Set up each test"""
        self.launcher_path = "Launcher.py"
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.captured_output = StringIO()

    def tearDown(self):
        """Clean up after each test"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr


class TestEnvironmentSetup(TestProductSearchSystemLauncher):
    """Test environment setup and dependencies"""

    @patch('Launcher.logger')
    def test_setup_environment_success(self, mock_logger):
        """Test successful environment setup"""
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = Mock()

            # Import the function
            sys.path.insert(0, '.')
            from Launcher import setup_environment

            result = setup_environment()
            self.assertTrue(result)

    @patch('Launcher.logger')
    def test_setup_environment_missing_modules(self, mock_logger):
        """Test environment setup with missing modules"""

        def mock_import_side_effect(name):
            if name in ['Integrater', 'Vector']:
                raise ImportError(f"No module named '{name}'")
            return Mock()

        with patch('builtins.__import__', side_effect=mock_import_side_effect):
            sys.path.insert(0, '.')
            from Launcher import setup_environment

            result = setup_environment()
            self.assertFalse(result)
            mock_logger.error.assert_called()


class TestScraperFunctionality(TestProductSearchSystemLauncher):
    """Test scraper functionality"""

    @patch('Launcher.IntegratedProductScraper')
    @patch('Launcher.logger')
    def test_run_scraper_success(self, mock_logger, mock_scraper_class):
        """Test successful scraper run"""
        # Mock scraper instance
        mock_scraper = Mock()
        mock_scraper.scrape_and_store.return_value = {
            'status': 'completed',
            'stored_products': 100
        }
        mock_scraper_class.return_value = mock_scraper

        sys.path.insert(0, '.')
        from Launcher import run_scraper

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = run_scraper()

            self.assertTrue(result)
            mock_scraper.scrape_and_store.assert_called_once()
            output = mock_stdout.getvalue()
            self.assertIn("Successfully scraped", output)

    @patch('Launcher.IntegratedProductScraper')
    @patch('Launcher.logger')
    def test_run_scraper_failure(self, mock_logger, mock_scraper_class):
        """Test scraper failure handling"""
        mock_scraper = Mock()
        mock_scraper.scrape_and_store.return_value = {
            'status': 'failed',
            'error': 'Connection timeout'
        }
        mock_scraper_class.return_value = mock_scraper

        sys.path.insert(0, '.')
        from Launcher import run_scraper

        with patch('sys.stdout', new_callable=StringIO):
            result = run_scraper()

            self.assertFalse(result)

    @patch('Launcher.IntegratedProductScraper')
    @patch('Launcher.logger')
    def test_run_scraper_exception(self, mock_logger, mock_scraper_class):
        """Test scraper exception handling"""
        mock_scraper_class.side_effect = Exception("Import error")

        sys.path.insert(0, '.')
        from Launcher import run_scraper

        result = run_scraper()
        self.assertFalse(result)
        mock_logger.error.assert_called()


class TestLangGraphSystem(TestProductSearchSystemLauncher):
    """Test LangGraph system functionality"""

    @patch('Launcher.LangGraphProductSearcher')
    @patch('Launcher.logger')
    def test_run_langgraph_system_success(self, mock_logger, mock_searcher_class):
        """Test successful LangGraph system run"""
        # Mock searcher instance
        mock_searcher = Mock()
        mock_message = Mock()
        mock_message.content = "Found some great gaming laptops for you!"

        mock_searcher.search.return_value = {
            'success': True,
            'results': ['result1', 'result2', 'result3'],
            'refinement_count': 2,
            'messages': [mock_message]
        }
        mock_searcher_class.return_value = mock_searcher

        sys.path.insert(0, '.')
        from Launcher import run_langgraph_system

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = run_langgraph_system()

            self.assertTrue(result)
            # Should be called 3 times for 3 test queries
            self.assertEqual(mock_searcher.search.call_count, 3)
            output = mock_stdout.getvalue()
            self.assertIn("Found 3 results", output)

    @patch('Launcher.LangGraphProductSearcher')
    @patch('Launcher.logger')
    def test_run_langgraph_system_search_failure(self, mock_logger, mock_searcher_class):
        """Test LangGraph system with search failures"""
        mock_searcher = Mock()
        mock_searcher.search.return_value = {
            'success': False,
            'error': 'Search timeout'
        }
        mock_searcher_class.return_value = mock_searcher

        sys.path.insert(0, '.')
        from Launcher import run_langgraph_system

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = run_langgraph_system()

            self.assertTrue(result)  # Function still returns True even if individual searches fail
            output = mock_stdout.getvalue()
            self.assertIn("Search failed", output)


class TestImageBot(TestProductSearchSystemLauncher):
    """Test image bot functionality"""

    @patch('Launcher.ImageSearchBot')
    @patch('Launcher.logger')
    def test_run_image_bot_streamlit(self, mock_logger, mock_bot_class):
        """Test image bot with Streamlit interface"""
        mock_bot = Mock()
        mock_bot.run_streamlit_app.return_value = None
        mock_bot_class.return_value = mock_bot

        sys.path.insert(0, '.')
        from Launcher import run_image_bot

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = run_image_bot("streamlit")

            self.assertTrue(result)
            mock_bot.run_streamlit_app.assert_called_once()
            output = mock_stdout.getvalue()
            self.assertIn("Starting Streamlit", output)

    @patch('Launcher.ImageSearchBot')
    @patch('Launcher.logger')
    def test_run_image_bot_console(self, mock_logger, mock_bot_class):
        """Test image bot with console interface"""
        mock_bot = Mock()
        mock_bot.run_console_interface.return_value = None
        mock_bot_class.return_value = mock_bot

        sys.path.insert(0, '.')
        from Launcher import run_image_bot

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = run_image_bot("console")

            self.assertTrue(result)
            mock_bot.run_console_interface.assert_called_once()
            output = mock_stdout.getvalue()
            self.assertIn("Starting console", output)

    @patch('Launcher.ImageSearchBot')
    @patch('Launcher.logger')
    def test_run_image_bot_exception(self, mock_logger, mock_bot_class):
        """Test image bot exception handling"""
        mock_bot_class.side_effect = Exception("Bot initialization failed")

        sys.path.insert(0, '.')
        from Launcher import run_image_bot

        result = run_image_bot("streamlit")
        self.assertFalse(result)
        mock_logger.error.assert_called()


class TestDatabaseOperations(TestProductSearchSystemLauncher):
    """Test database operations"""

    @patch('Launcher.ProductSeekerVectorDB')
    @patch('Launcher.logger')
    def test_check_database_status_with_data(self, mock_logger, mock_db_class):
        """Test database status check with data"""
        mock_db = Mock()
        mock_db.get_database_stats.return_value = {
            'total_products': 150,
            'products_with_images': 120
        }
        mock_db_class.return_value = mock_db

        sys.path.insert(0, '.')
        from Launcher import check_database_status

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = check_database_status()

            self.assertTrue(result)
            output = mock_stdout.getvalue()
            self.assertIn("Total Products: 150", output)
            self.assertIn("Database is ready", output)

    @patch('Launcher.ProductSeekerVectorDB')
    @patch('Launcher.logger')
    def test_check_database_status_empty(self, mock_logger, mock_db_class):
        """Test database status check with empty database"""
        mock_db = Mock()
        mock_db.get_database_stats.return_value = {
            'total_products': 0,
            'products_with_images': 0
        }
        mock_db_class.return_value = mock_db

        sys.path.insert(0, '.')
        from Launcher import check_database_status

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = check_database_status()

            self.assertFalse(result)
            output = mock_stdout.getvalue()
            self.assertIn("Database is empty", output)

    @patch('Launcher.ProductSeekerVectorDB')
    @patch('Launcher.logger')
    def test_check_database_status_exception(self, mock_logger, mock_db_class):
        """Test database status check exception handling"""
        mock_db_class.side_effect = Exception("Database connection failed")

        sys.path.insert(0, '.')
        from Launcher import check_database_status

        result = check_database_status()
        self.assertFalse(result)
        mock_logger.error.assert_called()


class TestMainFunction(TestProductSearchSystemLauncher):
    """Test main function and command line interface"""

    @patch('sys.argv', ['Launcher.py', 'status'])
    @patch('Launcher.check_database_status')
    @patch('Launcher.setup_environment')
    def test_main_status_command(self, mock_setup, mock_check_db):
        """Test main function with status command"""
        mock_setup.return_value = True
        mock_check_db.return_value = True

        sys.path.insert(0, '.')
        from Launcher import main

        with patch('sys.stdout', new_callable=StringIO):
            main()

            mock_setup.assert_called_once()
            mock_check_db.assert_called_once()

    @patch('sys.argv', ['Launcher.py', 'scrape'])
    @patch('Launcher.run_scraper')
    @patch('Launcher.setup_environment')
    def test_main_scrape_command(self, mock_setup, mock_scraper):
        """Test main function with scrape command"""
        mock_setup.return_value = True
        mock_scraper.return_value = True

        sys.path.insert(0, '.')
        from Launcher import main

        with patch('sys.stdout', new_callable=StringIO):
            main()

            mock_setup.assert_called_once()
            mock_scraper.assert_called_once()

    @patch('sys.argv', ['Launcher.py', 'langgraph'])
    @patch('Launcher.run_langgraph_system')
    @patch('Launcher.check_database_status')
    @patch('Launcher.setup_environment')
    def test_main_langgraph_command(self, mock_setup, mock_check_db, mock_langgraph):
        """Test main function with langgraph command"""
        mock_setup.return_value = True
        mock_check_db.return_value = True
        mock_langgraph.return_value = True

        sys.path.insert(0, '.')
        from Launcher import main

        with patch('sys.stdout', new_callable=StringIO):
            main()

            mock_setup.assert_called_once()
            mock_check_db.assert_called_once()
            mock_langgraph.assert_called_once()


class TestIntegrationScenarios(TestProductSearchSystemLauncher):
    """Integration tests for complete workflows"""

    @patch('Launcher.run_scraper')
    @patch('Launcher.run_langgraph_system')
    @patch('Launcher.check_database_status')
    @patch('Launcher.setup_environment')
    def test_full_setup_workflow(self, mock_setup, mock_check_db, mock_langgraph, mock_scraper):
        """Test complete full-setup workflow"""
        mock_setup.return_value = True
        mock_scraper.return_value = True
        mock_langgraph.return_value = True
        mock_check_db.return_value = True

        sys.path.insert(0, '.')

        # Mock sys.argv for full-setup command
        with patch('sys.argv', ['Launcher.py', 'full-setup']):
            from Launcher import main

            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                main()

                output = mock_stdout.getvalue()
                self.assertIn("Complete setup finished successfully", output)

                # Verify all steps were called
                mock_scraper.assert_called_once()
                mock_langgraph.assert_called_once()
                mock_check_db.assert_called()


class TestErrorHandling(TestProductSearchSystemLauncher):
    """Test error handling and edge cases"""

    @patch('sys.argv', ['Launcher.py', 'bot'])
    @patch('Launcher.check_database_status')
    @patch('Launcher.setup_environment')
    def test_bot_command_empty_database(self, mock_setup, mock_check_db):
        """Test bot command with empty database"""
        mock_setup.return_value = True
        mock_check_db.return_value = False  # Empty database

        sys.path.insert(0, '.')
        from Launcher import main

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with self.assertRaises(SystemExit) as context:
                main()

            self.assertEqual(context.exception.code, 1)
            output = mock_stdout.getvalue()
            self.assertIn("Database not ready", output)

    @patch('sys.argv', ['Launcher.py', 'invalid_command'])
    def test_invalid_command(self):
        """Test handling of invalid commands"""
        sys.path.insert(0, '.')

        # This should raise SystemExit due to argparse error
        with self.assertRaises(SystemExit):
            from Launcher import main
            main()


def create_test_report():
    """Create a comprehensive test report"""
    print("\n" + "=" * 80)
    print("PRODUCT SEARCH SYSTEM - TEST REPORT")
    print("=" * 80)

    # Run the test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    # Custom test runner with detailed output
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)

    # Print results
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")

    print("\nDETAILED OUTPUT:")
    print(stream.getvalue())

    return result.wasSuccessful()


def run_integration_test():
    """Run a simulated integration test"""
    print("\n" + "=" * 60)
    print("INTEGRATION TEST - SIMULATED FULL WORKFLOW")
    print("=" * 60)

    try:
        # Test if we can import the launcher
        print("✓ Testing launcher import...")
        sys.path.insert(0, '.')
        import Launcher
        print("✓ Launcher imported successfully")

        # Test configuration constants
        print("✓ Testing configuration...")
        assert hasattr(Launcher, 'SCRAPER_OUTPUT')
        assert hasattr(Launcher, 'DATABASE_PATH')
        assert hasattr(Launcher, 'COLLECTION_NAME')
        print("✓ Configuration constants found")

        # Test function availability
        functions_to_test = [
            'setup_environment',
            'run_scraper',
            'run_langgraph_system',
            'run_image_bot',
            'check_database_status',
            'main'
        ]

        for func_name in functions_to_test:
            if hasattr(Launcher, func_name):
                print(f"✓ Function '{func_name}' available")
            else:
                print(f"✗ Function '{func_name}' missing")
                return False

        print("\n✅ Integration test passed - All components available")
        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 PRODUCT SEARCH SYSTEM TEST SUITE")
    print("=" * 80)

    # Run integration test first
    integration_success = run_integration_test()

    # Run unit tests
    unit_test_success = create_test_report()

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)
    print(f"Integration Tests: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    print(f"Unit Tests: {'✅ PASSED' if unit_test_success else '❌ FAILED'}")

    if integration_success and unit_test_success:
        print("\n🎉 ALL TESTS PASSED - System ready for deployment!")
        exit_code = 0
    else:
        print("\n⚠️  SOME TESTS FAILED - Please review and fix issues")
        exit_code = 1

    print("\n📋 MANUAL TESTING CHECKLIST:")
    print("□ Run 'python Launcher.py status' to check database")
    print("□ Run 'python Launcher.py scrape' to test scraping")
    print("□ Run 'python Launcher.py langgraph' to test search system")
    print("□ Run 'python Launcher.py bot' to test web interface")
    print("□ Run 'python Launcher.py full-setup' for complete workflow")

    sys.exit(exit_code)


# C:\Users\themi\AppData\Local\Programs\Python\Python313\python.exe "C:/Program Files/JetBrains/PyCharm Community Edition 2024.1.2/plugins/python-ce/helpers/pycharm/_jb_unittest_runner.py" --path C:\Users\themi\PycharmProjects\ProductSeeker\tst.py
# Testing started at 11:38 AM ...
# Launching unittests with arguments python -m unittest C:\Users\themi\PycharmProjects\ProductSeeker\tst.py in C:\Users\themi\PycharmProjects\ProductSeeker
#
# Test environment created at: C:\Users\themi\AppData\Local\Temp\tmpqt26z4a6
# Test environment cleaned up
# Test environment created at: C:\Users\themi\AppData\Local\Temp\tmpaa9c7kt8
#
# Error
# Traceback (most recent call last):
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1421, in patched
#     with self.decoration_helper(patched,
#          ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
#                                 args,
#                                 ^^^^^
#                                 keywargs) as (newargs, newkeywargs):
#                                 ^^^^^^^^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\contextlib.py", line 141, in __enter__
#     return next(self.gen)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1403, in decoration_helper
#     arg = exit_stack.enter_context(patching)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\contextlib.py", line 530, in enter_context
#     result = _enter(cm)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1495, in __enter__
#     original, local = self.get_original()
#                       ~~~~~~~~~~~~~~~~~^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1465, in get_original
#     raise AttributeError(
#         "%s does not have the attribute %r" % (target, name)
#     )
# AttributeError: <module 'Launcher' from 'C:\\Users\\themi\\PycharmProjects\\ProductSeeker\\Launcher.py'> does not have the attribute 'ProductSeekerVectorDB'
#
#
# Error
# Traceback (most recent call last):
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1421, in patched
#     with self.decoration_helper(patched,
#          ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
#                                 args,
#                                 ^^^^^
#                                 keywargs) as (newargs, newkeywargs):
#                                 ^^^^^^^^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\contextlib.py", line 141, in __enter__
#     return next(self.gen)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1403, in decoration_helper
#     arg = exit_stack.enter_context(patching)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\contextlib.py", line 530, in enter_context
#     result = _enter(cm)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1495, in __enter__
#     original, local = self.get_original()
#                       ~~~~~~~~~~~~~~~~~^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1465, in get_original
#     raise AttributeError(
#         "%s does not have the attribute %r" % (target, name)
#     )
# AttributeError: <module 'Launcher' from 'C:\\Users\\themi\\PycharmProjects\\ProductSeeker\\Launcher.py'> does not have the attribute 'ProductSeekerVectorDB'
#
#
# Error
# Traceback (most recent call last):
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1421, in patched
#     with self.decoration_helper(patched,
#          ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
#                                 args,
#                                 ^^^^^
#                                 keywargs) as (newargs, newkeywargs):
#                                 ^^^^^^^^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\contextlib.py", line 141, in __enter__
#     return next(self.gen)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1403, in decoration_helper
#     arg = exit_stack.enter_context(patching)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\contextlib.py", line 530, in enter_context
#     result = _enter(cm)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1495, in __enter__
#     original, local = self.get_original()
#                       ~~~~~~~~~~~~~~~~~^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1465, in get_original
#     raise AttributeError(
#         "%s does not have the attribute %r" % (target, name)
#     )
# AttributeError: <module 'Launcher' from 'C:\\Users\\themi\\PycharmProjects\\ProductSeeker\\Launcher.py'> does not have the attribute 'ProductSeekerVectorDB'
#
# Test environment cleaned up
# Test environment created at: C:\Users\themi\AppData\Local\Temp\tmpw5kvdfow
#
# Error
# Traceback (most recent call last):
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1424, in patched
#     return func(*newargs, **newkeywargs)
#   File "C:\Users\themi\PycharmProjects\ProductSeeker\tst.py", line 158, in test_setup_environment_missing_modules
#     from Launcher import setup_environment
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1167, in __call__
#     return self._mock_call(*args, **kwargs)
#            ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1171, in _mock_call
#     return self._execute_mock_call(*args, **kwargs)
#            ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1232, in _execute_mock_call
#     result = effect(*args, **kwargs)
# TypeError: TestEnvironmentSetup.test_setup_environment_missing_modules.<locals>.mock_import_side_effect() takes 1 positional argument but 5 were given
#
# Test environment cleaned up
# Test environment created at: C:\Users\themi\AppData\Local\Temp\tmpbk2nejru
# usage: Launcher.py [-h] [--skip-checks]
#                    {scrape,langgraph,bot,console-bot,status,full-setup}
# Launcher.py: error: argument command: invalid choice: 'invalid_command' (choose from scrape, langgraph, bot, console-bot, status, full-setup)
# Test environment cleaned up
# Test environment created at: C:\Users\themi\AppData\Local\Temp\tmpac85t6on
#
# Error
# Traceback (most recent call last):
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1421, in patched
#     with self.decoration_helper(patched,
#          ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
#                                 args,
#                                 ^^^^^
#                                 keywargs) as (newargs, newkeywargs):
#                                 ^^^^^^^^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\contextlib.py", line 141, in __enter__
#     return next(self.gen)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1403, in decoration_helper
#     arg = exit_stack.enter_context(patching)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\contextlib.py", line 530, in enter_context
#     result = _enter(cm)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1495, in __enter__
#     original, local = self.get_original()
#                       ~~~~~~~~~~~~~~~~~^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1465, in get_original
#     raise AttributeError(
#         "%s does not have the attribute %r" % (target, name)
#     )
# AttributeError: <module 'Launcher' from 'C:\\Users\\themi\\PycharmProjects\\ProductSeeker\\Launcher.py'> does not have the attribute 'ImageSearchBot'
#
#
# Error
# Traceback (most recent call last):
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1421, in patched
#     with self.decoration_helper(patched,
#          ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
#                                 args,
#                                 ^^^^^
#                                 keywargs) as (newargs, newkeywargs):
#                                 ^^^^^^^^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\contextlib.py", line 141, in __enter__
#     return next(self.gen)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1403, in decoration_helper
#     arg = exit_stack.enter_context(patching)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\contextlib.py", line 530, in enter_context
#     result = _enter(cm)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1495, in __enter__
#     original, local = self.get_original()
#                       ~~~~~~~~~~~~~~~~~^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1465, in get_original
#     raise AttributeError(
#         "%s does not have the attribute %r" % (target, name)
#     )
# AttributeError: <module 'Launcher' from 'C:\\Users\\themi\\PycharmProjects\\ProductSeeker\\Launcher.py'> does not have the attribute 'ImageSearchBot'
#
#
# Error
# Traceback (most recent call last):
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1421, in patched
#     with self.decoration_helper(patched,
#          ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
#                                 args,
#                                 ^^^^^
#                                 keywargs) as (newargs, newkeywargs):
#                                 ^^^^^^^^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\contextlib.py", line 141, in __enter__
#     return next(self.gen)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1403, in decoration_helper
#     arg = exit_stack.enter_context(patching)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\contextlib.py", line 530, in enter_context
#     result = _enter(cm)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1495, in __enter__
#     original, local = self.get_original()
#                       ~~~~~~~~~~~~~~~~~^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1465, in get_original
#     raise AttributeError(
#         "%s does not have the attribute %r" % (target, name)
#     )
# AttributeError: <module 'Launcher' from 'C:\\Users\\themi\\PycharmProjects\\ProductSeeker\\Launcher.py'> does not have the attribute 'ImageSearchBot'
#
# Test environment cleaned up
# Test environment created at: C:\Users\themi\AppData\Local\Temp\tmpt8v54blb
# Test environment cleaned up
# Test environment created at: C:\Users\themi\AppData\Local\Temp\tmpm8s5twl3
# Test environment cleaned up
# Test environment created at: C:\Users\themi\AppData\Local\Temp\tmpme6fms_9
# Test environment cleaned up
#
# Error
# Traceback (most recent call last):
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1421, in patched
#     with self.decoration_helper(patched,
#          ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
#                                 args,
#                                 ^^^^^
#                                 keywargs) as (newargs, newkeywargs):
#                                 ^^^^^^^^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\contextlib.py", line 141, in __enter__
#     return next(self.gen)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1403, in decoration_helper
#     arg = exit_stack.enter_context(patching)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\contextlib.py", line 530, in enter_context
#     result = _enter(cm)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1495, in __enter__
#     original, local = self.get_original()
#                       ~~~~~~~~~~~~~~~~~^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1465, in get_original
#     raise AttributeError(
#         "%s does not have the attribute %r" % (target, name)
#     )
# AttributeError: <module 'Launcher' from 'C:\\Users\\themi\\PycharmProjects\\ProductSeeker\\Launcher.py'> does not have the attribute 'IntegratedProductScraper'
#
#
# Error
# Traceback (most recent call last):
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1421, in patched
#     with self.decoration_helper(patched,
#          ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
#                                 args,
#                                 ^^^^^
#                                 keywargs) as (newargs, newkeywargs):
#                                 ^^^^^^^^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\contextlib.py", line 141, in __enter__
#     return next(self.gen)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1403, in decoration_helper
#     arg = exit_stack.enter_context(patching)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\contextlib.py", line 530, in enter_context
#     result = _enter(cm)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1495, in __enter__
#     original, local = self.get_original()
#                       ~~~~~~~~~~~~~~~~~^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1465, in get_original
#     raise AttributeError(
#         "%s does not have the attribute %r" % (target, name)
#     )
# AttributeError: <module 'Launcher' from 'C:\\Users\\themi\\PycharmProjects\\ProductSeeker\\Launcher.py'> does not have the attribute 'IntegratedProductScraper'
#
#
#
# Ran 19 tests in 46.474s
#
# FAILED (errors=10)
#
# Error
# Traceback (most recent call last):
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1421, in patched
#     with self.decoration_helper(patched,
#          ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
#                                 args,
#                                 ^^^^^
#                                 keywargs) as (newargs, newkeywargs):
#                                 ^^^^^^^^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\contextlib.py", line 141, in __enter__
#     return next(self.gen)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1403, in decoration_helper
#     arg = exit_stack.enter_context(patching)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\contextlib.py", line 530, in enter_context
#     result = _enter(cm)
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1495, in __enter__
#     original, local = self.get_original()
#                       ~~~~~~~~~~~~~~~~~^^
#   File "C:\Users\themi\AppData\Local\Programs\Python\Python313\Lib\unittest\mock.py", line 1465, in get_original
#     raise AttributeError(
#         "%s does not have the attribute %r" % (target, name)
#     )
# AttributeError: <module 'Launcher' from 'C:\\Users\\themi\\PycharmProjects\\ProductSeeker\\Launcher.py'> does not have the attribute 'IntegratedProductScraper'
#
#
# Process finished with exit code 1