"""
Quick test to verify backend structure is correct
Run: python test_structure.py
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing backend structure...\n")
    
    tests = []
    
    # Test config
    try:
        from app.config.settings import settings
        print("✅ Config: settings imported successfully")
        print(f"   - App Title: {settings.APP_TITLE}")
        print(f"   - Backend Port: {settings.BACKEND_PORT}")
        tests.append(True)
    except Exception as e:
        print(f"❌ Config: Failed to import settings - {e}")
        tests.append(False)
    
    # Test models
    try:
        from app.models.schemas import AnalysisRequest, ChatRequest, ChatResponse
        print("✅ Models: Pydantic schemas imported successfully")
        tests.append(True)
    except Exception as e:
        print(f"❌ Models: Failed to import schemas - {e}")
        tests.append(False)
    
    # Test utils
    try:
        from app.utils.ticker_validator import get_valid_ticker
        print("✅ Utils: ticker_validator imported successfully")
        tests.append(True)
    except Exception as e:
        print(f"❌ Utils: Failed to import ticker_validator - {e}")
        tests.append(False)
    
    # Test core modules
    try:
        from app.core import technical, fundamental, news, graph
        print("✅ Core: All analysis modules imported successfully")
        tests.append(True)
    except Exception as e:
        print(f"❌ Core: Failed to import analysis modules - {e}")
        tests.append(False)
    
    # Test routes
    try:
        from app.routes import analysis, chat, health
        print("✅ Routes: All route modules imported successfully")
        tests.append(True)
    except Exception as e:
        print(f"❌ Routes: Failed to import routes - {e}")
        tests.append(False)
    
    # Test main app
    try:
        from app.main import app
        print("✅ Main: FastAPI app imported successfully")
        print(f"   - Title: {app.title}")
        print(f"   - Version: {app.version}")
        tests.append(True)
    except Exception as e:
        print(f"❌ Main: Failed to import FastAPI app - {e}")
        tests.append(False)
    
    print("\n" + "="*50)
    passed = sum(tests)
    total = len(tests)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Backend structure is correct.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(test_imports())
