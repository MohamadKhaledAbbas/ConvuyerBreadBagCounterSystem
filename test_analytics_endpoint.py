#!/usr/bin/env python3
"""
Test Analytics Endpoint - Verify async behavior and improvements.

Tests:
1. Import validation
2. Dependency injection
3. Service methods
4. Data structure improvements (pre-grouped runs)
"""

import sys
sys.path.insert(0, '.')

from datetime import datetime, timedelta
from src.config.config_manager import get_config
from src.logging.Database import DatabaseManager
from src.endpoint.repositories.analytics_repository import AnalyticsRepository
from src.endpoint.services.analytics_service import AnalyticsService


def test_imports():
    """Test that all imports work correctly."""
    print("=" * 60)
    print("TEST 1: Import Validation")
    print("=" * 60)
    
    try:
        from src.endpoint.routes.analytics import router, get_analytics_service
        print("✓ Analytics routes imported successfully")
        print(f"  Routes: {[r.path for r in router.routes]}")
        
        from fastapi import Depends
        print("✓ FastAPI dependencies imported")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_service_creation():
    """Test service creation with dependency injection pattern."""
    print("\n" + "=" * 60)
    print("TEST 2: Service Creation")
    print("=" * 60)
    
    try:
        config = get_config()
        db = DatabaseManager(config.db_path)
        repo = AnalyticsRepository(db)
        service = AnalyticsService(repo)
        
        print("✓ Service created successfully")
        print(f"  Config: {config}")
        print(f"  Database: {db.db_path}")
        print(f"  Repository: {repo}")
        print(f"  Service: {service}")
        
        db.close()
        return True
    except Exception as e:
        print(f"✗ Service creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_structure_improvements():
    """Test that data structure includes pre-grouped runs."""
    print("\n" + "=" * 60)
    print("TEST 3: Data Structure Improvements")
    print("=" * 60)
    
    try:
        config = get_config()
        db = DatabaseManager(config.db_path)
        repo = AnalyticsRepository(db)
        service = AnalyticsService(repo)
        
        # Test with a small time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        print(f"  Querying data from {start_time} to {end_time}")
        data = service.get_analytics_data(start_time, end_time)
        
        # Verify data structure
        assert 'meta' in data, "Missing 'meta' key"
        assert 'data' in data, "Missing 'data' key"
        assert 'timeline' in data, "Missing 'timeline' key"
        print("✓ Data structure has all required keys")
        
        # Check for pre-grouped runs
        timeline = data['timeline']
        assert 'runs_by_type' in timeline, "Missing 'runs_by_type' key (pre-grouped runs)"
        print("✓ Timeline includes pre-grouped runs (runs_by_type)")
        
        runs_by_type = timeline['runs_by_type']
        print(f"  Pre-grouped runs by type: {len(runs_by_type)} types")
        for bag_type_id, runs in runs_by_type.items():
            print(f"    {bag_type_id}: {len(runs)} runs")
        
        # Verify classifications are sorted by count
        classifications = data['data']['classifications']
        if len(classifications) > 1:
            for i in range(len(classifications) - 1):
                assert classifications[i]['count'] >= classifications[i+1]['count'], \
                    "Classifications not sorted by count"
            print("✓ Classifications sorted by count (descending)")
        
        db.close()
        return True
    except Exception as e:
        print(f"✗ Data structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_datetime_parsing():
    """Test datetime parsing with various formats."""
    print("\n" + "=" * 60)
    print("TEST 4: DateTime Parsing")
    print("=" * 60)
    
    try:
        config = get_config()
        db = DatabaseManager(config.db_path)
        repo = AnalyticsRepository(db)
        service = AnalyticsService(repo)
        
        # Test various formats
        test_cases = [
            ("2024-01-15T14:30:00", True),
            ("2024-01-15 14:30:00", True),
            ("2024-01-15T14:30", True),
            ("invalid", False),
            ("2024-13-45T99:99:99", False),
        ]
        
        for test_input, should_pass in test_cases:
            try:
                result = service.parse_datetime(test_input)
                if should_pass:
                    print(f"✓ Parsed '{test_input}' -> {result}")
                else:
                    print(f"✗ Should have failed: '{test_input}'")
                    return False
            except Exception as e:
                if not should_pass:
                    print(f"✓ Correctly rejected '{test_input}'")
                else:
                    print(f"✗ Should have passed: '{test_input}' - {e}")
                    return False
        
        db.close()
        return True
    except Exception as e:
        print(f"✗ DateTime parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_group_runs_by_type():
    """Test the _group_runs_by_type helper method."""
    print("\n" + "=" * 60)
    print("TEST 5: Group Runs By Type")
    print("=" * 60)
    
    try:
        config = get_config()
        db = DatabaseManager(config.db_path)
        repo = AnalyticsRepository(db)
        service = AnalyticsService(repo)
        
        # Mock runs data
        runs = [
            {'bag_type_id': 'A', 'count': 5},
            {'bag_type_id': 'B', 'count': 3},
            {'bag_type_id': 'A', 'count': 2},
            {'bag_type_id': 'C', 'count': 1},
            {'bag_type_id': 'B', 'count': 4},
        ]
        
        grouped = service._group_runs_by_type(runs)
        
        assert 'A' in grouped, "Missing group A"
        assert 'B' in grouped, "Missing group B"
        assert 'C' in grouped, "Missing group C"
        assert len(grouped['A']) == 2, f"Expected 2 runs for A, got {len(grouped['A'])}"
        assert len(grouped['B']) == 2, f"Expected 2 runs for B, got {len(grouped['B'])}"
        assert len(grouped['C']) == 1, f"Expected 1 run for C, got {len(grouped['C'])}"
        
        print("✓ Runs grouped correctly by type")
        print(f"  Group A: {len(grouped['A'])} runs")
        print(f"  Group B: {len(grouped['B'])} runs")
        print(f"  Group C: {len(grouped['C'])} runs")
        
        db.close()
        return True
    except Exception as e:
        print(f"✗ Group runs test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ANALYTICS ENDPOINT TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_service_creation,
        test_data_structure_improvements,
        test_datetime_parsing,
        test_group_runs_by_type,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
