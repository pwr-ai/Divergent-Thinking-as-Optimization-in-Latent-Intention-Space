#!/usr/bin/env python3
"""
Test script for SWE-bench scoring logic.

Tests the scoring formula and evaluation flow:
- score = f2p + (1/60) * p2p
- f2p: fail-to-pass ratio (previously failing tests that now pass)
- p2p: pass-to-pass ratio (previously passing tests that still pass)
- resolved: True if in resolved list OR (f2p == 1.0 AND p2p == 1.0)

Run: python test_scoring.py
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

# Import the scoring functions
from csc_swe_loop.scoring import score_from_instance_result
from csc_swe_loop.swebench_eval import extract_instance_result


class TestScoring(unittest.TestCase):
    """Test the score_from_instance_result function."""

    def test_perfect_score_resolved(self):
        """Test perfect score with resolved=True."""
        inst = {"resolved": True, "f2p": 1.0, "p2p": 1.0}
        score, info = score_from_instance_result(inst)
        
        expected_score = 1.0 + (1.0 / 60.0) * 1.0  # 1.0167
        self.assertAlmostEqual(score, expected_score, places=4)
        self.assertTrue(info["resolved"])
        self.assertEqual(info["f2p"], 1.0)
        self.assertEqual(info["p2p"], 1.0)

    def test_zero_score_unresolved(self):
        """Test zero score with resolved=False."""
        inst = {"resolved": False, "f2p": 0.0, "p2p": 0.0}
        score, info = score_from_instance_result(inst)
        
        expected_score = 0.0 + (1.0 / 60.0) * 0.0  # 0.0
        self.assertAlmostEqual(score, expected_score, places=4)
        self.assertFalse(info["resolved"])
        self.assertEqual(info["f2p"], 0.0)
        self.assertEqual(info["p2p"], 0.0)

    def test_partial_f2p_full_p2p(self):
        """Test partial f2p (some tests fixed) with full p2p (no regressions)."""
        inst = {"resolved": False, "f2p": 0.5, "p2p": 1.0}
        score, info = score_from_instance_result(inst)
        
        expected_score = 0.5 + (1.0 / 60.0) * 1.0  # 0.5167
        self.assertAlmostEqual(score, expected_score, places=4)
        self.assertFalse(info["resolved"])
        self.assertEqual(info["f2p"], 0.5)
        self.assertEqual(info["p2p"], 1.0)

    def test_full_f2p_partial_p2p(self):
        """Test full f2p (all tests fixed) with partial p2p (some regressions)."""
        inst = {"resolved": False, "f2p": 1.0, "p2p": 0.5}
        score, info = score_from_instance_result(inst)
        
        expected_score = 1.0 + (1.0 / 60.0) * 0.5  # 1.0083
        self.assertAlmostEqual(score, expected_score, places=4)
        # Not resolved because p2p < 1.0
        self.assertFalse(info["resolved"])
        self.assertEqual(info["f2p"], 1.0)
        self.assertEqual(info["p2p"], 0.5)

    def test_resolved_inferred_from_perfect_scores(self):
        """Test that resolved is inferred when f2p=1.0 and p2p=1.0 even if not in input."""
        inst = {"f2p": 1.0, "p2p": 1.0}  # No "resolved" key
        score, info = score_from_instance_result(inst)
        
        self.assertTrue(info["resolved"])

    def test_resolved_false_even_with_perfect_scores_if_explicit(self):
        """Test that explicit resolved=False is overridden by perfect scores."""
        inst = {"resolved": False, "f2p": 1.0, "p2p": 1.0}
        score, info = score_from_instance_result(inst)
        
        # Should be True because f2p=1.0 and p2p=1.0
        self.assertTrue(info["resolved"])

    def test_compute_from_counts(self):
        """Test computing f2p/p2p from pass/fail counts."""
        inst = {
            "fail_to_pass_passed": 8,
            "fail_to_pass_total": 10,
            "pass_to_pass_passed": 45,
            "pass_to_pass_total": 50,
        }
        score, info = score_from_instance_result(inst)
        
        expected_f2p = 8 / 10  # 0.8
        expected_p2p = 45 / 50  # 0.9
        expected_score = expected_f2p + (1.0 / 60.0) * expected_p2p
        
        self.assertAlmostEqual(info["f2p"], expected_f2p, places=4)
        self.assertAlmostEqual(info["p2p"], expected_p2p, places=4)
        self.assertAlmostEqual(score, expected_score, places=4)

    def test_alternative_key_names(self):
        """Test using alternative key names (fail_to_pass_ratio)."""
        inst = {"fail_to_pass_ratio": 0.75, "pass_to_pass_ratio": 0.95}
        score, info = score_from_instance_result(inst)
        
        self.assertEqual(info["f2p"], 0.75)
        self.assertEqual(info["p2p"], 0.95)

    def test_fallback_to_resolved_flag(self):
        """Test fallback when no f2p/p2p data available."""
        # Resolved case
        inst_resolved = {"resolved": True}
        score, info = score_from_instance_result(inst_resolved)
        self.assertEqual(info["f2p"], 1.0)
        self.assertEqual(info["p2p"], 1.0)
        
        # Unresolved case
        inst_unresolved = {"resolved": False}
        score, info = score_from_instance_result(inst_unresolved)
        self.assertEqual(info["f2p"], 0.0)
        self.assertEqual(info["p2p"], 0.0)

    def test_empty_input(self):
        """Test with empty input."""
        inst = {}
        score, info = score_from_instance_result(inst)
        
        self.assertEqual(info["f2p"], 0.0)
        self.assertEqual(info["p2p"], 0.0)
        self.assertFalse(info["resolved"])

    def test_zero_total_counts(self):
        """Test with zero total counts (division by zero protection)."""
        inst = {
            "fail_to_pass_passed": 0,
            "fail_to_pass_total": 0,  # Zero total
            "pass_to_pass_passed": 0,
            "pass_to_pass_total": 0,  # Zero total
            "resolved": False,
        }
        score, info = score_from_instance_result(inst)
        
        # Should fallback to resolved flag
        self.assertEqual(info["f2p"], 0.0)
        self.assertEqual(info["p2p"], 0.0)

    def test_score_formula_weighting(self):
        """Test that f2p is weighted much more than p2p."""
        # f2p improvement should have much larger impact than p2p
        inst_f2p_only = {"f2p": 0.5, "p2p": 0.0}
        inst_p2p_only = {"f2p": 0.0, "p2p": 0.5}
        
        score_f2p, _ = score_from_instance_result(inst_f2p_only)
        score_p2p, _ = score_from_instance_result(inst_p2p_only)
        
        # f2p improvement of 0.5 should be worth much more than p2p of 0.5
        self.assertGreater(score_f2p, score_p2p * 50)  # f2p is worth 60x more


class TestExtractInstanceResult(unittest.TestCase):
    """Test the extract_instance_result function."""

    def test_extract_from_instance_results(self):
        """Test extracting from 'instance_results' key."""
        results = {
            "instance_results": {
                "test__instance-1": {"resolved": True, "f2p": 1.0, "p2p": 1.0}
            }
        }
        result = extract_instance_result(results, "test__instance-1")
        
        self.assertIsNotNone(result)
        self.assertTrue(result["resolved"])

    def test_extract_from_results_key(self):
        """Test extracting from 'results' key."""
        results = {
            "results": {
                "test__instance-1": {"resolved": False, "f2p": 0.5, "p2p": 0.8}
            }
        }
        result = extract_instance_result(results, "test__instance-1")
        
        self.assertIsNotNone(result)
        self.assertFalse(result["resolved"])

    def test_extract_from_resolved_list(self):
        """Test building minimal result from resolved list."""
        results = {
            "resolved": ["test__instance-1", "test__instance-2"],
            "applied": ["test__instance-1", "test__instance-2", "test__instance-3"],
        }
        
        # Instance in resolved list
        result = extract_instance_result(results, "test__instance-1")
        self.assertIsNotNone(result)
        self.assertTrue(result["resolved"])
        self.assertEqual(result["f2p"], 1.0)
        self.assertEqual(result["p2p"], 1.0)
        
        # Instance not in resolved list but in applied
        result = extract_instance_result(results, "test__instance-3")
        self.assertIsNotNone(result)
        self.assertFalse(result["resolved"])
        self.assertEqual(result["f2p"], 0.0)
        self.assertEqual(result["p2p"], 0.0)

    def test_extract_not_found(self):
        """Test when instance is not found."""
        results = {"some_key": "some_value"}
        result = extract_instance_result(results, "test__instance-1")
        
        self.assertIsNone(result)

    def test_extract_error_result(self):
        """Test extracting from error result."""
        results = {"error": "Report not found", "instance_id": "test__instance-1"}
        result = extract_instance_result(results, "test__instance-1")
        
        self.assertIsNone(result)

    def test_extract_swebench_v3_format(self):
        """Test extracting from swebench v3 format (instance_id as top-level key)."""
        # This is the actual format from swebench v3.0.17
        results = {
            "sqlfluff__sqlfluff-1625": {
                "patch_is_None": False,
                "patch_exists": True,
                "patch_successfully_applied": True,
                "resolved": False,
                "tests_status": {
                    "FAIL_TO_PASS": {
                        "success": [],
                        "failure": ["test/cli/commands_test.py::test__cli__command_directed"]
                    },
                    "PASS_TO_PASS": {
                        "success": [f"test_{i}" for i in range(64)],  # 64 passing tests
                        "failure": []
                    },
                    "FAIL_TO_FAIL": {"success": [], "failure": []},
                    "PASS_TO_FAIL": {"success": [], "failure": []}
                }
            }
        }
        
        result = extract_instance_result(results, "sqlfluff__sqlfluff-1625")
        
        self.assertIsNotNone(result)
        self.assertFalse(result["resolved"])
        self.assertTrue(result["patch_applied"])
        self.assertEqual(result["f2p"], 0.0)  # 0/1 tests fixed
        self.assertEqual(result["p2p"], 1.0)  # 64/64 tests still pass
        self.assertEqual(result["fail_to_pass_passed"], 0)
        self.assertEqual(result["fail_to_pass_total"], 1)
        self.assertEqual(result["pass_to_pass_passed"], 64)
        self.assertEqual(result["pass_to_pass_total"], 64)

    def test_extract_swebench_v3_resolved(self):
        """Test extracting resolved instance from swebench v3 format."""
        results = {
            "test__instance-1": {
                "patch_successfully_applied": True,
                "resolved": True,
                "tests_status": {
                    "FAIL_TO_PASS": {
                        "success": ["test_fix_1", "test_fix_2"],  # All fixed
                        "failure": []
                    },
                    "PASS_TO_PASS": {
                        "success": ["test_pass_1", "test_pass_2", "test_pass_3"],
                        "failure": []
                    },
                    "FAIL_TO_FAIL": {"success": [], "failure": []},
                    "PASS_TO_FAIL": {"success": [], "failure": []}
                }
            }
        }
        
        result = extract_instance_result(results, "test__instance-1")
        
        self.assertIsNotNone(result)
        self.assertTrue(result["resolved"])
        self.assertEqual(result["f2p"], 1.0)  # 2/2 tests fixed
        self.assertEqual(result["p2p"], 1.0)  # 3/3 tests still pass

    def test_extract_swebench_v3_partial_fix(self):
        """Test extracting partial fix from swebench v3 format."""
        results = {
            "test__instance-1": {
                "patch_successfully_applied": True,
                "resolved": False,
                "tests_status": {
                    "FAIL_TO_PASS": {
                        "success": ["test_fix_1"],  # 1 fixed
                        "failure": ["test_fix_2", "test_fix_3"]  # 2 still failing
                    },
                    "PASS_TO_PASS": {
                        "success": ["test_pass_1", "test_pass_2"],  # 2 still pass
                        "failure": ["test_pass_3"]  # 1 regression
                    },
                    "FAIL_TO_FAIL": {"success": [], "failure": []},
                    "PASS_TO_FAIL": {"success": [], "failure": []}
                }
            }
        }
        
        result = extract_instance_result(results, "test__instance-1")
        
        self.assertIsNotNone(result)
        self.assertFalse(result["resolved"])
        self.assertAlmostEqual(result["f2p"], 1/3, places=4)  # 1/3 tests fixed
        self.assertAlmostEqual(result["p2p"], 2/3, places=4)  # 2/3 tests still pass


class TestScoreIntegration(unittest.TestCase):
    """Integration tests combining extraction and scoring."""

    def test_full_pipeline_resolved(self):
        """Test full pipeline with resolved instance."""
        # Simulate harness output
        harness_output = {
            "resolved": ["sqlfluff__sqlfluff-1625"],
            "applied": ["sqlfluff__sqlfluff-1625"],
        }
        
        # Extract result
        inst_result = extract_instance_result(harness_output, "sqlfluff__sqlfluff-1625")
        self.assertIsNotNone(inst_result)
        
        # Score it
        score, info = score_from_instance_result(inst_result)
        
        self.assertTrue(info["resolved"])
        self.assertEqual(info["f2p"], 1.0)
        self.assertEqual(info["p2p"], 1.0)
        expected_score = 1.0 + (1.0 / 60.0) * 1.0
        self.assertAlmostEqual(score, expected_score, places=4)

    def test_full_pipeline_unresolved(self):
        """Test full pipeline with unresolved instance."""
        harness_output = {
            "resolved": [],  # Empty - nothing resolved
            "applied": ["sqlfluff__sqlfluff-1625"],
        }
        
        inst_result = extract_instance_result(harness_output, "sqlfluff__sqlfluff-1625")
        self.assertIsNotNone(inst_result)
        
        score, info = score_from_instance_result(inst_result)
        
        self.assertFalse(info["resolved"])
        self.assertEqual(info["f2p"], 0.0)
        self.assertEqual(info["p2p"], 0.0)
        self.assertEqual(score, 0.0)

    def test_full_pipeline_with_detailed_results(self):
        """Test full pipeline with detailed test results."""
        harness_output = {
            "instance_results": {
                "sqlfluff__sqlfluff-1625": {
                    "resolved": False,
                    "fail_to_pass_passed": 3,
                    "fail_to_pass_total": 5,
                    "pass_to_pass_passed": 18,
                    "pass_to_pass_total": 20,
                }
            }
        }
        
        inst_result = extract_instance_result(harness_output, "sqlfluff__sqlfluff-1625")
        score, info = score_from_instance_result(inst_result)
        
        expected_f2p = 3 / 5  # 0.6
        expected_p2p = 18 / 20  # 0.9
        expected_score = expected_f2p + (1.0 / 60.0) * expected_p2p
        
        self.assertAlmostEqual(info["f2p"], expected_f2p, places=4)
        self.assertAlmostEqual(info["p2p"], expected_p2p, places=4)
        self.assertAlmostEqual(score, expected_score, places=4)


class TestScoreEdgeCases(unittest.TestCase):
    """Test edge cases in scoring."""

    def test_negative_values_clamped(self):
        """Test that negative values don't break scoring."""
        # This shouldn't happen in practice but test robustness
        inst = {"f2p": -0.1, "p2p": -0.2}
        score, info = score_from_instance_result(inst)
        
        # Should still compute (values pass through as-is currently)
        self.assertEqual(info["f2p"], -0.1)
        self.assertEqual(info["p2p"], -0.2)

    def test_values_over_one(self):
        """Test that values > 1.0 don't break scoring."""
        # This shouldn't happen but test robustness
        inst = {"f2p": 1.5, "p2p": 1.2}
        score, info = score_from_instance_result(inst)
        
        self.assertEqual(info["f2p"], 1.5)
        self.assertEqual(info["p2p"], 1.2)

    def test_string_values_converted(self):
        """Test handling of string numeric values."""
        inst = {"f2p": "0.5", "p2p": "0.8"}
        try:
            score, info = score_from_instance_result(inst)
            # If it doesn't raise, values should be converted to float
            self.assertIsInstance(info["f2p"], float)
            self.assertIsInstance(info["p2p"], float)
        except (ValueError, TypeError):
            # Expected if string conversion fails
            pass


class TestScoreComparison(unittest.TestCase):
    """Test score comparisons for CMA-ES optimization."""

    def test_better_f2p_scores_higher(self):
        """Test that better f2p always scores higher."""
        inst1 = {"f2p": 0.0, "p2p": 1.0}
        inst2 = {"f2p": 0.5, "p2p": 1.0}
        inst3 = {"f2p": 1.0, "p2p": 1.0}
        
        score1, _ = score_from_instance_result(inst1)
        score2, _ = score_from_instance_result(inst2)
        score3, _ = score_from_instance_result(inst3)
        
        self.assertLess(score1, score2)
        self.assertLess(score2, score3)

    def test_p2p_provides_tiebreaker(self):
        """Test that p2p provides meaningful differentiation."""
        inst1 = {"f2p": 0.5, "p2p": 0.0}
        inst2 = {"f2p": 0.5, "p2p": 0.5}
        inst3 = {"f2p": 0.5, "p2p": 1.0}
        
        score1, _ = score_from_instance_result(inst1)
        score2, _ = score_from_instance_result(inst2)
        score3, _ = score_from_instance_result(inst3)
        
        self.assertLess(score1, score2)
        self.assertLess(score2, score3)

    def test_f2p_dominates_p2p(self):
        """Test that f2p improvement beats p2p improvement."""
        # Small f2p improvement vs large p2p improvement
        inst_small_f2p = {"f2p": 0.1, "p2p": 0.0}
        inst_large_p2p = {"f2p": 0.0, "p2p": 1.0}
        
        score_f2p, _ = score_from_instance_result(inst_small_f2p)
        score_p2p, _ = score_from_instance_result(inst_large_p2p)
        
        # Even 0.1 f2p should beat 1.0 p2p because p2p is scaled down by 60x
        self.assertGreater(score_f2p, score_p2p)


def print_score_table():
    """Print a table of example scores for reference."""
    print("\n" + "=" * 70)
    print("SCORING REFERENCE TABLE")
    print("Formula: score = f2p + (1/60) * p2p")
    print("=" * 70)
    print(f"{'f2p':>6} {'p2p':>6} {'score':>10} {'resolved':>10}")
    print("-" * 70)
    
    test_cases = [
        {"f2p": 0.0, "p2p": 0.0, "resolved": False},
        {"f2p": 0.0, "p2p": 0.5, "resolved": False},
        {"f2p": 0.0, "p2p": 1.0, "resolved": False},
        {"f2p": 0.25, "p2p": 0.5, "resolved": False},
        {"f2p": 0.25, "p2p": 1.0, "resolved": False},
        {"f2p": 0.5, "p2p": 0.5, "resolved": False},
        {"f2p": 0.5, "p2p": 1.0, "resolved": False},
        {"f2p": 0.75, "p2p": 0.5, "resolved": False},
        {"f2p": 0.75, "p2p": 1.0, "resolved": False},
        {"f2p": 1.0, "p2p": 0.0, "resolved": False},
        {"f2p": 1.0, "p2p": 0.5, "resolved": False},
        {"f2p": 1.0, "p2p": 1.0, "resolved": True},
    ]
    
    for case in test_cases:
        score, info = score_from_instance_result(case)
        print(f"{case['f2p']:>6.2f} {case['p2p']:>6.2f} {score:>10.4f} {str(info['resolved']):>10}")
    
    print("=" * 70)


def run_mock_evaluation_flow():
    """Demonstrate the evaluation flow with mock data."""
    print("\n" + "=" * 70)
    print("MOCK EVALUATION FLOW DEMONSTRATION")
    print("=" * 70)
    
    # Simulate different scenarios
    scenarios = [
        {
            "name": "Scenario 1: All tests pass (resolved)",
            "harness_output": {
                "resolved": ["sqlfluff__sqlfluff-1625"],
                "applied": ["sqlfluff__sqlfluff-1625"],
            },
        },
        {
            "name": "Scenario 2: Patch applied but tests fail",
            "harness_output": {
                "resolved": [],
                "applied": ["sqlfluff__sqlfluff-1625"],
            },
        },
        {
            "name": "Scenario 3: Report not found (error)",
            "harness_output": {
                "error": "Report not found",
                "instance_id": "sqlfluff__sqlfluff-1625",
            },
        },
        {
            "name": "Scenario 4: Partial test pass (detailed)",
            "harness_output": {
                "instance_results": {
                    "sqlfluff__sqlfluff-1625": {
                        "resolved": False,
                        "fail_to_pass_passed": 2,
                        "fail_to_pass_total": 5,
                        "pass_to_pass_passed": 20,
                        "pass_to_pass_total": 20,
                    }
                }
            },
        },
        {
            "name": "Scenario 5: swebench v3 format (actual report)",
            "harness_output": {
                "sqlfluff__sqlfluff-1625": {
                    "patch_is_None": False,
                    "patch_exists": True,
                    "patch_successfully_applied": True,
                    "resolved": False,
                    "tests_status": {
                        "FAIL_TO_PASS": {
                            "success": [],
                            "failure": ["test/cli/commands_test.py::test__cli__command_directed"]
                        },
                        "PASS_TO_PASS": {
                            "success": [f"test_{i}" for i in range(64)],
                            "failure": []
                        },
                        "FAIL_TO_FAIL": {"success": [], "failure": []},
                        "PASS_TO_FAIL": {"success": [], "failure": []}
                    }
                }
            },
        },
        {
            "name": "Scenario 6: swebench v3 format (resolved)",
            "harness_output": {
                "sqlfluff__sqlfluff-1625": {
                    "patch_successfully_applied": True,
                    "resolved": True,
                    "tests_status": {
                        "FAIL_TO_PASS": {
                            "success": ["test/cli/commands_test.py::test__cli__command_directed"],
                            "failure": []
                        },
                        "PASS_TO_PASS": {
                            "success": [f"test_{i}" for i in range(64)],
                            "failure": []
                        },
                        "FAIL_TO_FAIL": {"success": [], "failure": []},
                        "PASS_TO_FAIL": {"success": [], "failure": []}
                    }
                }
            },
        },
    ]
    
    instance_id = "sqlfluff__sqlfluff-1625"
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print("-" * 50)
        
        # Extract result
        inst_result = extract_instance_result(scenario["harness_output"], instance_id)
        
        if inst_result is None:
            print(f"  extract_instance_result returned None")
            print(f"  Results keys: {list(scenario['harness_output'].keys())}")
            score, info = 0.0, {"resolved": False, "f2p": 0.0, "p2p": 0.0}
        else:
            score, info = score_from_instance_result(inst_result)
        
        print(f"  score={score:.3f} f2p={info['f2p']:.3f} p2p={info['p2p']:.3f} resolved={info['resolved']}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Print reference tables first
    print_score_table()
    run_mock_evaluation_flow()
    
    # Run unit tests
    print("\n" + "=" * 70)
    print("RUNNING UNIT TESTS")
    print("=" * 70 + "\n")
    
    unittest.main(verbosity=2, exit=False)
