"""Pipeline integrity and regression tests.

Covers:
- Shuffle is invariant to results-file order
- Missing / duplicate / failed / unexpected results are detected
- allow_incomplete keeps the full denominator (and marks missing as incorrect)
- Parser handles **42**, "42 clips", \\boxed, an echoed <answer>, and inf
- Full 1000x50 batch validates
- Validator catches a changed temperature, a changed prompt, and a leaked key
- The mock experiment runs end to end at 5x3
"""

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.inference_budget.batch import (
    create_batch_request,
    generate_batch_jsonl,
    validate_batch_file,
)
from src.inference_budget.config import load_config, resolve_path
from src.inference_budget.dataset import load_processed_dataset
from src.inference_budget.evaluation import (
    IncompleteResultsError,
    check_result_integrity,
    compute_accuracy,
    parse_sheshnag_results,
    run_evaluation,
)
from src.inference_budget.parsing import extract_model_answer, normalize_answer


class TestShuffleOrderInvariance:
    """Test that seeded shuffle is 100% invariant to results-file row ordering."""

    def test_shuffle_invariant_to_file_order(self):
        config = {
            "dataset": {"sample_size": 2},
            "experiment": {
                "generations_per_question": 5,
                "inference_budgets": [1, 3, 5],
                "shuffle_seed": 42,
            },
        }
        dataset_df = pd.DataFrame([
            {"question_id": 1, "question": "Q1", "gold_answer": "42"},
            {"question_id": 2, "question": "Q2", "gold_answer": "100"},
        ])

        # Create 5 generations for Q1 and Q2 with different answers
        gens = []
        answers_q1 = ["40", "42", "42", "38", "42"]
        answers_q2 = ["90", "100", "95", "100", "100"]
        for g_id, ans in enumerate(answers_q1, 1):
            gens.append({
                "question_id": 1,
                "generation_id": g_id,
                "custom_id": f"gsm8k_q000001_g{g_id:04d}",
                "request_ok": True,
                "error": None,
                "raw_response": f"FINAL_ANSWER: {ans}",
                "parsed_answer": ans,
                "answer_extraction_method": "FINAL_ANSWER",
                "parse_success": True,
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
            })
        for g_id, ans in enumerate(answers_q2, 1):
            gens.append({
                "question_id": 2,
                "generation_id": g_id,
                "custom_id": f"gsm8k_q000002_g{g_id:04d}",
                "request_ok": True,
                "error": None,
                "raw_response": f"FINAL_ANSWER: {ans}",
                "parsed_answer": ans,
                "answer_extraction_method": "FINAL_ANSWER",
                "parse_success": True,
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
            })

        df_ordered = pd.DataFrame(gens)

        # Shuffle df rows completely (including interleaving questions and reversing generation_ids)
        df_shuffled = df_ordered.sample(frac=1.0, random_state=123).reset_index(drop=True)

        wide1, long1 = run_evaluation(df_ordered, dataset_df, config)
        wide2, long2 = run_evaluation(df_shuffled, dataset_df, config)

        pd.testing.assert_frame_equal(wide1, wide2)
        pd.testing.assert_frame_equal(long1, long2)


class TestResultIntegrityChecks:
    """Test detection of missing, duplicate, failed, or unexpected results."""

    @pytest.fixture
    def base_setup(self):
        config = {
            "dataset": {"sample_size": 2},
            "experiment": {
                "generations_per_question": 3,
                "inference_budgets": [1, 3],
                "shuffle_seed": 42,
            },
        }
        dataset_df = pd.DataFrame([
            {"question_id": 1, "question": "Q1", "gold_answer": "10"},
            {"question_id": 2, "question": "Q2", "gold_answer": "20"},
        ])
        gens = []
        for q in [1, 2]:
            for g in range(1, 4):
                gens.append({
                    "question_id": q,
                    "generation_id": g,
                    "custom_id": f"gsm8k_q{q:06d}_g{g:04d}",
                    "request_ok": True,
                    "error": None,
                    "raw_response": "FINAL_ANSWER: 10",
                    "parsed_answer": "10",
                    "answer_extraction_method": "FINAL_ANSWER",
                    "parse_success": True,
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                })
        return config, dataset_df, pd.DataFrame(gens)

    def test_missing_rows_detected(self, base_setup):
        config, dataset_df, gens_df = base_setup
        short_df = gens_df.iloc[:-1]  # Drop 1 row
        with pytest.raises(IncompleteResultsError, match="Row count mismatch|has 2 generations"):
            check_result_integrity(short_df, dataset_df, config)

    def test_duplicate_detected(self, base_setup):
        config, dataset_df, gens_df = base_setup
        dupe_df = pd.concat([gens_df, gens_df.iloc[[0]]], ignore_index=True)
        with pytest.raises(IncompleteResultsError, match="duplicate"):
            check_result_integrity(dupe_df, dataset_df, config)

    def test_failed_request_detected(self, base_setup):
        config, dataset_df, gens_df = base_setup
        gens_df.loc[0, "request_ok"] = False
        gens_df.loc[0, "error"] = "HTTP status 500"
        with pytest.raises(IncompleteResultsError, match="failed request"):
            check_result_integrity(gens_df, dataset_df, config)

    def test_unexpected_id_detected(self, base_setup):
        config, dataset_df, gens_df = base_setup
        gens_df.loc[0, "question_id"] = -1
        with pytest.raises(IncompleteResultsError, match="unexpected"):
            check_result_integrity(gens_df, dataset_df, config)

    def test_budget_exceeds_generations_raises(self, base_setup):
        config, dataset_df, gens_df = base_setup
        config["experiment"]["inference_budgets"] = [1, 5]  # 5 > generations_per_question (3)
        with pytest.raises(IncompleteResultsError, match="exceeds generations_per_question"):
            check_result_integrity(gens_df, dataset_df, config)


class TestAllowIncompleteDenominator:
    """Test allow_incomplete=True keeps full denominator and counts missing as incorrect."""

    def test_allow_incomplete_keeps_denominator(self):
        config = {
            "dataset": {"sample_size": 3},
            "experiment": {
                "generations_per_question": 2,
                "inference_budgets": [1, 2],
                "shuffle_seed": 42,
            },
        }
        dataset_df = pd.DataFrame([
            {"question_id": 1, "question": "Q1", "gold_answer": "10"},
            {"question_id": 2, "question": "Q2", "gold_answer": "20"},
            {"question_id": 3, "question": "Q3", "gold_answer": "30"},  # Missing from results
        ])
        gens = []
        for q in [1, 2]:
            for g in range(1, 3):
                gens.append({
                    "question_id": q,
                    "generation_id": g,
                    "custom_id": f"gsm8k_q{q:06d}_g{g:04d}",
                    "request_ok": True,
                    "error": None,
                    "raw_response": f"FINAL_ANSWER: {q * 10}",
                    "parsed_answer": str(q * 10),
                    "answer_extraction_method": "FINAL_ANSWER",
                    "parse_success": True,
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                })
        gens_df = pd.DataFrame(gens)

        wide_df, long_df = run_evaluation(gens_df, dataset_df, config, allow_incomplete=True)

        # Question 3 must still be in results
        assert 3 in wide_df["question_id"].values
        q3_row = wide_df[wide_df["question_id"] == 3].iloc[0]
        assert not q3_row["correct_N1"]
        assert not q3_row["correct_N2"]

        summary = compute_accuracy(long_df)
        for _, row in summary.iterrows():
            assert row["total"] == 3  # Full 3 denominator kept!
            assert row["correct"] == 2


class TestParserEdgeCases:
    """Test parser handles **42**, '42 clips', \\boxed, echoed <answer>, and inf."""

    def test_normalize_markdown_bold(self):
        assert normalize_answer("**42**") == "42"

    def test_normalize_trailing_unit_clips(self):
        assert normalize_answer("42 clips") == "42"
        assert normalize_answer("42 paper clips") == "42"

    def test_normalize_boxed(self):
        assert normalize_answer(r"\boxed{42}") == "42"
        assert normalize_answer(r"\boxed{\text{42}}") == "42"

    def test_normalize_inf_and_nan(self):
        assert normalize_answer("inf") == "inf"
        assert normalize_answer("-inf") == "-inf"
        assert normalize_answer("+inf") == "inf"
        assert normalize_answer("infinity") == "inf"
        assert normalize_answer("nan") == "nan"

    def test_extract_echoed_answer_placeholder_skipped(self):
        response = (
            "Solve the problem...\n"
            "At the very end output: FINAL_ANSWER: <answer>\n"
            "Here is my step by step work.\n"
            "FINAL_ANSWER: 42"
        )
        ans, method = extract_model_answer(response)
        assert ans == "42"
        assert method == "FINAL_ANSWER"

    def test_extract_markdown_final_answer(self):
        response1 = "Reasoning...\n**FINAL_ANSWER:** 42"
        ans1, m1 = extract_model_answer(response1)
        assert ans1 == "42"
        assert m1 == "FINAL_ANSWER"

        response2 = "Reasoning...\nFINAL_ANSWER**: 42"
        ans2, m2 = extract_model_answer(response2)
        assert ans2 == "42"
        assert m2 == "FINAL_ANSWER"

    def test_fraction_not_merged_with_decimal(self):
        assert normalize_answer("1/2") == "1/2"
        assert normalize_answer("0.5") == "0.5"
        from src.inference_budget.parsing import answers_match
        assert not answers_match("1/2", "0.5")


class TestValidatorCatchesDefects:
    """Test validate_batch_file catches changed temperature, prompt, and leaked keys."""

    def test_validator_catches_changed_temperature(self, tmp_path):
        config = load_config()
        # Create a single line with temperature 0.9 (config default is 0.7)
        batch_file = tmp_path / "test_temp.jsonl"
        req = create_batch_request(
            custom_id="gsm8k_q000001_g0001",
            model_id="test-model",
            prompt="Solve the following mathematical word problem carefully.\n\nProblem:\n1+1?",
            generation_params={"temperature": 0.9, "top_p": 1.0, "max_tokens": 512},
        )
        batch_file.write_text(json.dumps(req) + "\n", encoding="utf-8")
        errors = validate_batch_file(batch_file, config)
        assert any("Generation params" in e or "temperature" in e for e in errors)

    def test_validator_catches_changed_prompt(self, tmp_path):
        config = load_config()
        batch_file = tmp_path / "test_prompt.jsonl"
        req = create_batch_request(
            custom_id="gsm8k_q000001_g0001",
            model_id="test-model",
            prompt="A completely different prompt without the instruction prefix.",
            generation_params={"temperature": 0.7, "top_p": 1.0, "max_tokens": 512},
        )
        batch_file.write_text(json.dumps(req) + "\n", encoding="utf-8")
        errors = validate_batch_file(batch_file, config)
        assert any("instruction prefix" in e for e in errors)

    def test_validator_catches_leaked_key(self, tmp_path):
        config = load_config()
        batch_file = tmp_path / "test_key.jsonl"
        req = create_batch_request(
            custom_id="gsm8k_q000001_g0001",
            model_id="test-model",
            prompt="Solve the following mathematical word problem carefully.\n\nProblem:\n1+1?",
            generation_params={"temperature": 0.7, "top_p": 1.0, "max_tokens": 512},
        )
        # Embed an sk- key in the JSON
        raw_json = json.dumps(req)
        raw_json_with_key = raw_json[:-1] + ', "auth": "sk-live1234567890abcdef"}'
        batch_file.write_text(raw_json_with_key + "\n", encoding="utf-8")
        errors = validate_batch_file(batch_file, config)
        assert any("sk-..." in e for e in errors)


class TestFullBatchValidation:
    """Test that a generated 1000x50 batch validates completely."""

    def test_full_1000x50_batch_validates(self, tmp_path):
        config = load_config()
        dataset_path = resolve_path(config["paths"]["dataset_processed"])
        if not dataset_path.exists():
            pytest.skip("Processed dataset not available on disk")

        dataset_df = pd.read_csv(dataset_path)
        batch_out = tmp_path / "sheshnag_test_batch.jsonl"

        # Generate batch
        generate_batch_jsonl(
            dataset=dataset_df,
            model_id="actual-test-model-id",
            config=config,
            output_path=batch_out,
        )

        errors = validate_batch_file(batch_out, config)
        assert errors == [], f"Full batch validation failed with errors: {errors}"


class TestMockEndToEnd:
    """Test that mock runs end to end at 5x3."""

    def test_mock_5x3_subprocess(self):
        res = subprocess.run(
            [sys.executable, "scripts/run_mock.py", "--questions", "5", "--generations", "3"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"run_mock failed:\n{res.stderr}\n{res.stdout}"
        assert "MOCK EXPERIMENT COMPLETE" in res.stdout
        assert "Result integrity check passed." in res.stdout
