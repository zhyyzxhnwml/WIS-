from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from train.wis.collator import CondDropoutCollator, PreferenceDropoutCollator, encode_sample, prompt_ids, render_prompt
from train.wis.config import ROOT, load_config
from train.wis.data import assert_disjoint, read_jsonl, validate_response, validate_rows, write_jsonl
from train.wis.evaluate import blind_packet, summarize
from train.wis.infer import parse_prediction
from train.wis.prepare import grouped_split
from train.wis.runtime import preflight

try:
    import torch
except (ImportError, OSError):
    torch = None

EXAMPLES = ROOT / "train/wis/data/examples"


class FakeTokenizer:
    chat_template = "test"
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<user>" + messages[0]["content"] + "</user><assistant><think>\n"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 2 for c in text]


class OfflineTest(unittest.TestCase):
    def setUp(self):
        self.train = read_jsonl(EXAMPLES / "sft_train.jsonl")
        self.evaluation = read_jsonl(EXAMPLES / "sft_eval.jsonl")

    def test_demo_is_blocked_and_production_requires_review(self):
        with self.assertRaisesRegex(ValueError, "demo sample"):
            validate_rows(self.train)
        validate_rows(self.train, allow_demo=True)
        row = copy.deepcopy(self.train[0])
        row["demo"] = False
        with self.assertRaisesRegex(ValueError, "reviewed=true"):
            validate_rows([row])

    def test_bad_mapping_and_empty_validation_are_rejected(self):
        row = copy.deepcopy(self.train[0])
        row["completion"]["mechanisms"][0]["mapping"][0]["condition"] = ""
        with self.assertRaisesRegex(ValueError, "mapping.condition"):
            validate_rows([row], allow_demo=True)

    def test_question_family_and_optional_mechanism_leakage(self):
        assert_disjoint({"train": self.train, "eval": self.evaluation})
        changed = copy.deepcopy(self.evaluation[0])
        changed["problem_group"] = self.train[0]["problem_group"]
        with self.assertRaisesRegex(ValueError, "problem_group"):
            assert_disjoint({"train": self.train, "eval": [changed]})
        changed["problem_group"] = "other"
        changed["mechanism_group"] = self.train[0]["mechanism_group"]
        with self.assertRaisesRegex(ValueError, "mechanism_group"):
            assert_disjoint({"train": self.train, "eval": [changed]}, True)

    def test_exact_duplicate_question_is_rejected_even_if_group_wrong(self):
        changed = copy.deepcopy(self.evaluation[0])
        changed["problem"]["question"] = "  " + self.train[0]["problem"]["question"]
        with self.assertRaisesRegex(ValueError, "question"):
            assert_disjoint({"train": self.train, "eval": [changed]})

    def test_grouped_split_is_reproducible_and_keeps_pairs(self):
        rows = []
        for i in range(9):
            row = copy.deepcopy(self.train[0])
            row.update(id=str(i), problem_group=f"group-{i}")
            row["problem"]["question"] = f"question-{i}"
            rows.append(row)
        pair = copy.deepcopy(rows[0])
        pair["id"] = "pair"
        rows.append(pair)
        result = grouped_split(rows)
        self.assertEqual(result, grouped_split(rows))
        self.assertEqual(sum(len(v) for v in result.values()), len(rows))
        containing = [name for name, split in result.items() if any(r["id"] == "0" for r in split)]
        self.assertTrue(any(r["id"] == "pair" for r in result[containing[0]]))

    def test_jsonl_writer_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            write_jsonl(path, self.train)
            self.assertEqual(read_jsonl(path), self.train)
            with self.assertRaises(FileExistsError):
                write_jsonl(path, [])

    def test_every_reference_config_preflights_without_loading_model(self):
        import contextlib
        import io
        for file in (ROOT / "train/wis/configs").glob("*.json"):
            cfg = load_config(file)
            kind = "dpo" if file.stem == "dpo" else "sft"
            with contextlib.redirect_stdout(io.StringIO()):
                preflight(cfg, kind, allow_demo=True)

    def test_dpo_null_preferences_cannot_be_assumed(self):
        cfg = load_config(ROOT / "train/wis/configs/dpo.json")
        cfg["cond_dropout"] = 0.15
        with self.assertRaisesRegex(ValueError, "null_preference_validated"):
            preflight(cfg, "dpo", True)

    def test_invalid_config_is_rejected(self):
        raw = json.loads((ROOT / "train/wis/configs/sft_baseline.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            raw["entropy_lambda"] = -0.1
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "entropy_lambda"):
                load_config(path)

    def test_chat_boundaries_and_no_thinking_prefix_are_shared(self):
        tokenizer = FakeTokenizer()
        text = render_prompt(tokenizer, self.train[0]["problem"])
        self.assertIn("</user><assistant><think>\n</think>", text)
        encoded = encode_sample(tokenizer, self.train[0], 10000)
        prefix = prompt_ids(tokenizer, self.train[0]["problem"])
        self.assertEqual(encoded["input_ids"][:len(prefix)], prefix)
        self.assertEqual(encoded["labels"][:len(prefix)], [-100] * len(prefix))
        self.assertEqual(encoded["labels"][-1], tokenizer.eos_token_id)
        truncated = encode_sample(tokenizer, self.train[0], 32)
        self.assertLessEqual(len(truncated["input_ids"]), 32)
        self.assertEqual(len(truncated["input_ids"]), len(truncated["labels"]))

    def test_parser_accepts_explicit_deferral_and_rejects_broken_output(self):
        obj, error = parse_prediction(json.dumps(self.train[0]["completion"]))
        self.assertIsNone(error)
        self.assertTrue(obj["mechanisms"])
        self.assertTrue(obj["mechanisms"][0]["mapping"])
        self.assertIsNone(parse_prediction("a convincing but unstructured explanation")[0])
        deferred = {"status": "needs_clarification", "reason": "场景未知", "questions": ["哪里受到冲击？"]}
        self.assertEqual(parse_prediction(json.dumps(deferred))[0], deferred)
        with self.assertRaises(ValueError):
            validate_response(dict(deferred, questions=[]))

    def test_training_data_can_teach_appropriate_deferral(self):
        row = copy.deepcopy(self.train[0])
        row["mechanism_group"] = "deferral"
        row["completion"] = {"status": "no_candidate", "reason": "明确约束冲突", "questions": []}
        validate_rows([row], allow_demo=True)
        encoded = encode_sample(FakeTokenizer(), row, 10000)
        self.assertTrue(any(value != -100 for value in encoded["labels"]))

    def test_fixed_evaluation_questions_and_no_train_overlap(self):
        rows = read_jsonl(EXAMPLES / "eval_prompts.jsonl")
        self.assertEqual(len(rows), 20)
        validate_rows(rows, "prompts")
        assert_disjoint({"train": self.train, "eval": rows})

    def test_unreviewed_outputs_have_no_invented_quality_score(self):
        predictions = [{"prediction_id": "opaque", "question_id": "q", "scale": 1.0,
                        "completion": self.train[0]["completion"]}]
        report = summarize(predictions)["by_scale"]["1.0"]
        self.assertIsNone(report["accepted_among_reviewed"])
        self.assertIsNone(report["accepted_mechanisms_per_question"])
        reviews = [{"prediction_id": "opaque", "source_supported": True, "mapping_valid": True,
                    "constraints_satisfied": True, "mechanism_group": "collapse"}]
        reviewed = summarize(predictions, reviews)["by_scale"]["1.0"]
        self.assertEqual(reviewed["accepted"], 1)
        self.assertEqual(reviewed["accepted_mechanisms_per_question"], 1)
        self.assertNotIn("scale", blind_packet(predictions)[0])
        with self.assertRaisesRegex(ValueError, "boolean"):
            summarize(predictions, [dict(reviews[0], mapping_valid="yes")])

    def test_deferrals_are_reported_separately(self):
        predictions = [{"prediction_id": "a", "question_id": "q", "scale": 1,
                        "completion": {"status": "no_candidate", "reason": "约束冲突", "questions": []}}]
        report = summarize(predictions, [{"prediction_id": "a", "appropriate_deferral": True}])["by_scale"]["1"]
        self.assertEqual(report["deferrals"], 1)
        self.assertEqual(report["appropriate_deferrals"], 1)
        self.assertEqual(report["accepted"], 0)
        self.assertEqual(report["review_coverage"], 1.0)


@unittest.skipIf(torch is None, "PyTorch is not installed; numerical CPU checks skipped")
class TensorTest(unittest.TestCase):
    def test_sft_entropy_hook_preserves_ce_and_is_disabled_during_eval(self):
        import sys
        import types
        from unittest.mock import patch
        from train.wis.entropy import sft_trainer_class

        class FakeTrainer:
            def __init__(self, **kwargs):
                pass

        class FakeModel(torch.nn.Module):
            def forward(self, **inputs):
                return types.SimpleNamespace(loss=torch.tensor(2.0), logits=torch.zeros((1, 3, 3)))

        module = types.ModuleType("transformers")
        module.Trainer = FakeTrainer
        with patch.dict(sys.modules, {"transformers": module}):
            trainer = sft_trainer_class()(entropy_lambda=0.1)
        model = FakeModel()
        inputs = {"labels": torch.tensor([[-100, 1, 2]])}
        self.assertAlmostEqual(trainer.compute_loss(model, inputs).item(), 2.0 - 0.1 * math.log(3), places=6)
        model.eval()
        self.assertEqual(trainer.compute_loss(model, inputs).item(), 2.0)

    def test_dpo_entropy_hook_keeps_preference_objective_and_masks_prompt(self):
        import sys
        import types
        from unittest.mock import patch
        from train.wis.entropy import dpo_trainer_class

        class FakeDPOTrainer:
            def __init__(self, **kwargs):
                self.processing_class = FakeTokenizer()

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                return (torch.tensor(2.0), {"preference_metric": 0.8}) if return_outputs else torch.tensor(2.0)

        class FakeModel(torch.nn.Module):
            def forward(self, input_ids, **kwargs):
                return types.SimpleNamespace(logits=torch.zeros((1, input_ids.shape[1], 3)))

        module = types.ModuleType("trl")
        module.DPOTrainer = FakeDPOTrainer
        with patch.dict(sys.modules, {"trl": module}):
            trainer = dpo_trainer_class()(entropy_lambda=0.1)
        inputs = {"prompt_input_ids": torch.tensor([[0, 4]]), "prompt_attention_mask": torch.tensor([[0, 1]]),
                  "chosen_input_ids": torch.tensor([[2, 1]]), "chosen_attention_mask": torch.tensor([[1, 1]])}
        model = FakeModel()
        loss, metadata = trainer.compute_loss(model, inputs, return_outputs=True)
        self.assertAlmostEqual(loss.item(), 2.0 - 0.1 * math.log(3), places=6)
        self.assertEqual(metadata, {"preference_metric": 0.8})
        model.eval()
        self.assertEqual(trainer.compute_loss(model, inputs).item(), 2.0)

    def test_dropout_rebuilds_masks_and_does_not_mutate_rows(self):
        rows = read_jsonl(EXAMPLES / "sft_train.jsonl")
        original = copy.deepcopy(rows)
        tokenizer = FakeTokenizer()
        collator = CondDropoutCollator(tokenizer, 10000, probability=1.0)
        batch = collator(rows)
        self.assertEqual(rows, original)
        self.assertEqual(batch["input_ids"].shape, batch["attention_mask"].shape)
        self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)
        self.assertTrue(torch.all(batch["labels"][batch["attention_mask"] == 0] == -100))
        n = len(prompt_ids(tokenizer))
        self.assertEqual(batch["input_ids"][0, :n].tolist(), prompt_ids(tokenizer))
        evaluator = CondDropoutCollator(tokenizer, 10000, probability=1.0, training_check=lambda: False)
        expected = encode_sample(tokenizer, rows[0], 10000)
        self.assertEqual(evaluator(rows)["input_ids"][0, :len(expected["input_ids"])].tolist(), expected["input_ids"])

    def test_preference_dropout_replaces_one_shared_prompt(self):
        original = [{"prompt_input_ids": [1, 2], "chosen_input_ids": [3], "rejected_input_ids": [4]}]
        wrapped = PreferenceDropoutCollator(lambda rows: rows, [9], 1.0)
        replaced = wrapped(original)
        self.assertEqual(original[0]["prompt_input_ids"], [1, 2])
        self.assertEqual(replaced[0]["prompt_input_ids"], [9])
        self.assertEqual(replaced[0]["chosen_input_ids"], [3])
        self.assertEqual(replaced[0]["rejected_input_ids"], [4])
        with self.assertRaisesRegex(ValueError, "caching"):
            wrapped([dict(original[0], ref_chosen_logps=0.0)])

    def test_entropy_masks_prompt_padding_and_has_finite_gradients(self):
        from train.wis.entropy import masked_entropy
        logits = torch.zeros((1, 4, 3), requires_grad=True)
        labels = torch.tensor([[-100, -100, 2, -100]])
        value = masked_entropy(logits, labels)
        self.assertAlmostEqual(value.item(), math.log(3), places=6)
        (logits.sum() * 0 + value).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        sharp = torch.zeros((1, 4, 3), requires_grad=True)
        with torch.no_grad():
            sharp[0, 1, 0] = 4
        entropy_value = masked_entropy(sharp, labels)
        entropy_value.backward()
        self.assertLess(entropy_value.item(), value.item())
        self.assertTrue(torch.all(sharp.grad[0, 0] == 0))
        self.assertTrue(torch.all(sharp.grad[0, 2:] == 0))
        self.assertNotEqual(sharp.grad[0, 1, 0].item(), 0)
        self.assertEqual(masked_entropy(logits, torch.full_like(labels, -100)).item(), 0)

    def test_cfg_endpoints_and_normalized_sampling(self):
        from train.wis.cfg import cfg_scores, sample_token
        lc = torch.tensor([[2.0, 0.0, -1.0]])
        lu = torch.tensor([[0.0, 1.0, -1.0]])
        self.assertTrue(torch.allclose(cfg_scores(lc, lu, 1), torch.log_softmax(lc, -1)))
        self.assertTrue(torch.allclose(cfg_scores(lc, lu, 0), torch.log_softmax(lu, -1)))
        expected = torch.log_softmax(lu, -1) + 0.45 * (torch.log_softmax(lc, -1) - torch.log_softmax(lu, -1))
        self.assertTrue(torch.allclose(cfg_scores(lc, lu, 0.45), expected))
        self.assertEqual(sample_token(lc, 0.7, 0.01).item(), 0)

    def test_cfg_branches_share_history_keep_distinct_caches_and_stop(self):
        from types import SimpleNamespace
        from train.wis.cfg import generate_ids

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.calls = []

            def forward(self, input_ids, attention_mask, past_key_values=None, **kwargs):
                ids = input_ids[0].tolist()
                history = list(past_key_values or []) + ids
                self.calls.append((ids, attention_mask.shape[1], list(past_key_values or [])))
                return SimpleNamespace(logits=torch.tensor([[[0.0, -10.0, 10.0, -10.0]]]), past_key_values=history)

        model = FakeModel()
        self.assertEqual(generate_ids(model, [5, 6], [7], 0.45, 3, top_p=0.01), [2, 2, 2])
        self.assertEqual(len(model.calls), 6)
        self.assertEqual(model.calls[2][0], model.calls[3][0])
        self.assertEqual(model.calls[2][1], 3)
        self.assertEqual(model.calls[3][1], 2)
        self.assertEqual(model.calls[2][2], [5, 6])
        self.assertEqual(model.calls[3][2], [7])
        model = FakeModel()
        self.assertEqual(generate_ids(model, [5], [7], 1, 3, top_p=0.01, eos_token_ids=[2]), [2])
        self.assertEqual(len(model.calls), 1)


if __name__ == "__main__":
    unittest.main()
