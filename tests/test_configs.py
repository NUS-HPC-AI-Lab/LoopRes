import unittest
from pathlib import Path

import yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
# Every training key scripts.train reads; `max_steps` and `lr_scheduler_kwargs` are its optional reads.
TRAINING_KEYS = {
    "run_name", "output_dir", "report_to", "seed", "epochs", "max_steps", "per_device_train_batch_size",
    "gradient_accumulation_steps", "learning_rate", "adam_beta1", "adam_beta2", "weight_decay", "max_grad_norm",
    "warmup_ratio", "lr_scheduler_type", "lr_scheduler_kwargs", "gradient_checkpointing", "save_steps",
}
# `loopres.model.build`'s keys: the base model, and for LoopRes its constructor's.
MODEL_KEYS = {"name", "attn_implementation", "steps", "loop_layers", "attn_res_blocks", "previous_step"}


class ConfigTest(unittest.TestCase):
    def test_run_configs(self) -> None:
        paths = sorted(path for path in CONFIGS.glob("*.yaml") if not path.name.startswith("eval"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text())
                self.assertLessEqual(set(config["model"]), MODEL_KEYS)
                self.assertEqual(config["model"]["attn_implementation"], "flash_attention_2")
                self.assertEqual(set(config["data"]), {"train", "sample_length"})
                training = config["training"]
                self.assertLessEqual(set(training), TRAINING_KEYS)
                self.assertLessEqual(TRAINING_KEYS - {"max_steps", "lr_scheduler_kwargs"}, set(training))
                self.assertEqual(training["run_name"], path.stem)
                self.assertEqual(training["output_dir"], f"outputs/{path.stem}")
                self.assertIsInstance(training["learning_rate"], float)
                self.assertIsInstance(training["per_device_train_batch_size"], int)

    def test_the_evaluation_config(self) -> None:
        from lm_eval.tasks import TaskManager

        entries = yaml.safe_load((CONFIGS / "eval.yaml").read_text())["benchmarks"]
        tasks = TaskManager(include_path="scripts/tasks", include_defaults=False).all_tasks
        for name, entry in entries.items():
            with self.subTest(benchmark=name):
                self.assertIsInstance(entry.get("limit", 1), int)
                self.assertIsInstance(entry.get("samples", 1), int)
                if "task" in entry:
                    self.assertIn(entry["task"], tasks)
                else:
                    self.assertIn(entry["evalplus"], {"humaneval", "mbpp"})
                    self.assertIn(Path(entry["problems"]).name, ("HumanEvalPlus-v0.1.10.jsonl", "MbppPlus-v0.2.0.jsonl"))


if __name__ == "__main__":
    unittest.main()
