import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

from scripts.train import TrainModel, checkpoint_step, load_state, parameter_groups, save_state
from tests.fakes import tiny_lm, tiny_loopres


class TrainModelTest(unittest.TestCase):
    def batch(self, length: int = 8) -> dict[str, torch.Tensor]:
        ids = torch.randint(0, 97, (1, length))
        labels = torch.full((1, length), -100)
        labels[0, 3:6] = ids[0, 3:6]
        return {"input_ids": ids, "position_ids": torch.arange(length)[None], "labels": labels}

    def test_ce_is_the_cross_entropy_at_the_labelled_tokens(self) -> None:
        model = tiny_lm()
        train_model = TrainModel(model)
        batch = self.batch()
        ce = train_model(batch)
        self.assertEqual(ce.shape, (3,))
        with torch.no_grad():
            hidden = model.model(batch["input_ids"], position_ids=batch["position_ids"]).last_hidden_state
            logits = model.lm_head(hidden).float()
        expected = F.cross_entropy(logits[0, 3:6], batch["labels"][0, 3:6], reduction="none")
        torch.testing.assert_close(ce, expected)

    def test_the_loss_backpropagates_through_the_chunked_recompute(self) -> None:
        train_model = TrainModel(tiny_loopres(layers=4, steps=2, loop_layers=[1, 3]))
        ce = train_model(self.batch())
        self.assertEqual(ce.shape, (3,))
        ce.sum().backward()
        query = train_model.model.model.attn_res_output_query
        self.assertIsNotNone(query.grad)


class ParameterGroupTest(unittest.TestCase):
    def test_matrices_decay_and_vectors_do_not(self) -> None:
        module = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        matrices, vectors = parameter_groups(module, weight_decay=1.0e-2)
        named = {id(parameter): name for name, parameter in module.named_parameters()}
        self.assertEqual(matrices["weight_decay"], 1.0e-2)
        self.assertEqual(vectors["weight_decay"], 0.0)
        self.assertEqual({named[id(p)] for p in matrices["params"]}, {"0.weight"})
        self.assertEqual({named[id(p)] for p in vectors["params"]}, {"0.bias", "1.weight", "1.bias"})

    def test_loopres_queries_keys_and_norms_do_not_decay(self) -> None:
        model = tiny_loopres(layers=4, steps=2, loop_layers=[1, 3])
        train_model = TrainModel(model)
        matrices, vectors = parameter_groups(train_model, weight_decay=1.0e-2)
        named = {id(parameter): name for name, parameter in train_model.named_parameters()}
        undecayed = {named[id(p)] for p in vectors["params"]}
        self.assertTrue(any("prelude_key" in name for name in undecayed))
        self.assertTrue(any("block_keys" in name for name in undecayed))
        self.assertTrue(any("update_keys" in name for name in undecayed))
        self.assertTrue(any("attn_res_query" in name for name in undecayed))
        self.assertFalse(any("res_" in named[id(p)] for p in matrices["params"]))


class CheckpointStateTest(unittest.TestCase):
    def test_save_and_load_restore_model_optimizer_and_trainer(self) -> None:
        train_model = TrainModel(tiny_lm())
        optimizer = torch.optim.AdamW(parameter_groups(train_model, 1.0e-2), lr=1.0e-3)
        ids = torch.randint(0, 97, (1, 6))
        labels = ids.clone()
        batch = {"input_ids": ids, "position_ids": torch.arange(6)[None], "labels": labels}
        train_model(batch).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        trainer = {"step": 7, "alignment": {"world": 1}}
        # The suite runs on the CPU; the CUDA RNG state is stood in for.
        rng = mock.patch.multiple(
            torch.cuda, get_rng_state=torch.get_rng_state, set_rng_state=lambda state: None
        )
        with tempfile.TemporaryDirectory() as directory, rng:
            checkpoint = Path(directory) / "checkpoint-7"
            checkpoint.mkdir()
            save_state(checkpoint, train_model, optimizer, trainer, rank=0, distributed=False)
            self.assertEqual(checkpoint_step(checkpoint), 7)
            expected = {name: tensor.clone() for name, tensor in train_model.state_dict().items()}
            with torch.no_grad():
                for parameter in train_model.parameters():
                    parameter.add_(1.0)
            restored = load_state(checkpoint, train_model, optimizer, rank=0)
        self.assertEqual(restored, trainer)
        for name, tensor in train_model.state_dict().items():
            torch.testing.assert_close(tensor, expected[name], msg=name)


if __name__ == "__main__":
    unittest.main()
