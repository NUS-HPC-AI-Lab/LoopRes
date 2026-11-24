import tempfile
import unittest
from unittest import mock

import torch
from transformers import AutoModelForCausalLM, DynamicCache, Qwen3ForCausalLM

from loopres import model as modeling
from loopres.model import LoopResForCausalLM, build
from tests.fakes import tiny_config, tiny_lm, tiny_loopres

# The window of previous steps whose core blocks each core sub-layer and the readout read.
VARIANTS = [{"previous_step": 1}, {"previous_step": 2}]


def loopres_model(**variant) -> LoopResForCausalLM:
    model = tiny_loopres(**variant)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "res_" in name:
                parameter.add_(torch.randn_like(parameter) * 0.5)
    return model


class BuildTest(unittest.TestCase):
    def test_loop_keys_pick_loopres_and_none_the_base_model(self) -> None:
        loop = {"steps": 2, "loop_layers": [1, 5], "attn_res_blocks": 2, "previous_step": 1}
        base = mock.patch.object(
            modeling.AutoModelForCausalLM, "from_pretrained", lambda name, **kw: tiny_lm(layers=6)
        )
        config = mock.patch.object(
            modeling.LoopResConfig, "from_pretrained", lambda name, **kw: tiny_config(layers=6, **kw)
        )
        pretrained = mock.patch.object(
            modeling.LoopResForCausalLM, "from_pretrained", lambda name, config, **kw: LoopResForCausalLM(config)
        )
        with base, config, pretrained:
            sft = build({"name": "x", "attn_implementation": "eager"}, torch.float32)
            loopres = build({"name": "x", "attn_implementation": "eager", **loop}, torch.float32)
        self.assertIsInstance(sft, Qwen3ForCausalLM)
        self.assertNotIsInstance(sft, LoopResForCausalLM)
        self.assertIsInstance(loopres, LoopResForCausalLM)

    def test_blocks_must_divide_the_core_layers(self) -> None:
        with self.assertRaises(ValueError):
            tiny_config(layers=6, steps=2, loop_layers=[1, 5], attn_res_blocks=3, previous_step=1)


class LoopResModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = torch.randint(0, 97, (2, 9))

    @torch.no_grad()
    def test_core_sub_layers_read_the_prelude_and_the_previous_steps_blocks_and_the_readout_feeds_the_coda(self) -> None:
        for variant in VARIANTS:
            with self.subTest(**variant):
                model = loopres_model(**variant)
                base = model.model
                embeds = base.embed_tokens(self.ids)
                positions = torch.arange(self.ids.shape[1])
                attention = {
                    "position_ids": positions[None],
                    "position_embeddings": base.rotary_emb(embeds, positions[None]),
                    "attention_mask": base._update_causal_mask(None, embeds, positions, None, False),
                }

                def stock(layer, hidden):
                    hidden = hidden + layer.self_attn(hidden_states=layer.input_layernorm(hidden), **attention)[0]
                    return hidden + layer.mlp(layer.post_attention_layernorm(hidden))

                def combine(sources, query, norm, keys, value):
                    """Each source weighed by `query` against its static key plus `value` against its RMS norm."""
                    logits = []
                    for source, key in zip(sources, keys, strict=True):
                        logit = key @ query + norm(source) @ value
                        logits.append(logit.expand(source.shape[:-1]))
                    weights = torch.stack(logits).softmax(0).unsqueeze(-1)
                    return sum(weight * source for weight, source in zip(weights, sources))

                def read(layer, step, sources, keys, total, partial):
                    """One core sub-layer pair: `partial` is the block's update sum so far, `total` its static key."""
                    summed = [total] if partial else []
                    attn = combine(
                        [*sources, *partial], layer.attn_res_query[step], layer.attn_res_norm, [*keys, *summed],
                        layer.attn_res_value_query[step],
                    )
                    update = layer.self_attn(hidden_states=layer.input_layernorm(attn), **attention)[0]
                    partial = update if not partial else partial[0] + update
                    hidden = combine(
                        [*sources, partial], layer.mlp_res_query[step], layer.mlp_res_norm, [*keys, total],
                        layer.mlp_res_value_query[step],
                    )
                    return partial + layer.mlp(layer.post_attention_layernorm(hidden))

                # The prelude is layer 0 alone, the coda layer 5; the core blocks are layers 1-2 and 3-4.
                prelude = stock(base.layers[0], embeds)
                history = []
                output = base(self.ids).last_hidden_state
                previous = variant["previous_step"]
                for step in range(model.config.steps):
                    # Each source beside its own key: the prelude's, then the window steps' block keys.
                    inputs, keys = [prelude], [base.prelude_key]
                    for earlier in range(max(step - previous, 0), step):
                        inputs += history[earlier]
                        keys += [base.block_keys[earlier][0], base.block_keys[earlier][1]]
                    own = [base.block_keys[step][0], base.block_keys[step][1]]
                    sums = [base.update_keys[step][0], base.update_keys[step][1]]
                    first = read(base.layers[1], step, inputs, keys, sums[0], [])
                    first = read(base.layers[2], step, inputs, keys, sums[0], [first])
                    second = read(base.layers[3], step, [*inputs, first], [*keys, own[0]], sums[1], [])
                    second = read(base.layers[4], step, [*inputs, first], [*keys, own[0]], sums[1], [second])
                    history.append([first, second])
                # The readout, once, over the last step's sources; then the coda and the final norm.
                state = combine(
                    [*inputs, first, second], base.attn_res_output_query, base.attn_res_output_norm,
                    [*keys, *own], base.attn_res_output_value_query,
                )
                torch.testing.assert_close(output, base.norm(stock(base.layers[5], state)))


class DecodeTest(unittest.TestCase):
    @torch.no_grad()
    def test_cached_decode_matches_the_full_forward(self) -> None:
        for variant in VARIANTS:
            with self.subTest(**variant):
                model = loopres_model(**variant)
                ids = torch.randint(0, 97, (1, 9))
                full = model(ids, use_cache=False).logits
                cache = DynamicCache()
                logits = [model(ids[:, :4], past_key_values=cache, use_cache=True).logits]
                for position in range(4, ids.shape[1]):
                    step = model(ids[:, position : position + 1], past_key_values=cache, use_cache=True)
                    logits.append(step.logits)
                torch.testing.assert_close(torch.cat(logits, dim=1), full, rtol=1.0e-4, atol=1.0e-4)

    @torch.no_grad()
    def test_generate_matches_greedy_full_forwards(self) -> None:
        model = loopres_model()
        ids = torch.randint(0, 97, (1, 5))
        expected = ids
        for _ in range(4):
            token = model(expected, use_cache=False).logits[:, -1:].argmax(-1)
            expected = torch.cat([expected, token], dim=1)
        generated = model.generate(ids, max_new_tokens=4, do_sample=False)
        self.assertEqual(generated.tolist(), expected.tolist())


class PretrainedInitTest(unittest.TestCase):
    def test_loading_a_base_checkpoint_initialises_the_loop_parameters(self) -> None:
        """`from_pretrained` materialises the parameters the base checkpoint lacks through `_init_weights`, not the
        constructor: without the override they come back as uninitialised memory."""
        with tempfile.TemporaryDirectory() as directory:
            tiny_lm(layers=6).save_pretrained(directory)
            config = tiny_config("eager", layers=6, steps=3, loop_layers=[1, 5], attn_res_blocks=2, previous_step=1)
            model = LoopResForCausalLM.from_pretrained(directory, config=config)
        base = model.model
        queries = [base.attn_res_output_query, base.attn_res_output_value_query]
        queries += [query for layer in base.layers[1:5] for name in
                    ("attn_res_query", "mlp_res_query", "attn_res_value_query", "mlp_res_value_query")
                    for query in getattr(layer, name)]
        for query in queries:
            self.assertTrue((query == 0).all())
        keys = torch.stack([base.prelude_key, *(key for step in base.block_keys for key in step),
                            *(key for step in base.update_keys for key in step)])
        self.assertTrue(keys.isfinite().all())
        # Drawn from N(0, 1): neither zeros nor garbage.
        self.assertGreater(keys.std().item(), 0.5)
        self.assertLess(keys.abs().max().item(), 10.0)


class CheckpointTest(unittest.TestCase):
    def test_a_saved_checkpoint_loads_through_auto_model_in_its_saved_dtype(self) -> None:
        ids = torch.randint(0, 97, (2, 9))
        model = loopres_model().to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = AutoModelForCausalLM.from_pretrained(directory, torch_dtype="auto", attn_implementation="eager").eval()
        self.assertIsInstance(loaded, LoopResForCausalLM)
        self.assertIs(loaded.dtype, torch.bfloat16)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0, msg=name)
        with torch.no_grad():
            # The weights are equal bit for bit; the forwards differ by kernel choice on the mmapped tensors.
            torch.testing.assert_close(loaded(ids).logits, model(ids).logits, rtol=5.0e-2, atol=5.0e-2)


if __name__ == "__main__":
    unittest.main()
