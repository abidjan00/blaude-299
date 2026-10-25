"""Behavioral checks for causality, padding, gradient flow, and real data shapes."""

from dataclasses import replace
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Install requirements-week2.txt to run Week 2 tests")
from blaude.model import BlaudeModel, ModelConfig
from blaude.loader import TokenDataset
from blaude.torch_data import to_torch_batch


@pytest.fixture
def model():
    torch.manual_seed(299)
    torch.set_num_threads(1)
    return BlaudeModel(ModelConfig(vocab_size=31, context_length=12, d_model=32,
                                  n_heads=4, n_layers=2, d_ff=64, dropout=0.0)).eval()


def test_output_shapes_and_gradients_reach_all_model_parts(model):
    inputs = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 0]])
    targets = torch.tensor([[4, 5, 6, 2], [8, 9, 2, -100]])
    result = model(inputs, labels=targets)
    assert result["logits"].shape == (2, 4, 31)
    assert result["loss"].ndim == 0 and torch.isfinite(result["loss"])
    result["loss"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for part in [model.token_embedding, model.position_embedding, model.blocks[0].attention.qkv,
                 model.blocks[0].feed_forward[0], model.output]:
        assert part.weight.grad.abs().sum() > 0
    assert torch.count_nonzero(model.token_embedding.weight.grad[0]) == 0


def test_future_tokens_cannot_change_any_earlier_prediction(model):
    original = torch.tensor([[3, 4, 5, 6, 7, 8]])
    with torch.no_grad():
        expected = model(original)["logits"]
        for split in range(1, original.shape[1]):
            changed = original.clone()
            changed[:, split:] = 20
            actual = model(changed)["logits"]
            torch.testing.assert_close(actual[:, :split], expected[:, :split], rtol=0, atol=1e-6)
        # Positive control: changing present/past context should matter.
        changed = original.clone()
        changed[:, 0] = 25
        assert not torch.allclose(model(changed)["logits"][:, 1], expected[:, 1])


def test_attention_weights_have_no_future_or_padded_keys(model):
    captured = []
    attention = model.blocks[0].attention
    hook = attention.attention_dropout.register_forward_hook(lambda module, args, output: captured.append(output.detach()))
    try:
        model(torch.tensor([[3, 4, 0, 0], [5, 6, 7, 0]]))
    finally:
        hook.remove()
    weights = captured[0]
    assert torch.count_nonzero(weights.triu(diagonal=1)) == 0
    assert torch.count_nonzero(weights[0, :, :, 2:]) == 0
    assert torch.count_nonzero(weights[1, :, :, 3:]) == 0
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))


def test_padding_and_padded_targets_do_not_change_real_logits_or_loss(model):
    original = torch.tensor([[3, 4, 5]])
    targets = torch.tensor([[4, 5, 2]])
    expected = model(original, labels=targets)
    padded = torch.tensor([[3, 4, 5, 24, 23]])
    mask = torch.tensor([[True, True, True, False, False]])
    # Deliberately provide normal IDs at padding positions: the mask must ignore them.
    actual = model(padded, mask, torch.tensor([[4, 5, 2, 19, 18]]))
    torch.testing.assert_close(actual["logits"][:, :3], expected["logits"], rtol=0, atol=1e-6)
    torch.testing.assert_close(actual["loss"], expected["loss"])
    assert torch.count_nonzero(actual["logits"][:, 3:]) == 0


def test_loss_uses_week1_targets_without_shifting_again(model):
    inputs = torch.tensor([[3, 4, 5, 6]])
    targets = torch.tensor([[4, 5, -100, 2]])
    result = model(inputs, labels=targets)
    selected = result["logits"][0, [0, 1, 3]]
    expected = -selected.log_softmax(dim=-1)[torch.arange(3), torch.tensor([4, 5, 2])].mean()
    torch.testing.assert_close(result["loss"], expected)


def test_week1_loader_connects_to_the_model(tmp_path, model):
    np.save(tmp_path / "train.tokens.npy", np.array([3, 4, 5, 6, 2, 7, 8, 2], dtype=np.uint32))
    np.save(tmp_path / "train.offsets.npy", np.array([0, 5, 8], dtype=np.int64))
    batch = to_torch_batch(next(TokenDataset(tmp_path, context_length=6).batches(2)))
    result = model(**batch)
    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.bool
    assert result["logits"].shape == (2, 6, 31)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()


@pytest.mark.parametrize("kwargs", [
    {"d_model": 30, "n_heads": 4}, {"n_layers": 0}, {"context_length": 0},
    {"dropout": 1}, {"pad_id": 31},
])
def test_invalid_configuration_has_a_clear_error(kwargs):
    with pytest.raises(ValueError):
        replace(ModelConfig(vocab_size=31), **kwargs)


@pytest.mark.parametrize("inputs,mask", [
    (torch.tensor([[0, 0]]), None),
    (torch.tensor([[0, 3]]), None),
    (torch.tensor([[3, 0, 4]]), None),
    (torch.tensor([[31]]), None),
    (torch.tensor([[-1]]), None),
    (torch.tensor([[3.0]]), None),
    (torch.ones((1, 13), dtype=torch.long), None),
    (torch.tensor([[3, 4]]), torch.tensor([[1, 1]])),
])
def test_bad_inputs_fail_without_nan(model, inputs, mask):
    with pytest.raises(ValueError):
        model(inputs, mask)


def test_no_valid_targets_is_a_clear_error(model):
    with pytest.raises(ValueError, match="real target"):
        model(torch.tensor([[3, 4]]), labels=torch.tensor([[-100, -100]]))


def test_randomly_initialized_model_can_memorize_a_tiny_batch():
    torch.manual_seed(299)
    torch.set_num_threads(1)
    model = BlaudeModel(ModelConfig(vocab_size=13, context_length=6, d_model=32,
                                   n_heads=4, n_layers=1, d_ff=64, dropout=0.0))
    inputs = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
    targets = torch.tensor([[4, 5, 6, 2], [8, 9, 10, 2]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    initial = model(inputs, labels=targets)["loss"].item()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs, labels=targets)["loss"]
        loss.backward()
        optimizer.step()
    result = model(inputs, labels=targets)
    assert result["loss"].item() < min(0.1, initial * 0.1)
    assert torch.equal(result["logits"].argmax(dim=-1), targets)
