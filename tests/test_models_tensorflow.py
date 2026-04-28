import pytest

tf = pytest.importorskip("tensorflow")

from src.models_tensorflow import build_mobilenetv2_13ch


@pytest.mark.slow
def test_mobilenetv2_13ch_forward_shape():
    model = build_mobilenetv2_13ch(num_classes=10, input_size=64)
    import numpy as np
    x = np.random.rand(2, 64, 64, 13).astype("float32")
    out = model.predict(x, verbose=0)
    assert out.shape == (2, 10)


@pytest.mark.slow
def test_mobilenetv2_13ch_input_shape():
    model = build_mobilenetv2_13ch(num_classes=10, input_size=64)
    assert model.input_shape == (None, 64, 64, 13)
    assert model.output_shape == (None, 10)
