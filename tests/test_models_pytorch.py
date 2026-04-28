import pytest

torch = pytest.importorskip("torch")

from src.models_pytorch import build_resnet50_13ch


@pytest.mark.slow
def test_resnet50_13ch_forward_shape():
    model = build_resnet50_13ch(num_classes=10)
    model.eval()
    x = torch.randn(2, 13, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10)


@pytest.mark.slow
def test_resnet50_13ch_first_conv_in_channels():
    model = build_resnet50_13ch(num_classes=10)
    assert model.conv1.in_channels == 13
    assert model.fc.out_features == 10
