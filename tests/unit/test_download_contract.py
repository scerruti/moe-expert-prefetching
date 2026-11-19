import dataset_loaders
import model_loader
import pytest

pytestmark = pytest.mark.unit


def test_dataset_downloads_use_immutable_revisions(monkeypatch):
    calls = []

    def download(*args, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(dataset_loaders, "load_dataset", download)
    dataset_loaders.GSM8KLoader().load()
    dataset_loaders.MBPPLoader().load()
    assert all(len(call.get("revision", "")) == 40 for call in calls)


@pytest.mark.parametrize("field", ["revision", "trust_remote_code"])
def test_tokenizer_download_is_pinned_and_native(monkeypatch, field):
    calls = []

    def download(*args, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(model_loader.AutoTokenizer, "from_pretrained", download)
    model_loader.ModelLoader("qwen1.5-moe-a2.7b", device="cpu").load_tokenizer()
    if field == "revision":
        assert len(calls[0].get("revision", "")) == 40
    else:
        assert calls[0]["trust_remote_code"] is False
