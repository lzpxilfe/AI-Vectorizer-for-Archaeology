"""The new Colab experiment cannot silently reuse old runs or new diagnostics."""

import json
from pathlib import Path


def test_text_notebook_is_unexecuted_and_every_code_cell_compiles():
    path = Path(__file__).resolve().parents[1] / "learning" / "colab_train_korea_text_weak.ipynb"
    document = json.loads(path.read_text())
    for index, cell in enumerate(document["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), str(path) + ":" + str(index), "exec")


def test_text_notebook_reuses_only_frozen_data_and_uses_separate_fresh_runs():
    path = Path(__file__).resolve().parents[1] / "learning" / "colab_train_korea_text_weak.ipynb"
    document = json.loads(path.read_text())
    configuration = {}
    exec("".join(document["cells"][1]["source"]), configuration)
    assert configuration["DATASET_ZIP"] == "/content/drive/MyDrive/ArchaeoTrace/ai-weak-korea/weak-dataset.zip"
    assert configuration["EXPECTED_DATASET_SHA256"] == "b2fd2a087f75de4e5e1d31b44f13dfdb1a8af3963e18e68f81f9cb4a91e32249"
    assert configuration["OUTPUT_DIRECTORY"] == "/content/drive/MyDrive/ArchaeoTrace/ai-weak-korea-general-text-v1/runs"
    assert configuration["SEEDS"] == (3407, 3408, 3409)
    assert configuration["TRAIN_OPTIONS"] == {"ink_recolor": True, "label_gaps": True, "word_gaps": True}
    assert configuration["MODEL_ID"] == "ai-weak-contour-guide-korea-text-v1"
    code = "\n".join("".join(cell["source"]) for cell in document["cells"] if cell["cell_type"] == "code")
    assert 'frozen.manifest_sha256 != EXPECTED_DATASET_SHA256' in code
    assert 'sample = training[index]' in code
    assert 'sample = validation[index]' not in code
    assert 'name.startswith("learning.")' in code
    assert 'Old experiment source is cached' in code
    assert 'Resume checkpoint used different augmentation options.' in code
