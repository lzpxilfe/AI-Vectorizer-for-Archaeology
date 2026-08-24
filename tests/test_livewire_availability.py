import numpy as np
import pytest

import ai_vectorizer.core.livewire as livewire_module


def test_broken_scipy_import_is_a_cached_unavailable_runtime(monkeypatch):
    calls = 0

    def fail_import():
        nonlocal calls
        calls += 1
        raise OSError("compiled SciPy extension cannot be loaded")

    livewire_module._get_livewire_runtime.cache_clear()
    monkeypatch.setattr(livewire_module, "_import_livewire_runtime", fail_import)
    try:
        assert not livewire_module.is_livewire_available()
        assert not livewire_module.is_livewire_available()
        image = np.zeros((8, 8), dtype=np.uint8)
        with pytest.raises(livewire_module.LiveWireUnavailable, match="compiled SciPy extension"):
            livewire_module.build_livewire_tree(image, image, (4, 4))
        assert calls == 1
    finally:
        livewire_module._get_livewire_runtime.cache_clear()
