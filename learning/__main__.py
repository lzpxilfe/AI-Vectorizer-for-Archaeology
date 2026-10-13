"""Dataset preparation, Colab training and CPU evaluation entry point."""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Keep optional PyTorch/ONNX imports out of data preparation and QGIS.
    modules = {"train":"train", "export":"export", "evaluate":"evaluate"}
    if argv and argv[0] in modules:
        import importlib
        result = importlib.import_module("learning." + modules[argv.pop(0)]).main(argv)
        return result or 0
    from .data_cli import add_data_commands, dispatch_data
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_commands(parser)
    return dispatch_data(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
