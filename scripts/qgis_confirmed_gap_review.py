#!/usr/bin/env python3
"""Compare explicit G previews on frozen visible endpoints in QGIS Python.

Run with QGIS Python and an isolated QGIS_CUSTOM_CONFIG_PATH. Use PYTHONPATH
to select an extracted candidate ZIP. Source hashes bind the loaded package.
The hidden curve has no accuracy reference: this records refusal, geometry,
exact preview/save equality and one Undo, never hidden-curve F1.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


_qgis_review_application = None


def run(fixtures, dataset):
    global _qgis_review_application
    import numpy as np
    from osgeo import gdal
    from qgis.core import Qgis, QgsApplication, QgsCoordinateReferenceSystem, QgsPointXY, QgsVectorLayer
    from qgis.gui import QgsMapCanvas
    from qgis.PyQt.QtCore import QCoreApplication, QEvent, QPointF, Qt
    import ai_vectorizer
    from ai_vectorizer.core.edge_detector import EdgeDetector
    from ai_vectorizer.core.livewire import LiveWireConfig, build_livewire_tree
    from ai_vectorizer.core.map_grid import detect_map_grid
    from ai_vectorizer.tools.smart_trace_tool import SmartTraceTool
    from qgis_grid_comparison import load_cases

    cases = load_cases(fixtures, dataset)
    gdal.UseExceptions()
    app = _qgis_review_application = QgsApplication([], False)
    app.initQgis()
    crs = QgsCoordinateReferenceSystem("EPSG:3857")
    records = []
    try:
        for case in cases:
            source = gdal.Open(case["image"])
            rgb = np.moveaxis(source.ReadAsArray()[:3], 0, -1).copy()
            source = None
            evidence = EdgeDetector(method=EdgeDetector.METHOD_INK).detect_ink_evidence(rgb)
            edges = np.uint8(evidence.centerline) * 255
            grid = detect_map_grid(rgb)
            start, end = [tuple(map(float, p)) for p in case["gap_endpoint_pair_xy"]]
            tree = build_livewire_tree(
                rgb, edges, start, evidence=evidence, grid_prior=grid,
                config=LiveWireConfig(label_gap_connections=False),
            )
            fallback = tree.trace(end)
            row = {"id": case["id"], "expected_pair": case["expected_pair"],
                   "hidden_geometry_scored": False, "traces": {}}
            for after in (False, True):
                canvas = QgsMapCanvas()
                canvas.setDestinationCrs(crs)
                layer = QgsVectorLayer("LineString?crs=EPSG:3857", "explicit gap review", "memory")
                tool = SmartTraceTool(canvas, SimpleNamespace(crs=lambda: crs), layer,
                                      freehand=True, language="en")
                try:
                    tool.freehand = False
                    tool.edge_method = EdgeDetector.METHOD_INK
                    tool.edge_weight = 1.0
                    tool.is_tracing = True
                    tool.start_point = QgsPointXY(*start)
                    tool.path_points = [QgsPointXY(*start)]
                    tool.checkpoints = [0]
                    tool.last_map_point = tool.path_points[-1]
                    tool.last_input_point = tool.path_points[-1]
                    tool.last_hover_pos = QgsPointXY(*end)
                    tool._livewire_request_point = QgsPointXY(*end)
                    tool._cache_generation = 1
                    tool.cached_ink_evidence = evidence
                    tool.cached_edges = edges
                    tool.cached_rgb_image = rgb
                    tool.cache_transform = {"x_min": 0., "y_max": float(rgb.shape[0]), "px_w": 1., "px_h": 1.}
                    # Native coordinate review avoids GUI rounding. The actual
                    # map tool, preview acceptance and QGIS edit buffer run.
                    tool.map_to_pixel_float = lambda p: (p.x(), p.y())
                    tool.map_to_pixel = lambda p: (round(p.x()), round(p.y()))
                    tool.pixel_to_map = lambda x, y: QgsPointXY(x, y)
                    tool.toMapCoordinates = lambda p: QgsPointXY(p.x(), p.y())
                    canvas.mapUnitsPerPixel = lambda: 1.0
                    tool._request_livewire_tree = mock.Mock(return_value=False)
                    tool.ask_elevation = lambda: 100.0
                    messages = []
                    tool._push_message = lambda text, *args: messages.append(text)
                    tool.preview_path = [QgsPointXY(*p) for p in fallback[1:]]
                    accepted_proposal = tool.preview_manual_gap_bridge(allow_confirmed_curve=after)
                    points = [(p.x(), p.y()) for p in [tool.path_points[-1], *tool.preview_path]]
                    one_undo = saved_exact = False
                    if accepted_proposal:
                        button = getattr(Qt, "LeftButton", None) or Qt.MouseButton.LeftButton
                        event = SimpleNamespace(pos=lambda: QPointF(*end), button=lambda: button,
                                                buttons=lambda: 0, modifiers=lambda: 0)
                        tool.canvasPressEvent(event)
                        preview = [(p.x(), p.y()) for p in tool.path_points]
                        if preview != points:
                            raise RuntimeError("manual endpoint click changed visible proposal")
                        right_button = getattr(Qt, "RightButton", None) or Qt.MouseButton.RightButton
                        tool.canvasPressEvent(SimpleNamespace(
                            pos=lambda: QPointF(*end), button=lambda: right_button,
                            buttons=lambda: 0, modifiers=lambda: 0,
                        ))
                        if layer.featureCount() != 1:
                            raise RuntimeError("manual proposal was not saved once")
                        saved = [(p.x(), p.y()) for p in next(layer.getFeatures()).geometry().asPolyline()]
                        saved_exact = saved == points
                        if not saved_exact:
                            raise RuntimeError("saved geometry differs from manual preview")
                        layer.undoStack().undo()
                        one_undo = layer.featureCount() == 0
                        if not one_undo:
                            raise RuntimeError("one Undo did not remove the proposal")
                    row["traces"]["after" if after else "before"] = {
                        "proposal_created": bool(accepted_proposal), "points_xy": points,
                        "saved_exact": saved_exact, "one_undo": one_undo, "messages": messages,
                    }
                finally:
                    tool.dispose()
                    if layer.isEditable():
                        layer.rollBack()
                    tool.deleteLater()
                    canvas.deleteLater()
                    layer.deleteLater()
                    app.processEvents()
                    deferred = getattr(QEvent, "DeferredDelete", None) or QEvent.Type.DeferredDelete
                    QCoreApplication.sendPostedEvents(None, deferred)
            records.append(row)
        package = Path(ai_vectorizer.__file__).parent
        hashes = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in package.rglob("*.py")}
        return {"ok": True, "qgis_version": Qgis.QGIS_VERSION,
                "ink_runtime": EdgeDetector.get_ink_runtime_status(), "numpy_version": np.__version__,
                "fixtures_sha256": hashlib.sha256(Path(fixtures).read_bytes()).hexdigest(),
                "source_hashes": hashes, "human_approved": False,
                "scope": "explicit G preview; native coordinates; hidden curve unknown; no model inference",
                "cases": records}
    finally:
        tool = layer = canvas = None
        gc.collect()
        app.processEvents()
        app.exitQgis()


def render_review(result, fixtures, dataset, output):
    """Show raw pixels beside refused Ink and the actual G proposal."""
    from PIL import Image, ImageDraw
    from qgis_grid_comparison import load_cases

    cases = {case["id"]: case for case in load_cases(fixtures, dataset)}
    output.mkdir(parents=True, exist_ok=True)
    for row in result["cases"]:
        case = cases[row["id"]]
        source = Image.open(case["image"]).convert("RGB")
        x0, y0, x1, y1 = case["raw_evidence_window_xyxy"]
        scale = 6
        width, height = (x1 - x0) * scale, (y1 - y0) * scale
        figure = Image.new("RGB", (3 * width, height + 48), "white")
        for col, key in enumerate((None, "before", "after")):
            tile = source.crop((x0, y0, x1, y1)).resize((width, height), Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(tile)
            convert = lambda p: ((p[0] - x0) * scale, (p[1] - y0) * scale)
            if key is not None:
                trace = row["traces"][key]
                draw.line([convert(p) for p in trace["points_xy"]],
                          fill=(0, 145, 0) if key == "after" else (205, 0, 175), width=2)
                label = key + (": G curve" if trace["proposal_created"] else ": G refused (Ink shown)")
            else:
                label = "Raw + explicit ends"
            for p in case["gap_endpoint_pair_xy"]:
                x, y = convert(p)
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(225, 0, 0), width=2)
            figure.paste(tile, (col * width, 24))
            ImageDraw.Draw(figure).text((col * width + 4, 5), label, fill="black")
        ImageDraw.Draw(figure).text((5, height + 30), "Hidden curve unknown; no hidden-geometry accuracy score.", fill="black")
        figure.save(output / (case["id"] + "-G-comparison.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-json", type=Path, help="Render a completed review with ordinary Python + Pillow")
    args = parser.parse_args()
    if args.render_json:
        render_review(json.loads(args.render_json.read_text()), args.fixtures, args.dataset, args.output)
        return
    result = run(args.fixtures, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"ok": result["ok"], "cases": [
        {"id": c["id"], "before": c["traces"]["before"]["proposal_created"],
         "after": c["traces"]["after"]["proposal_created"]} for c in result["cases"]]}))


if __name__ == "__main__":
    main()
