#!/usr/bin/env python3
"""Run an installed-plugin tracing smoke test in an isolated QGIS profile.

The default entry point is an outer, dependency-free orchestrator.  It checks
and installs one ArchaeoTrace ZIP into a newly-created profile, launches the
real QGIS desktop binary with that profile, and consumes a JSON result written
by the in-process bootstrap below.  The user's ordinary QGIS profile is never
read or modified.

This is intentionally a release/manual smoke rather than a normal pytest job:
it requires a complete QGIS desktop installation and exercises its Python
plugin loader, GUI interface, raster provider, task manager and map tool.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Optional, Sequence
import zipfile


PLUGIN_ID = "ai_vectorizer"
PROFILE_NAME = "archaeotrace-clean-smoke"
BOOTSTRAP_ENV = "ARCHAEOTRACE_CLEAN_PROFILE_BOOTSTRAP"
PROFILE_ROOT_ENV = "ARCHAEOTRACE_CLEAN_PROFILE_ROOT"
PLUGIN_DIR_ENV = "ARCHAEOTRACE_CLEAN_PLUGIN_DIR"
RESULT_PATH_ENV = "ARCHAEOTRACE_CLEAN_RESULT_PATH"
MAX_ARCHIVE_BYTES = 20_000_000
MAX_EXPANDED_BYTES = 40_000_000
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_EXPECTED_VERSION = "0.1.6"


class SmokeFailure(RuntimeError):
    """A deterministic clean-profile contract failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Validate a QGIS plugin archive before writing anything from it."""

    members: list[zipfile.ZipInfo] = []
    folded_names: set[str] = set()
    expanded_bytes = 0
    metadata_seen = False
    for member in archive.infolist():
        raw_name = member.filename
        if not raw_name or "\\" in raw_name or member.flag_bits & 0x1:
            raise SmokeFailure(f"Unsafe ZIP member name or encryption: {raw_name!r}")
        path = PurePosixPath(raw_name)
        if path.is_absolute() or not path.parts:
            raise SmokeFailure(f"Unsafe absolute ZIP member: {raw_name!r}")
        if any(part in ("", ".", "..") for part in path.parts):
            raise SmokeFailure(f"Unsafe ZIP traversal member: {raw_name!r}")
        if path.parts[0] != PLUGIN_ID:
            raise SmokeFailure(
                f"ZIP must contain only the {PLUGIN_ID}/ top-level directory: "
                f"{raw_name!r}"
            )
        unix_mode = member.external_attr >> 16
        if unix_mode and stat.S_ISLNK(unix_mode):
            raise SmokeFailure(f"ZIP symbolic links are not allowed: {raw_name!r}")
        folded = raw_name.rstrip("/").casefold()
        if folded in folded_names:
            raise SmokeFailure(f"Case-insensitive ZIP member collision: {raw_name!r}")
        folded_names.add(folded)
        expanded_bytes += int(member.file_size)
        if expanded_bytes > MAX_EXPANDED_BYTES:
            raise SmokeFailure("ZIP expands beyond the clean-profile smoke limit.")
        if path == PurePosixPath(PLUGIN_ID, "metadata.txt"):
            metadata_seen = True
        members.append(member)
    if not metadata_seen:
        raise SmokeFailure(f"ZIP is missing {PLUGIN_ID}/metadata.txt.")
    return members


def _archive_version(archive: zipfile.ZipFile) -> str:
    raw_metadata = archive.read(f"{PLUGIN_ID}/metadata.txt").decode("utf-8")
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read_string(raw_metadata)
    return parser.get("general", "version").strip()


def install_archive(
    archive_path: Path,
    profile_root: Path,
    *,
    expected_version: str = DEFAULT_EXPECTED_VERSION,
) -> tuple[Path, str]:
    """Install *archive_path* only into an empty, caller-owned profile root."""

    archive_path = archive_path.expanduser().resolve(strict=True)
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise SmokeFailure("Plugin ZIP exceeds the QGIS publication size limit.")
    requested_profile_root = profile_root.expanduser().absolute()
    if requested_profile_root.is_symlink():
        raise SmokeFailure(
            f"Refusing a symbolic-link profile root: {requested_profile_root}"
        )
    profile_root = requested_profile_root.resolve()
    if profile_root.exists() and not profile_root.is_dir():
        raise SmokeFailure(f"Profile root is not a directory: {profile_root}")
    if profile_root.exists() and any(profile_root.iterdir()):
        raise SmokeFailure(
            f"Refusing to alter a non-empty profile root: {profile_root}"
        )
    plugin_parent = (
        profile_root
        / "profiles"
        / PROFILE_NAME
        / "python"
        / "plugins"
    )
    with zipfile.ZipFile(archive_path) as archive:
        # Complete every no-write check before creating the profile tree.
        members = _archive_members(archive)
        archive_version = _archive_version(archive)
        if archive_version != expected_version:
            raise SmokeFailure(
                f"Expected metadata version {expected_version}, got "
                f"{archive_version}."
            )
        profile_root.mkdir(parents=True, exist_ok=True)
        plugin_parent.mkdir(parents=True, exist_ok=False)
        for member in members:
            relative = PurePosixPath(member.filename)
            destination = plugin_parent.joinpath(*relative.parts)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise SmokeFailure(
                    f"ZIP attempted to overwrite an installed member: {relative}"
                )
            with archive.open(member) as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)

    plugin_dir = plugin_parent / PLUGIN_ID
    if not (plugin_dir / "metadata.txt").is_file():
        raise SmokeFailure("Installed plugin metadata disappeared after extraction.")
    # QGIS 3.44 creates a random authentication master password in the OS
    # keychain for a brand-new profile.  The offscreen platform cannot answer
    # that native prompt, and a smoke must not write credentials to the user's
    # keychain in any case.  Disable the helper only in this disposable
    # profile before QgisApp is constructed.  The key is the public
    # QgsSettings::Auth value used by QgsAuthManager::passwordHelperEnabled().
    settings_file = (
        profile_root
        / "profiles"
        / PROFILE_NAME
        / "qgis.org"
        / "QGIS3.ini"
    )
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    settings_file.write_text(
        "[auth]\nuse_password_helper=false\n",
        encoding="utf-8",
    )
    return plugin_dir, archive_version


def _wait_until(predicate, description: str, timeout_seconds: float = 20.0) -> None:
    """Pump the QGIS event loop until an asynchronous invariant is true."""

    from qgis.core import QgsApplication

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        QgsApplication.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise SmokeFailure(f"Timed out waiting for {description}.")


def _make_synthetic_raster(path: Path) -> dict:
    """Create a georeferenced colored contour with realistic distractors."""

    import numpy as np
    from osgeo import gdal, osr

    width = height = 256
    pixels = np.full((height, width, 3), 238, dtype=np.uint8)
    # Faint grid/text-like interruptions should not beat the colored contour.
    pixels[::32, :, :] = 215
    pixels[:, ::48, :] = 220
    pixels[54:60, 96:132, :] = 190
    pixels[165:170, 145:205, :] = 195

    def curve_row(x_value):
        return 92.0 + 0.20 * x_value + 12.0 * np.sin(x_value / 24.0)

    for x_value in range(18, 235):
        center = int(round(curve_row(x_value)))
        for delta in range(-2, 3):
            row = center + delta
            if 0 <= row < height:
                # A blue historical-map contour: one channel remains dark,
                # so RGB-aware Ink v2 must retain it instead of using only
                # luminance thresholding.
                pixels[row, x_value, :] = (32, 74, 205)
        parallel = center + 22
        if 0 <= parallel < height:
            pixels[parallel, x_value, :] = (176, 180, 190)

    gdal.UseExceptions()
    dataset = gdal.GetDriverByName("GTiff").Create(
        os.fspath(path),
        width,
        height,
        3,
        gdal.GDT_Byte,
        options=("COMPRESS=DEFLATE",),
    )
    if dataset is None:
        raise SmokeFailure("GDAL could not create the synthetic raster.")
    dataset.SetGeoTransform((0.0, 1.0, 0.0, 256.0, 0.0, -1.0))
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(3857)
    dataset.SetProjection(spatial_reference.ExportToWkt())
    for band_index in range(3):
        dataset.GetRasterBand(band_index + 1).WriteArray(pixels[:, :, band_index])
    dataset.FlushCache()
    dataset = None

    start_x, end_x = 40.0, 180.0
    start_row = float(curve_row(start_x))
    end_row = float(curve_row(end_x))
    return {
        "size": (width, height),
        "start": (start_x, 256.0 - start_row),
        "end": (end_x, 256.0 - end_row),
        "curve_row": curve_row,
    }


def _qt_value(owner, legacy_name: str, scope_name: str):
    legacy = getattr(owner, legacy_name, None)
    if legacy is not None:
        return legacy
    return getattr(getattr(owner, scope_name), legacy_name)


def _run_trace_scenario(
    *,
    canvas,
    raster_layer,
    fixture: dict,
    name: str,
    edge_weight: float,
    freehand: bool,
    smart_recovery: bool,
    rapid_accept: bool = False,
    defer_during_cache: bool = False,
    enhanced_accept: bool = False,
    sample_color: bool = False,
    learned_engine=None,
    learned_guide: bool = False,
    learned_weights=None,
    grid_avoidance: bool = True,
    livewire_config_overrides=None,
    trace_diagnostics_dir=None,
) -> dict:
    """Drive real SmartTraceTool press/move/finish handlers to one feature."""

    import numpy as np
    from qgis.PyQt.QtCore import Qt
    from qgis.core import QgsPointXY, QgsProject

    from ai_vectorizer.core.edge_detector import EdgeDetector
    from ai_vectorizer.recovery import RECOVERY_STATE_INK_FALLBACK
    from ai_vectorizer.tools.smart_trace_tool import SmartTraceTool

    state_events: list[tuple[str, str]] = []
    tool = SmartTraceTool(
        canvas,
        raster_layer,
        None,
        edge_weight=edge_weight,
        freehand=freehand,
        edge_method=EdgeDetector.METHOD_INK,
        language="en",
        smart_recovery=smart_recovery,
        recovery_engine=None,
        learned_engine=learned_engine,
        learned_guide=learned_guide,
        learned_support_weight=learned_weights[0] if learned_weights is not None else None,
        learned_avoidance_weight=learned_weights[1] if learned_weights is not None else None,
        grid_avoidance_enabled=grid_avoidance,
        recovery_state_callback=lambda state, detail: state_events.append(
            (state, detail)
        ),
    )
    output_layer = tool.vector_layer
    if livewire_config_overrides:
        from dataclasses import replace

        current_config = tool._livewire_config()
        tool._livewire_config = lambda: replace(current_config, **livewire_config_overrides)
    canvas.setLayers([output_layer, raster_layer])

    left_button = _qt_value(Qt, "LeftButton", "MouseButton")
    right_button = _qt_value(Qt, "RightButton", "MouseButton")
    no_button = _qt_value(Qt, "NoButton", "MouseButton")
    no_modifier = _qt_value(Qt, "NoModifier", "KeyboardModifier")

    class Event:
        def __init__(self, position, *, button=no_button, buttons=no_button):
            self._position = position
            self._button = button
            self._buttons = buttons

        def pos(self):
            return self._position

        def button(self):
            return self._button

        def buttons(self):
            return self._buttons

        @staticmethod
        def modifiers():
            return no_modifier

    start_map = QgsPointXY(*fixture["start"])
    end_map = QgsPointXY(*fixture["end"])
    try:
        canvas.setMapTool(tool)
        prepare_input = fixture.get("prepare_input")
        if prepare_input is not None:
            prepare_input(tool)
        if freehand or edge_weight <= 0.0:
            if tool._needs_edge_cache():
                raise SmokeFailure(f"{name} unexpectedly requested raster evidence.")
            if tool.cached_edges is not None or tool.cached_ink_evidence is not None:
                raise SmokeFailure(f"{name} populated an edge/evidence cache.")
        elif not defer_during_cache:
            _wait_until(
                lambda: tool._ink_evidence_task is None
                and tool.cached_edges is not None and not tool.cache_dirty,
                f"{name} Ink cache",
            )
            if tool.cached_ink_evidence is None:
                raise SmokeFailure(f"{name} fell back instead of publishing Ink v2.")
            if learned_guide and learned_engine is not None:
                _wait_until(lambda: tool._learned_prior is not None and tool._learned_task is None, f"{name} learned prior", timeout_seconds=40)

        # Awaited event loops may finish a dock layout and resize the canvas.
        # Convert immediately before each gesture, as a real mouse event does.
        if prepare_input is not None:
            prepare_input(tool)
            if not freehand and edge_weight > 0.0 and not defer_during_cache:
                _wait_until(
                    lambda: tool._ink_evidence_task is None
                    and tool.cached_edges is not None and not tool.cache_dirty,
                    f"{name} pixel-aligned Ink cache",
                )
                if learned_guide and learned_engine is not None:
                    _wait_until(lambda: tool._learned_prior is not None and tool._learned_task is None,
                                f"{name} pixel-aligned learned prior", timeout_seconds=40)
        start_screen = tool.toCanvasCoordinates(start_map)
        actual_start_map = tool.toMapCoordinates(start_screen)
        tool.canvasPressEvent(Event(start_screen, button=left_button))
        if not tool.is_tracing:
            raise SmokeFailure(f"{name} did not begin tracing.")
        if (
            not freehand
            and edge_weight > 0.0
            and not rapid_accept
            and not defer_during_cache
        ):
            _wait_until(
                lambda: tool._livewire_task is None
                and tool._livewire_tree is not None,
                f"{name} LiveWire tree",
            )

        sampled_color = None
        if sample_color:
            from types import SimpleNamespace

            sample_x = 70.0
            sample_map = QgsPointXY(sample_x, 256.0 - float(fixture["curve_row"](sample_x)))
            tool.canvasMoveEvent(Event(tool.toCanvasCoordinates(sample_map)))
            tool.keyPressEvent(SimpleNamespace(
                key=lambda: _qt_value(Qt, "Key_C", "Key"),
                modifiers=lambda: no_modifier,
                accept=lambda: None,
            ))
            if tool._ink_color_profile is None:
                raise SmokeFailure(f"{name} did not sample the native RGB contour.")
            sampled_color = tool._ink_color_profile.sample_rgb
            if tool.cached_trace_guidance is None:
                raise SmokeFailure(f"{name} did not apply color mismatch guidance.")
            _wait_until(
                lambda: tool._livewire_task is None and tool._livewire_tree is not None,
                f"{name} color-guided LiveWire tree",
            )

        validate_input = fixture.get("validate_input")
        if validate_input is not None:
            validate_input(tool)
        end_screen = tool.toCanvasCoordinates(end_map)
        actual_end_map = tool.toMapCoordinates(end_screen)
        tool.canvasMoveEvent(Event(end_screen))
        route_provenance = None
        route_diagnostics = None
        tree = tool._livewire_tree
        if tree is not None and hasattr(tree, "trace_with_provenance"):
            cursor = tool.last_hover_pos or end_map
            cursor_pixel = tool.map_to_pixel_float(cursor)
            _route, route_provenance = tree.trace_with_provenance(cursor_pixel)
            if trace_diagnostics_dir is not None:
                connection = tree.occluded_connection
                candidate = connection.trace(cursor_pixel) if connection is not None else None
                route_diagnostics = {
                    "root": tree.root, "origin": tree.origin, "shape": tree.shape,
                    "cursor_pixel": cursor_pixel,
                    "occluded_connection": connection is not None,
                    "candidate": candidate,
                    "ordinary_intersects": connection.intersects(tree._trace_without_occlusion(cursor_pixel)) if connection is not None else False,
                    "outside_route_available": connection.outside_route(cursor_pixel) is not None if connection is not None else False,
                    "display_accepted": connection.accepts_display(candidate) if candidate is not None else False,
                }
                text_connection = getattr(tree, "text_occluded_connection", None)
                text_candidate = text_connection.trace(cursor_pixel) if text_connection is not None else None
                route_diagnostics["text_occlusion"] = {
                    "available": text_connection is not None,
                    "candidate": text_candidate,
                    "outside_route_available": text_connection.outside_route(cursor_pixel) is not None if text_connection is not None else False,
                    "display_accepted": text_connection.accepts_display(text_candidate) if text_candidate is not None else False,
                }
                diagnostic_path = Path(trace_diagnostics_dir)
                diagnostic_path.mkdir(parents=True, exist_ok=True)
                evidence = tool.cached_ink_evidence
                np.savez_compressed(
                    diagnostic_path / (name + ".npz"),
                    image=tool.cached_rgb_image, edges=tool.cached_edges,
                    centerline=evidence.centerline, center_score=evidence.center_score,
                    tangent_x=evidence.tangent_x, tangent_y=evidence.tangent_y,
                    coherence=evidence.coherence,
                    root=tree.root, origin=tree.origin, target=cursor_pixel,
                )
        if not tool.preview_path:
            raise SmokeFailure(f"{name} did not display a cursor preview.")
        if enhanced_accept:
            from ai_vectorizer.recovery import RECOVERY_STATE_ENHANCED

            enhanced_points = []
            for x_value in np.linspace(
                fixture["start"][0],
                fixture["end"][0],
                10,
            )[1:]:
                row = float(fixture["curve_row"](float(x_value)))
                enhanced_points.append(
                    QgsPointXY(float(x_value), 256.0 - row)
                )
            tool.preview_path = enhanced_points
            tool._livewire_request_point = QgsPointXY(end_map)
            tool._current_recovery_state = RECOVERY_STATE_ENHANCED
            tool._render_preview()
        preview = [QgsPointXY(point) for point in tool.preview_path]
        if (
            not freehand
            and edge_weight > 0.0
            and not rapid_accept
            and not defer_during_cache
            and len(preview) < 8
        ):
            raise SmokeFailure(
                f"{name} did not use a traced centerline (only {len(preview)} points)."
            )

        tool.canvasPressEvent(Event(end_screen, button=left_button))
        if rapid_accept or defer_during_cache:
            if tool._pending_livewire_accept_point is None:
                raise SmokeFailure(
                    f"{name} committed the temporary preview before LiveWire was ready."
                )
            queued_target = QgsPointXY(tool._pending_livewire_accept_point)
            if not tool._pending_livewire_auto_accept:
                raise SmokeFailure(f"{name} lost its deferred acceptance policy.")
            if len(tool.path_points) != 1:
                raise SmokeFailure(
                    f"{name} added a straight chord while the tree was pending."
                )
            # A third rapid click must not overwrite the first queued target.
            extra_x = 210.0
            extra_row = float(fixture["curve_row"](extra_x))
            extra_map = QgsPointXY(extra_x, 256.0 - extra_row)
            extra_screen = tool.toCanvasCoordinates(extra_map)
            tool.canvasPressEvent(Event(extra_screen, button=left_button))
            queued_after_extra = tool._pending_livewire_accept_point
            if (
                queued_after_extra is None
                or queued_after_extra.x() != queued_target.x()
                or queued_after_extra.y() != queued_target.y()
            ):
                raise SmokeFailure(
                    f"{name} let a later rapid click overwrite the queued anchor."
                )
            _wait_until(
                lambda: tool._pending_livewire_accept_point is None
                and len(tool.path_points) > 1,
                f"{name} deferred LiveWire acceptance",
            )
        if len(tool.path_points) < 2:
            raise SmokeFailure(f"{name} did not accept the visible preview.")
        accepted = [QgsPointXY(point) for point in tool.path_points]
        if enhanced_accept or (learned_guide and edge_weight > 0):
            expected = [accepted[0], *preview]
            if accepted != expected:
                raise SmokeFailure(
                    f"{name} replaced the visible Enhanced path while accepting it."
                )
        if not freehand and edge_weight > 0.0 and len(accepted) < 8:
            raise SmokeFailure(
                f"{name} accepted a sparse chord (only {len(accepted)} points)."
            )
        grid_prior = tool.cached_map_grid
        grid_summary = None
        if grid_prior is not None:
            grid_summary = {
                "horizontal_pixels": int(np.count_nonzero(grid_prior.horizontal_score)),
                "vertical_pixels": int(np.count_nonzero(grid_prior.vertical_score)),
            }
        tool.ask_elevation = lambda: 100.0
        tool.canvasPressEvent(Event(end_screen, button=right_button))
        if output_layer.featureCount() != 1:
            raise SmokeFailure(f"{name} did not save exactly one feature.")

        feature = next(output_layer.getFeatures())
        vertices = feature.geometry().asPolyline()
        if len(vertices) != len(accepted):
            raise SmokeFailure(f"{name} saved geometry differs from the accepted preview.")
        if [(p.x(), p.y()) for p in vertices] != [(p.x(), p.y()) for p in accepted]:
            raise SmokeFailure(f"{name} changed accepted coordinates while saving.")
        if sample_color:
            if tuple((point.x(), point.y()) for point in vertices) != tuple(
                (point.x(), point.y()) for point in accepted
            ):
                raise SmokeFailure(f"{name} changed the accepted color-guided geometry.")
            if tool._ink_color_profile is not None or tool.cached_trace_guidance is not None:
                raise SmokeFailure(f"{name} retained color guidance after finishing the contour.")
        start_error = float(
            np.hypot(vertices[0].x() - start_map.x(), vertices[0].y() - start_map.y())
        )
        end_error = float(
            np.hypot(vertices[-1].x() - end_map.x(), vertices[-1].y() - end_map.y())
        )
        endpoint_tolerance = fixture.get("endpoint_tolerance_px", 6.1 if learned_guide and learned_engine is not None else 1.5)
        if start_error > endpoint_tolerance or end_error > endpoint_tolerance:
            raise SmokeFailure(
                f"{name} changed trace endpoints ({start_error:.3f}, {end_error:.3f})."
            )

        median_center_error = None
        if not freehand and edge_weight > 0.0 and fixture.get("curve_row") is not None:
            errors = []
            curve_row = fixture["curve_row"]
            for point in vertices:
                source_row = 256.0 - point.y()
                errors.append(abs(source_row - float(curve_row(point.x()))))
            median_center_error = float(np.median(np.asarray(errors)))
            if median_center_error > 4.0 and not learned_guide:
                raise SmokeFailure(
                    f"{name} left the synthetic centerline (median error "
                    f"{median_center_error:.3f}px)."
                )

        if smart_recovery:
            if tool.smart_recovery_enabled:
                raise SmokeFailure("A missing recovery model became enabled.")
            if not state_events or state_events[-1][0] != RECOVERY_STATE_INK_FALLBACK:
                raise SmokeFailure("Missing-model recovery did not report Ink fallback.")
        elif tool._recovery_task is not None:
            raise SmokeFailure("Smart Recovery OFF scheduled a recovery task.")

        output_layer.undoStack().undo()
        if output_layer.featureCount() != 0:
            raise SmokeFailure(f"{name} did not undo the saved contour in one command.")
        output_layer.undoStack().redo()
        if output_layer.featureCount() != 1:
            raise SmokeFailure(f"{name} could not redo the contour.")
        if learned_guide and edge_weight <= 0 and (tool._learned_task is not None or tool._learned_prior is not None):
            raise SmokeFailure("0% assistance ran learned inference.")

        return {
            "name": name,
            "input_map_xy": {
                "start": [actual_start_map.x(), actual_start_map.y()],
                "end": [actual_end_map.x(), actual_end_map.y()],
            },
            "input_screen_xy": {
                "start": [start_screen.x(), start_screen.y()],
                "end": [end_screen.x(), end_screen.y()],
            },
            "vertices": len(vertices),
            "vertices_xy": [(point.x(), point.y()) for point in vertices],
            "route_provenance": route_provenance,
            "route_diagnostics": route_diagnostics,
            "start_error_px": round(start_error, 4),
            "end_error_px": round(end_error, 4),
            "median_center_error_px": (round(median_center_error, 4)
                                       if median_center_error is not None else None),
            "ink_v2": tool.cached_ink_evidence is not None,
            "smart_recovery_enabled": bool(tool.smart_recovery_enabled),
            "rapid_accept_deferred": bool(rapid_accept),
            "cache_accept_deferred": bool(defer_during_cache),
            "enhanced_preview_preserved": bool(enhanced_accept),
            "sampled_ink_color_rgb": sampled_color,
            "learned_guide_requested": learned_guide,
            "learned_support_weight": tool.learned_support_weight,
            "learned_avoidance_weight": tool.learned_avoidance_weight,
            "grid_avoidance": grid_avoidance,
            "detected_grid": grid_summary,
            "learned_preview_preserved": bool(learned_guide and edge_weight > 0),
            "one_undo": True,
            "states": [state for state, _detail in state_events],
        }
    finally:
        if canvas.mapTool() is tool:
            canvas.unsetMapTool(tool)
        tool.dispose()
        tool.deleteLater()
        # SmartTraceTool deliberately keeps user edits uncommitted so QGIS'
        # ordinary save/discard workflow remains authoritative.  This
        # disposable smoke has already inspected the saved feature, so close
        # its private edit buffer explicitly before removing the layer.  That
        # avoids an offscreen QGIS save/discard dialog without weakening the
        # product behavior under test.
        if output_layer.isEditable():
            output_layer.rollBack()
        QgsProject.instance().removeMapLayer(output_layer.id())


def run_qgis_bootstrap() -> None:
    """Execute inside QGIS after ``--code`` loads this source file."""

    result_path = Path(os.environ[RESULT_PATH_ENV]).resolve()
    result: dict = {"ok": False}
    try:
        from qgis.core import (
            Qgis,
            QgsApplication,
            QgsCoordinateReferenceSystem,
            QgsProject,
            QgsRasterLayer,
        )
        import qgis.utils

        profile_root = Path(os.environ[PROFILE_ROOT_ENV]).resolve(strict=True)
        expected_plugin_dir = Path(os.environ[PLUGIN_DIR_ENV]).resolve(strict=True)
        expected_settings_dir = (
            profile_root / "profiles" / PROFILE_NAME
        ).resolve(strict=True)
        active_settings_dir = Path(QgsApplication.qgisSettingsDirPath()).resolve()
        try:
            active_settings_dir.relative_to(expected_settings_dir)
        except ValueError as exc:
            raise SmokeFailure(
                f"QGIS escaped the isolated profile: {active_settings_dir}"
            ) from exc

        # Use QGIS' own plugin loader and live iface.  The noplugins launch
        # flag keeps every unrelated provider plugin disabled; this explicit
        # start is the only third-party plugin activation in the process.
        if not qgis.utils.loadPlugin(PLUGIN_ID):
            raise SmokeFailure("QGIS loadPlugin rejected the installed ZIP.")
        if not qgis.utils.startPlugin(PLUGIN_ID):
            raise SmokeFailure("QGIS startPlugin rejected the installed ZIP.")
        plugin = qgis.utils.plugins.get(PLUGIN_ID)
        if plugin is None:
            raise SmokeFailure("QGIS did not retain the started plugin instance.")

        import ai_vectorizer

        loaded_package = Path(ai_vectorizer.__file__).resolve()
        try:
            loaded_package.relative_to(expected_plugin_dir)
        except ValueError as exc:
            raise SmokeFailure(
                f"QGIS imported a non-profile plugin copy: {loaded_package}"
            ) from exc
        if len(plugin.actions) != 1 or plugin.toolbar is None:
            raise SmokeFailure("Plugin GUI action/toolbar did not initialize.")
        installed_metadata = configparser.ConfigParser()
        installed_metadata.read(
            expected_plugin_dir / "metadata.txt",
            encoding="utf-8",
        )
        installed_version = installed_metadata.get("general", "version").strip()

        plugin.run()
        dock = plugin.dialog
        if dock is None or dock.smart_recovery_check.isChecked() or dock.learned_guide_check.isChecked():
            raise SmokeFailure("Clean-profile Smart Recovery did not default OFF.")
        if dock.recovery_prepare_task is not None or dock.recovery_install_task is not None:
            raise SmokeFailure("Default-OFF UI started model preparation/download work.")
        if dock.recovery_engine is not None:
            raise SmokeFailure("Default-OFF UI retained a recovery engine.")
        if dock.learned_prepare_task is not None or dock.learned_engine is not None:
            raise SmokeFailure("Default-OFF UI prepared a learned model.")
        models_dir = Path(dock._sam_models_dir()).resolve()
        try:
            models_dir.relative_to(expected_settings_dir)
        except ValueError as exc:
            raise SmokeFailure(f"Model storage escaped the clean profile: {models_dir}") from exc

        raster_path = profile_root / "synthetic-colored-contour.tif"
        fixture = _make_synthetic_raster(raster_path)
        raster_layer = QgsRasterLayer(os.fspath(raster_path), "clean-profile raster")
        if not raster_layer.isValid():
            raise SmokeFailure("QGIS could not load the synthetic GeoTIFF.")
        project = QgsProject.instance()
        project.setCrs(QgsCoordinateReferenceSystem("EPSG:3857"))
        project.addMapLayer(raster_layer)
        canvas = qgis.utils.iface.mapCanvas()
        qgis.utils.iface.mainWindow().resize(960, 760)
        canvas.setDestinationCrs(raster_layer.crs())
        canvas.setLayers([raster_layer])
        canvas.setExtent(raster_layer.extent())
        canvas.refresh()
        _wait_until(lambda: not canvas.isDrawing(), "initial raster render")

        scenarios = [
            _run_trace_scenario(
                canvas=canvas,
                raster_layer=raster_layer,
                fixture=fixture,
                name="ink-v2-explicit-color-sample",
                edge_weight=1.0,
                freehand=False,
                smart_recovery=False,
                sample_color=True,
            ),
            _run_trace_scenario(
                canvas=canvas,
                raster_layer=raster_layer,
                fixture=fixture,
                name="freehand",
                edge_weight=0.75,
                freehand=True,
                smart_recovery=False,
            ),
            _run_trace_scenario(
                canvas=canvas,
                raster_layer=raster_layer,
                fixture=fixture,
                name="exact-cursor-0-percent-missing-recovery",
                edge_weight=0.0,
                freehand=False,
                smart_recovery=True,
            ),
            _run_trace_scenario(
                canvas=canvas,
                raster_layer=raster_layer,
                fixture=fixture,
                name="ink-v2-smart-recovery-off",
                edge_weight=0.75,
                freehand=False,
                smart_recovery=False,
            ),
            _run_trace_scenario(
                canvas=canvas,
                raster_layer=raster_layer,
                fixture=fixture,
                name="ink-v2-rapid-click",
                edge_weight=1.0,
                freehand=False,
                smart_recovery=False,
                rapid_accept=True,
            ),
            _run_trace_scenario(
                canvas=canvas,
                raster_layer=raster_layer,
                fixture=fixture,
                name="ink-v2-click-during-cache",
                edge_weight=1.0,
                freehand=False,
                smart_recovery=False,
                defer_during_cache=True,
            ),
            _run_trace_scenario(
                canvas=canvas,
                raster_layer=raster_layer,
                fixture=fixture,
                name="ink-v2-enhanced-wysiwyg",
                edge_weight=1.0,
                freehand=False,
                smart_recovery=False,
                enhanced_accept=True,
            ),
        ]

        bundle_path = os.environ.get("ARCHAEOTRACE_SMOKE_LEARNED_BUNDLE")
        if bundle_path:
            # Preserve QGIS' NumPy ABI before finding the separately installed
            # optional ONNX packages in the diagnostic environment.
            import numpy
            dependency_path = os.environ.get("ARCHAEOTRACE_SMOKE_LEARNED_DEPENDENCIES")
            if dependency_path:
                sys.path.append(dependency_path)
            from ai_vectorizer.core.learned_model import LearnedContourEngine
            def peak_rss():
                try:
                    import resource
                    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    return int(raw if sys.platform == "darwin" else raw * 1024)
                except ImportError:
                    return None
            peak_before = peak_rss()
            learned_engine = LearnedContourEngine.from_bundle(bundle_path)
            tile = numpy.full((512, 512, 3), 255, dtype=numpy.uint8)
            learned_engine.predict_tile(tile)
            tile_times = []
            for _ in range(25):
                started = time.perf_counter()
                learned_engine.predict_tile(tile)
                tile_times.append((time.perf_counter() - started) * 1000)
            learned_diagnostic = {
                "synthetic_model": bool(learned_engine.manifest["provenance"].get("synthetic_diagnostic")),
                "historical_map_accuracy_evidence": False,
                "input": "synthetic_solid_rgb", "warmup_iterations": 1,
                "prepared_iterations": 25, "prepared_tile_ms": tile_times,
                "prepared_tile_p95_ms": sorted(tile_times)[23],
                "peak_rss_growth_bytes": max(0, peak_rss() - peak_before) if peak_before is not None else None,
                "paired_holdout_ram_measured": False,
                "runtime": learned_engine.metadata,
            }
            scenarios.append(_run_trace_scenario(canvas=canvas, raster_layer=raster_layer,
                fixture=fixture, name="learned-guide-native", edge_weight=1.0, freehand=False,
                smart_recovery=False, learned_engine=learned_engine, learned_guide=True))
            scenarios.append(_run_trace_scenario(canvas=canvas, raster_layer=raster_layer,
                fixture=fixture, name="learned-guide-zero-assist", edge_weight=0.0, freehand=False,
                smart_recovery=False, learned_engine=learned_engine, learned_guide=True))
            # The guided lifecycle may legitimately leave the synthetic blue
            # line (diagnostic weights are random; real weights may have
            # learned map colour semantics), so record the miss instead of
            # letting "ok" imply the guide followed it.
            learned_error = scenarios[-2]["median_center_error_px"]
            learned_diagnostic["synthetic_centerline_error_px"] = learned_error
            learned_diagnostic["synthetic_centerline_followed"] = learned_error <= 4.0

        # The opt-in recovery bundle must remain absent.  The dock can migrate
        # its small, GPL-shipped HED network definition into persistent
        # storage; that is a legacy configuration asset, not either 41 MB
        # EfficientSAM weight.  Inspect the exact content-addressed recovery
        # contract so this assertion cannot confuse the two model families.
        from ai_vectorizer.core.efficientsam_recovery import (
            EfficientSAMRecoveryEngine,
        )

        recovery_status = EfficientSAMRecoveryEngine.inspect(models_dir)
        recovery_states = {
            artifact.spec.identifier: artifact.state
            for artifact in recovery_status.artifacts
        }
        if recovery_status.ready or any(
            state != "missing" for state in recovery_states.values()
        ):
            raise SmokeFailure(
                "Recovery weights appeared without install consent: "
                f"{recovery_states}"
            )

        result = {
            "ok": True,
            "qgis_version": Qgis.QGIS_VERSION,
            "profile_settings_dir": str(active_settings_dir),
            "loaded_plugin": str(loaded_package),
            "installed_metadata_version": installed_version,
            "smart_recovery_default_off": True,
            "recovery_model_states": recovery_states,
            "scenarios": scenarios,
            "learned_diagnostic": learned_diagnostic if bundle_path else None,
            "warnings": (
                ["learned-guide-native left the synthetic centerline "
                 f"(median {learned_diagnostic['synthetic_centerline_error_px']:.1f}px)"]
                if bundle_path and not learned_diagnostic["synthetic_centerline_followed"]
                else []
            ),
        }
        qgis.utils.unloadPlugin(PLUGIN_ID)
        project.clear()
    except BaseException as exc:  # QGIS must still return a machine-readable result.
        result = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        temporary = result_path.with_suffix(result_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, result_path)
        try:
            from qgis.PyQt.QtCore import QCoreApplication, QTimer

            QTimer.singleShot(0, QCoreApplication.quit)
        except Exception:
            pass


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="archive", type=Path, required=True)
    parser.add_argument(
        "--qgis-executable",
        type=Path,
        default=Path("/Applications/QGIS.app/Contents/MacOS/QGIS"),
        help="Real QGIS desktop executable (macOS default shown).",
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        help="Optional empty diagnostic root; omission uses and removes mkdtemp.",
    )
    parser.add_argument("--expected-version", default=DEFAULT_EXPECTED_VERSION)
    parser.add_argument("--learned-bundle", type=Path, help="Optional local diagnostic learned model bundle")
    parser.add_argument("--learned-dependencies", type=Path, help="Optional ONNX site-packages path; QGIS NumPy stays first")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Maximum QGIS process runtime in seconds.",
    )
    return parser.parse_args(argv)


def run_outer(args: argparse.Namespace, *, bootstrap_script=None) -> dict:
    qgis_executable = args.qgis_executable.expanduser().resolve(strict=True)
    if not os.access(qgis_executable, os.X_OK):
        raise SmokeFailure(f"QGIS executable is not runnable: {qgis_executable}")
    if args.timeout <= 0 or args.timeout > 600:
        raise SmokeFailure("Timeout must be between 1 and 600 seconds.")

    owned_temporary = None
    if args.profile_root is None:
        owned_temporary = tempfile.TemporaryDirectory(
            prefix="archaeotrace-clean-profile-"
        )
        profile_root = Path(owned_temporary.name)
    else:
        profile_root = args.profile_root
    try:
        plugin_dir, version = install_archive(
            args.archive,
            profile_root,
            expected_version=args.expected_version,
        )
        profile_root = profile_root.expanduser().resolve(strict=True)
        result_path = profile_root / "clean-profile-result.json"
        environment = os.environ.copy()
        if getattr(args, "learned_bundle", None):
            environment["ARCHAEOTRACE_SMOKE_LEARNED_BUNDLE"] = str(args.learned_bundle.resolve(strict=True))
        if getattr(args, "learned_dependencies", None):
            environment["ARCHAEOTRACE_SMOKE_LEARNED_DEPENDENCIES"] = str(args.learned_dependencies.resolve(strict=True))
        environment.update(
            {
                BOOTSTRAP_ENV: "1",
                PROFILE_ROOT_ENV: str(profile_root),
                PLUGIN_DIR_ENV: str(plugin_dir),
                RESULT_PATH_ENV: str(result_path),
                "QT_QPA_PLATFORM": "offscreen",
            }
        )
        command = [
            str(qgis_executable),
            "--nologo",
            "--noversioncheck",
            "--noplugins",
            "--defaultui",
            "--hide-browser",
            "--profiles-path",
            str(profile_root),
            "--profile",
            PROFILE_NAME,
            "--code",
            str(Path(bootstrap_script or __file__).resolve()),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=profile_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"")
            stderr = (exc.stderr or b"")
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            raise SmokeFailure(
                f"QGIS clean-profile smoke timed out after {args.timeout}s.\n"
                f"stdout={stdout[-4000:]}\nstderr={stderr[-4000:]}"
            ) from exc
        if not result_path.is_file():
            raise SmokeFailure(
                "QGIS produced no clean-profile result.\n"
                f"exit={completed.returncode}\nstdout={completed.stdout[-4000:]}\n"
                f"stderr={completed.stderr[-4000:]}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result.update(
            {
                "archive": str(args.archive.expanduser().resolve(strict=True)),
                "archive_sha256": sha256_file(args.archive.expanduser().resolve(strict=True)),
                "archive_metadata_version": version,
                "qgis_exit_code": completed.returncode,
                "isolated_profile_removed_after_run": owned_temporary is not None,
            }
        )
        if completed.returncode != 0:
            raise SmokeFailure(
                f"QGIS exited with {completed.returncode}: {completed.stderr[-4000:]}"
            )
        if not result.get("ok"):
            raise SmokeFailure(
                "Clean-profile bootstrap failed:\n"
                + json.dumps(result, indent=2, sort_keys=True)
            )
        return result
    finally:
        if owned_temporary is not None:
            owned_temporary.cleanup()


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = run_outer(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if os.environ.get(BOOTSTRAP_ENV) == "1":
    # ``--code`` runs while QgisApp's constructor is still returning.  Doing
    # plugin teardown and application quit inline there can make Qt destroy
    # GUI objects in partially-constructed order.  Defer the complete smoke
    # to the first normal event-loop turn instead.
    from qgis.PyQt.QtCore import QTimer

    QTimer.singleShot(0, run_qgis_bootstrap)
elif __name__ == "__main__":
    raise SystemExit(main())
