"""Runtime QGIS regressions for edit and plugin lifecycle safety.

The suite skips cleanly in ordinary Python and runs in the QGIS CI image (or
with QGIS' bundled Python locally).
"""

from pathlib import Path
import gc
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


try:
    from qgis.core import (
        QgsApplication,
        QgsCoordinateReferenceSystem,
        QgsCoordinateTransform,
        QgsDefaultValue,
        QgsFeature,
        QgsField,
        QgsFieldConstraints,
        QgsGeometry,
        QgsPointXY,
        QgsProject,
        QgsRectangle,
        QgsRasterLayer,
        QgsVectorLayer,
    )
    from qgis.gui import QgsMapCanvas
    from qgis.PyQt.QtCore import QEventLoop, QMetaType, QTimer, QUrl
    try:
        from qgis.PyQt.QtCore import QVariant
    except ImportError:
        QVariant = None
    from qgis.PyQt.QtWidgets import QMainWindow, QToolBar

    HAS_QGIS = True
except ImportError as exc:
    if os.environ.get("ARCHAEOTRACE_REQUIRE_QGIS") == "1":
        raise RuntimeError(
            "QGIS bindings are required for this runtime safety job"
        ) from exc
    HAS_QGIS = False


def _field_type(name):
    if QVariant is not None and hasattr(QVariant, name):
        return getattr(QVariant, name)
    meta_name = "QString" if name == "String" else name
    meta_types = getattr(QMetaType, "Type", None)
    if meta_types is not None and hasattr(meta_types, meta_name):
        return getattr(meta_types, meta_name)
    raise RuntimeError(name)


def _field_constraint(name):
    legacy = getattr(QgsFieldConstraints, name, None)
    if legacy is not None:
        return legacy
    return getattr(QgsFieldConstraints.Constraint, name)


def _constraint_strength(name):
    legacy = getattr(QgsFieldConstraints, name, None)
    if legacy is not None:
        return legacy
    return getattr(QgsFieldConstraints.ConstraintStrength, name)


@unittest.skipUnless(HAS_QGIS, "QGIS Python bindings are not installed")
class QgisRuntimeSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._owns_application = QgsApplication.instance() is None
        cls.application = QgsApplication.instance() or QgsApplication([], False)
        cls._temporary_directories = []
        if cls._owns_application:
            cls.application.initQgis()
        from ai_vectorizer.plugin import AIVectorizer
        from ai_vectorizer.tools.smart_trace_tool import SmartTraceTool

        cls.AIVectorizer = AIVectorizer
        cls.SmartTraceTool = SmartTraceTool

    @classmethod
    def tearDownClass(cls):
        QgsProject.instance().clear()
        if cls._owns_application:
            cls.application.exitQgis()
        for temporary_directory in cls._temporary_directories:
            temporary_directory.cleanup()

    def setUp(self):
        QgsProject.instance().clear()
        self.crs = QgsCoordinateReferenceSystem("EPSG:3857")
        self.canvas = QgsMapCanvas()
        self.canvas.setDestinationCrs(self.crs)

    def test_qgsfield_type_resolvers_match_the_active_qt_binding(self):
        from ai_vectorizer.tools.smart_trace_tool import (
            _field_type as tool_field_type,
        )
        from ai_vectorizer.ui.main_dialog import (
            _field_type as dialog_field_type,
        )

        for resolver in (_field_type, tool_field_type, dialog_field_type):
            with self.subTest(resolver=resolver.__module__):
                integer_field = QgsField("identifier", resolver("Int"))
                double_field = QgsField("elevation", resolver("Double"))
                self.assertEqual(integer_field.name(), "identifier")
                self.assertEqual(double_field.name(), "elevation")
                if QVariant is not None and hasattr(QVariant, "Double"):
                    self.assertEqual(resolver("Double"), QVariant.Double)

    def _tool(self, layer, raster_crs=None):
        crs = raster_crs or self.crs

        class Raster:
            def crs(self):
                return crs

        return self.SmartTraceTool(
            self.canvas,
            Raster(),
            layer,
            freehand=True,
            language="en",
        )

    def test_ink_v2_sampling_preserves_source_pixel_grid_or_falls_back(self):
        class Raster:
            @staticmethod
            def width():
                return 5000

            @staticmethod
            def height():
                return 3000

        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        tool.raster_layer = Raster()
        raster_extent = QgsRectangle(0, 0, 500, 300)
        visible_extent = QgsRectangle(100, 140, 150, 170)

        read_extent, tile_origin = tool._ink_evidence_extent_and_origin(
            visible_extent,
            raster_extent,
        )
        native_plan = tool._ink_evidence_sampling_plan(
            read_extent,
            raster_extent,
        )
        self.assertEqual(tile_origin, (865, 1249))
        self.assertEqual(native_plan[:2], ((702, 446), True))

        wide_plan = tool._ink_evidence_sampling_plan(
            raster_extent,
            raster_extent,
        )
        self.assertEqual(wide_plan[0], (1000, 600))
        self.assertFalse(wide_plan[1])
        self.assertIn("Zoom in", wide_plan[2])

    def test_ink_v2_pan_within_same_source_tiles_reuses_exact_context(self):
        class Raster:
            @staticmethod
            def width():
                return 512

            @staticmethod
            def height():
                return 512

        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        tool.raster_layer = Raster()
        raster_extent = QgsRectangle(0, 0, 512, 512)
        first_visible = QgsRectangle(140, 300, 200, 360)
        second_visible = QgsRectangle(170, 310, 230, 370)

        first_extent, first_origin = tool._ink_evidence_extent_and_origin(
            first_visible,
            raster_extent,
        )
        second_extent, second_origin = tool._ink_evidence_extent_and_origin(
            second_visible,
            raster_extent,
        )

        self.assertEqual(first_origin, (97, 97))
        self.assertEqual(second_origin, first_origin)
        self.assertEqual(first_extent, QgsRectangle(97, 225, 287, 415))
        self.assertEqual(second_extent, first_extent)
        self.assertEqual(
            tool._ink_evidence_sampling_plan(first_extent, raster_extent)[:2],
            ((190, 190), True),
        )

    def test_ink_cache_update_publishes_v2_or_exact_visible_v1_fallback(self):
        """Exercise the real task-construction path, including both snapshots."""

        import numpy as np

        from ai_vectorizer.core.edge_detector import EdgeDetector

        class Provider:
            @staticmethod
            def dataSourceUri():
                return "memory://ink-source"

        class Raster:
            def __init__(self, crs):
                self._crs = crs
                self._provider = Provider()

            def crs(self):
                return self._crs

            @staticmethod
            def extent():
                return QgsRectangle(0, 0, 512, 512)

            @staticmethod
            def width():
                return 512

            @staticmethod
            def height():
                return 512

            @staticmethod
            def id():
                return "ink-source"

            @staticmethod
            def isValid():
                return True

            def dataProvider(self):
                return self._provider

        raster = Raster(self.crs)
        output = QgsVectorLayer(
            "LineString?crs=EPSG:3857",
            "ink output",
            "memory",
        )
        self.canvas.setExtent(QgsRectangle(140, 300, 200, 360))
        tool = self.SmartTraceTool(
            self.canvas,
            raster,
            output,
            edge_weight=0.5,
            edge_method=EdgeDetector.METHOD_INK,
            language="en",
        )

        class ImmediateTaskManager:
            @staticmethod
            def addTask(task):
                result = bool(task.run())
                task.finished(result)
                return True

        def wait_for_ink_task():
            self.assertIsNone(tool._ink_evidence_task)

        def display_bands(_provider, _extent, width, height, **_kwargs):
            return [np.full((height, width), 220, dtype=np.uint8)]

        def native_bands(_provider, _extent, width, height, **_kwargs):
            display = np.full((height, width), 220, dtype=np.uint8)
            native = np.full((height, width), 22_000, dtype=np.uint16)
            return [display], [native], True

        try:
            with mock.patch.object(
                QgsApplication,
                "taskManager",
                return_value=ImmediateTaskManager(),
            ), mock.patch(
                "ai_vectorizer.tools.smart_trace_tool.read_raster_bands",
                side_effect=display_bands,
            ), mock.patch(
                "ai_vectorizer.tools.smart_trace_tool."
                "read_raster_bands_with_native",
                side_effect=native_bands,
            ):
                self.canvas.setMapTool(tool)
                wait_for_ink_task()
                self.assertIsNotNone(tool.cached_ink_evidence)
                self.assertEqual(
                    tool.cached_edges.shape,
                    tool.cached_ink_evidence.shape,
                )
                self.assertEqual(
                    tool.cached_rgb_image.shape[:2],
                    tool.cached_edges.shape,
                )

                class FailingEvidenceDetector:
                    @staticmethod
                    def detect_edges(fallback_image):
                        return np.zeros(
                            np.asarray(fallback_image).shape[:2],
                            dtype=np.uint8,
                        )

                    @staticmethod
                    def detect_ink_evidence(*_args, **_kwargs):
                        raise RuntimeError("deliberate v2 failure")

                    @staticmethod
                    def get_edge_cost_map(edges, _weight):
                        return np.ones(edges.shape, dtype=np.float32)

                expected_visible = tool._canvas_extent_in_raster_crs().intersect(
                    raster.extent()
                )
                tool.edge_detector = FailingEvidenceDetector()
                tool.cache_dirty = True
                tool.update_edge_cache()
                wait_for_ink_task()

                self.assertIsNone(tool.cached_ink_evidence)
                self.assertEqual(tool.cache_tile_origin, (0, 0))
                self.assertEqual(tool.cache_extent, expected_visible)
                self.assertEqual(
                    tool.cached_edges.shape,
                    (
                        int(tool.cache_transform["height"]),
                        int(tool.cache_transform["width"]),
                    ),
                )
        finally:
            if self.canvas.mapTool() is tool:
                self.canvas.unsetMapTool(tool)
            tool.dispose()
            tool.deleteLater()
            QgsApplication.processEvents()

    def test_disabled_v2_publishes_exact_visible_extent_v1_fallback(self):
        import numpy as np

        from ai_vectorizer.core.edge_detector import EdgeDetector
        from ai_vectorizer.tools.smart_trace_tool import _InkEvidenceTask

        visible = np.full((64, 80), 235, dtype=np.uint8)
        visible[8:56, 38:43] = 25
        visible_rgb = np.repeat(visible[..., None], 3, axis=2)
        expanded = np.full((96, 112), 180, dtype=np.uint8)
        expanded_rgb = np.repeat(expanded[..., None], 3, axis=2)
        detector = EdgeDetector(method=EdgeDetector.METHOD_INK)
        task = _InkEvidenceTask(
            detector=detector,
            fallback_image=visible,
            fallback_rgb_image=visible_rgb,
            fallback_cache_identity=("visible",),
            fallback_cache_extent=(10.0, 20.0, 90.0, 84.0),
            fallback_output_size=(80, 64),
            evidence_image=expanded,
            recovery_image=expanded_rgb,
            recovery_compatible=True,
            recovery_disabled_reason="",
            tile_origin=(31, 31),
            enable_evidence=False,
            evidence_disabled_reason="test fallback",
            build_cost_map=False,
            edge_weight=0.5,
            generation=7,
            cache_identity=("expanded request",),
            cache_extent=(0.0, 0.0, 112.0, 96.0),
            output_size=(112, 96),
            callback=lambda *_args: None,
        )

        # QGIS 3.22 images can ship NumPy 1.19, before
        # sliding_window_view existed. The stable v1 fallback must still be
        # produced so disabling v2 never leaves the cache empty.
        with mock.patch.object(
            np.lib.stride_tricks,
            "sliding_window_view",
            None,
            create=True,
        ):
            self.assertTrue(task.run(), task.error)
        np.testing.assert_array_equal(
            task.fallback_edges,
            detector.detect_edges(visible),
        )
        self.assertEqual(task.fallback_edges.shape, visible.shape)
        self.assertIsNone(task.evidence)

        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        tool._ink_evidence_task = task
        tool._ink_evidence_generation = 7
        tool._pending_cache_identity = task.cache_identity
        tool._disposed = False
        tool._current_ink_cache_identity = lambda: task.cache_identity
        tool._publish_edge_cache = mock.Mock()
        tool._emit_recovery_state = mock.Mock()
        tool.smart_recovery_requested = False
        tool.smart_recovery_enabled = False

        tool._on_ink_evidence_finished(
            task,
            True,
            task.fallback_edges,
            None,
            None,
            task.evidence_error,
            None,
        )

        published = tool._publish_edge_cache.call_args.kwargs
        np.testing.assert_array_equal(published["rgb_image"], visible_rgb)
        np.testing.assert_array_equal(published["edges"], task.fallback_edges)
        self.assertIsNone(published["evidence"])
        self.assertEqual(
            published["read_extent"],
            QgsRectangle(10.0, 20.0, 90.0, 84.0),
        )
        self.assertEqual(published["output_size"], (80, 64))
        self.assertEqual(published["tile_origin"], (0, 0))
        self.assertEqual(published["cache_identity"], ("visible",))

    def test_recovery_engine_can_be_attached_after_background_preparation(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        tool._disposed = False
        tool.smart_recovery_requested = True
        tool.freehand = False
        tool.edge_weight = 0.5
        from ai_vectorizer.core.edge_detector import EdgeDetector

        tool.edge_method = EdgeDetector.METHOD_INK
        tool._recovery_task = None
        tool._recovery_request = None
        tool._recovery_generation = 3
        tool._recovery_encoding = object()
        tool._recovery_encoding_cache_generation = 2
        tool.recovery_state_callback = mock.Mock()
        engine = SimpleNamespace(is_ready=True)

        self.assertTrue(tool.set_recovery_engine(engine))
        self.assertIs(tool.recovery_engine, engine)
        self.assertTrue(tool.smart_recovery_enabled)
        self.assertIsNone(tool._recovery_encoding)
        self.assertEqual(tool._current_recovery_state, "Ink")

    def test_recovery_prepare_task_verifies_and_initializes_off_ui_thread(self):
        from ai_vectorizer.ui.main_dialog import _RecoveryPrepareTask
        from ai_vectorizer.core import efficientsam_recovery

        status = SimpleNamespace(ready=True)

        class Engine:
            @staticmethod
            def inspect(cache_root):
                self.assertEqual(cache_root, "/verified/cache")
                return status

            def __init__(instance, cache_root):
                self.assertEqual(cache_root, "/verified/cache")
                instance.is_ready = True

        callback = mock.Mock()
        with mock.patch.object(
            efficientsam_recovery,
            "EfficientSAMRecoveryEngine",
            Engine,
        ):
            task = _RecoveryPrepareTask(
                "/verified/cache",
                9,
                True,
                callback,
            )
            self.assertTrue(task.run(), task.error)
            task.finished(True)

        self.assertIs(task.status, status)
        self.assertTrue(task.engine.is_ready)
        callback.assert_called_once_with(task, True, status, task.engine, None)

    def test_recovery_prepare_completion_is_generation_guarded_and_injects_ready_engine(self):
        from ai_vectorizer.ui.main_dialog import AIVectorizerDock

        stale = SimpleNamespace(generation=4)
        current = SimpleNamespace(generation=5)
        active_tool = SimpleNamespace(
            smart_recovery_requested=True,
            set_recovery_engine=mock.Mock(),
        )
        dock = SimpleNamespace(
            recovery_prepare_task=current,
            _recovery_prepare_generation=5,
            _shutting_down=False,
            _recovery_model_status=None,
            _recovery_prepare_error="",
            recovery_engine=None,
            recovery_install_task=None,
            recovery_install_btn=mock.Mock(),
            active_tool=active_tool,
            _refresh_recovery_availability=mock.Mock(),
        )
        engine = SimpleNamespace(is_ready=True)

        AIVectorizerDock._on_recovery_prepare_finished(
            dock,
            stale,
            True,
            SimpleNamespace(ready=True),
            engine,
            None,
        )
        self.assertIs(dock.recovery_prepare_task, current)
        active_tool.set_recovery_engine.assert_not_called()

        status = SimpleNamespace(ready=True)
        AIVectorizerDock._on_recovery_prepare_finished(
            dock,
            current,
            True,
            status,
            engine,
            None,
        )
        self.assertIsNone(dock.recovery_prepare_task)
        self.assertIs(dock._recovery_model_status, status)
        self.assertIs(dock.recovery_engine, engine)
        active_tool.set_recovery_engine.assert_called_once_with(engine)
        dock._refresh_recovery_availability.assert_called_once_with()

    def test_recovery_install_task_reports_cooperative_download_cancel_as_cancel(self):
        from ai_vectorizer.core import model_store
        from ai_vectorizer.ui.main_dialog import _RecoveryInstallTask

        callback = mock.Mock()
        task = _RecoveryInstallTask("/unused/cache", callback)
        with mock.patch.object(
            model_store,
            "fetch_bundle",
            side_effect=model_store.ModelDownloadCancelled("cancelled"),
        ):
            self.assertFalse(task.run())
        self.assertIsNone(task.error)
        self.assertIsNone(task.bundle)

    def test_recovery_install_task_uses_explicit_corrupt_object_repair(self):
        from ai_vectorizer.core import model_store
        from ai_vectorizer.ui.main_dialog import _RecoveryInstallTask

        callback = mock.Mock()
        repaired = object()
        task = _RecoveryInstallTask(
            "/unused/cache",
            callback,
            repair_corrupt=True,
        )
        with mock.patch.object(
            model_store,
            "repair_bundle",
            return_value=repaired,
        ) as repair, mock.patch.object(model_store, "fetch_bundle") as fetch:
            self.assertTrue(task.run(), task.error)

        repair.assert_called_once()
        fetch.assert_not_called()
        self.assertIs(task.bundle, repaired)
        self.assertIsNone(task.error)

    def test_recovery_install_store_return_is_the_cancel_commit_point(self):
        from ai_vectorizer.core import model_store
        from ai_vectorizer.ui.main_dialog import _RecoveryInstallTask

        task = _RecoveryInstallTask("/unused/cache", mock.Mock())
        with mock.patch.object(
            task,
            "isCanceled",
            side_effect=[False, True],
        ) as cancelled, mock.patch.object(
            model_store,
            "fetch_bundle",
            return_value=object(),
        ):
            self.assertTrue(task.run(), task.error)

        cancelled.assert_called_once_with()

    def test_recovery_install_scheduler_failure_restores_retry_state(self):
        from ai_vectorizer.ui import main_dialog

        status = SimpleNamespace(
            artifacts=(SimpleNamespace(state="corrupt"),),
        )
        button = mock.Mock()
        dock = SimpleNamespace(
            recovery_install_task=None,
            _recovery_model_status=status,
            _recovery_prepare_error="",
            recovery_install_btn=button,
            _cancel_recovery_prepare=mock.Mock(),
            _release_recovery_engine=mock.Mock(),
            _sam_models_dir=mock.Mock(return_value="/unused/cache"),
            _on_recovery_install_finished=mock.Mock(),
            _set_recovery_state=mock.Mock(),
            _refresh_recovery_availability=mock.Mock(),
            _tr=lambda _ko, en: en,
        )
        scheduled_task = object()
        manager = SimpleNamespace(
            addTask=mock.Mock(side_effect=RuntimeError("scheduler offline")),
        )
        application = SimpleNamespace(taskManager=lambda: manager)

        with mock.patch.object(
            main_dialog,
            "_RecoveryInstallTask",
            return_value=scheduled_task,
        ), mock.patch.object(main_dialog, "QgsApplication", application):
            main_dialog.AIVectorizerDock.install_recovery_model(dock)

        self.assertIsNone(dock.recovery_install_task)
        self.assertIs(dock._recovery_model_status, status)
        button.setEnabled.assert_has_calls([mock.call(False), mock.call(True)])
        dock._refresh_recovery_availability.assert_called_once_with()

    def test_failed_recovery_repair_reinspects_before_retry(self):
        from ai_vectorizer.ui.main_dialog import AIVectorizerDock

        task = SimpleNamespace(repair_corrupt=True)
        previous_status = SimpleNamespace(ready=False)
        dock = SimpleNamespace(
            recovery_install_task=task,
            _shutting_down=False,
            recovery_install_btn=mock.Mock(),
            _recovery_model_status=previous_status,
            _recovery_prepare_error="previous",
            _set_recovery_state=mock.Mock(),
            _refresh_recovery_availability=mock.Mock(),
            _tr=lambda _ko, en: en,
        )

        AIVectorizerDock._on_recovery_install_finished(
            dock,
            task,
            False,
            None,
            RuntimeError("offline"),
        )

        self.assertIsNone(dock.recovery_install_task)
        self.assertIsNone(dock._recovery_model_status)
        self.assertEqual(dock._recovery_prepare_error, "")
        dock.recovery_install_btn.setVisible.assert_called_once_with(False)
        dock._refresh_recovery_availability.assert_called_once_with()

    def test_retry_does_not_treat_an_enhanced_route_as_a_new_ink_champion(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        tool.smart_recovery_requested = True
        tool.smart_recovery_enabled = True
        tool._current_recovery_state = "Enhanced"
        tool.recovery_state_callback = mock.Mock()

        self.assertFalse(tool.retry_current_segment())
        self.assertEqual(tool._current_recovery_state, "Enhanced")
        self.assertIn(
            "already enhanced",
            tool.recovery_state_callback.call_args.args[1],
        )

    def test_retry_during_recovery_preserves_pending_preview_identity(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        identity = (4, ((1.0, 1.0), (8.0, 8.0)), (8.0, 8.0))
        task = object()
        request = {"request_generation": 7, "preview_identity": identity}
        tool.smart_recovery_requested = True
        tool.smart_recovery_enabled = True
        tool._current_recovery_state = "Recovering"
        tool._recovery_task = task
        tool._recovery_request = request
        tool._recovery_generation = 7
        tool._recovery_preview_identity = identity
        tool._pending_livewire_accept_point = QgsPointXY(8, 8)
        tool._pending_livewire_recovery_identity = identity
        tool._schedule_smart_recovery = mock.Mock()
        tool._emit_recovery_state = mock.Mock()

        self.assertFalse(tool.retry_current_segment())

        self.assertIs(tool._recovery_task, task)
        self.assertIs(tool._recovery_request, request)
        self.assertEqual(tool._recovery_generation, 7)
        self.assertEqual(tool._recovery_preview_identity, identity)
        self.assertEqual(tool._pending_livewire_recovery_identity, identity)
        tool._schedule_smart_recovery.assert_not_called()
        tool._emit_recovery_state.assert_called_once_with(
            "Recovering",
            "Recovery is already evaluating this Ink segment.",
        )

    def test_retry_while_cancelled_task_drains_does_not_stick_on_recovering(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        identity = (4, ((1.0, 1.0), (8.0, 8.0)), (8.0, 8.0))
        task = SimpleNamespace(
            request_generation=7,
            cache_generation=4,
            preview_identity=identity,
            encoding=object(),
        )
        tool.smart_recovery_requested = True
        tool.smart_recovery_enabled = True
        tool._current_recovery_state = "Ink"
        tool.recovery_state_callback = mock.Mock()
        tool._recovery_task = task
        tool._recovery_request = None
        tool._recovery_generation = 8
        tool._recovery_preview_identity = None
        tool._cache_generation = 4
        tool._disposed = False
        tool.is_tracing = True

        self.assertFalse(tool.retry_current_segment())
        self.assertEqual(tool._current_recovery_state, "Recovering")

        tool._on_recovery_preview_finished(
            task,
            False,
            None,
            None,
            None,
            None,
        )

        self.assertIsNone(tool._recovery_task)
        self.assertEqual(tool._current_recovery_state, "Ink fallback")
        self.assertIn(
            "Ink was kept",
            tool.recovery_state_callback.call_args.args[1],
        )

    def test_raster_data_change_invalidates_and_reschedules_only_when_active(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        tool._is_active = True
        tool._needs_edge_cache = mock.Mock(return_value=True)
        tool._schedule_edge_cache_update = mock.Mock()
        tool._clear_edge_cache = mock.Mock()

        tool._on_raster_data_changed()
        tool._schedule_edge_cache_update.assert_called_once_with()
        tool._clear_edge_cache.assert_not_called()

        tool._is_active = False
        tool._on_raster_data_changed()
        tool._clear_edge_cache.assert_called_once_with()

    def test_stale_recovery_cannot_replace_a_new_confident_ink_preview(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        old_identity = (4, ((1.0, 1.0), (8.0, 8.0)), (8.0, 8.0))
        task = SimpleNamespace(
            request_generation=7,
            cache_generation=4,
            preview_identity=old_identity,
            encoding=object(),
        )
        tool._recovery_task = task
        tool._recovery_request = {"old": True}
        tool._recovery_generation = 7
        tool._recovery_preview_identity = old_identity
        tool._cache_generation = 4
        tool._disposed = False
        tool.is_tracing = True
        tool.smart_recovery_enabled = True
        tool._recovery_cache_compatible = True
        tool._recovery_cache_disabled_reason = ""
        tool._livewire_disabled = False
        tool._livewire_tree = SimpleNamespace(root=(1, 1))
        tool._recovery_pixel_path = lambda: ((1.0, 1.0), (5.0, 4.0))
        tool.map_to_pixel_float = lambda _point: (5.0, 4.0)
        tool._current_livewire_anchor_pixel = lambda: (1, 1)
        tool._build_recovery_request = mock.Mock(
            return_value={"trigger": False, "reason": "ink_confident"}
        )
        tool.preview_path = ["new confident Ink"]
        tool._emit_recovery_state = mock.Mock()

        self.assertFalse(tool._schedule_smart_recovery(QgsPointXY(5, 4)))
        self.assertEqual(tool._recovery_generation, 8)
        self.assertIsNone(tool._recovery_request)
        tool._emit_recovery_state.assert_called_once()
        self.assertEqual(tool._emit_recovery_state.call_args.args[0], "Ink")
        state_call_count = tool._emit_recovery_state.call_count

        tool._on_recovery_preview_finished(
            task,
            True,
            SimpleNamespace(reached_target=True, path=((2, 2), (8, 8))),
            ((1.0, 1.0), (8.0, 8.0)),
            SimpleNamespace(accepted=True, reason="improved"),
            None,
        )

        self.assertEqual(tool.preview_path, ["new confident Ink"])
        self.assertEqual(tool._emit_recovery_state.call_count, state_call_count)

    def test_current_recovery_error_keeps_ink_champion_and_clears_request(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        identity = (4, ((1.0, 1.0), (8.0, 8.0)), (8.0, 8.0))
        task = SimpleNamespace(
            request_generation=7,
            cache_generation=4,
            preview_identity=identity,
            encoding=object(),
        )
        tool._recovery_task = task
        tool._recovery_request = {
            "request_generation": 7,
            "cache_generation": 4,
        }
        tool._recovery_generation = 7
        tool._recovery_preview_identity = identity
        tool._cache_generation = 4
        tool._disposed = False
        tool.is_tracing = True
        tool.smart_recovery_enabled = True
        tool.preview_path = ["immutable Ink champion"]
        tool._emit_recovery_state = mock.Mock()

        error = RuntimeError("injected ONNX failure")
        tool._on_recovery_preview_finished(
            task,
            False,
            None,
            None,
            None,
            error,
        )

        self.assertEqual(tool.preview_path, ["immutable Ink champion"])
        self.assertIsNone(tool._recovery_request)
        tool._emit_recovery_state.assert_called_once()
        self.assertEqual(tool._emit_recovery_state.call_args.args[0], "Ink fallback")
        self.assertIn("Ink was kept", tool._emit_recovery_state.call_args.args[1])
        self.assertIn("injected ONNX failure", tool._emit_recovery_state.call_args.args[1])

    def test_enhanced_preview_acceptance_matches_only_its_visible_target(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        tool._current_recovery_state = "Enhanced"
        tool.preview_path = [QgsPointXY(2, 2), QgsPointXY(10, 10)]
        tool._livewire_request_point = QgsPointXY(10, 10)
        tool.canvas = SimpleNamespace(mapUnitsPerPixel=lambda: 1.0)

        self.assertTrue(tool._visible_enhanced_preview_matches(QgsPointXY(15, 10)))
        self.assertFalse(tool._visible_enhanced_preview_matches(QgsPointXY(30, 10)))
        tool._current_recovery_state = "Ink"
        self.assertFalse(tool._visible_enhanced_preview_matches(QgsPointXY(10, 10)))

    def test_pending_livewire_fallback_drains_accept_and_close_actions(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        tool.is_tracing = True
        tool.auto_path = False
        tool.start_point = QgsPointXY(1, 2)
        tool._render_preview = mock.Mock()
        tool._push_message = mock.Mock()
        tool._commit_visible_livewire_segment = mock.Mock()

        tool._pending_livewire_accept_point = QgsPointXY(8, 9)
        tool._pending_livewire_auto_accept = True
        self.assertTrue(
            tool._resolve_pending_livewire_fallback(
                nearby=False,
                detail="exact cursor",
            )
        )
        self.assertIsNone(tool._pending_livewire_accept_point)
        self.assertFalse(tool._pending_livewire_auto_accept)
        tool._commit_visible_livewire_segment.assert_called_once()

        tool._commit_visible_livewire_segment.reset_mock()
        tool._pending_livewire_accept_point = QgsPointXY(3, 4)
        tool._pending_livewire_auto_accept = False
        self.assertTrue(
            tool._resolve_pending_livewire_fallback(
                nearby=False,
                detail="close preview",
            )
        )
        self.assertEqual(tool.preview_path, [tool.start_point])
        self.assertEqual(tool._livewire_request_point, tool.start_point)
        tool._commit_visible_livewire_segment.assert_not_called()

    def test_ready_livewire_tree_displays_deferred_close_without_committing(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        task = SimpleNamespace(generation=5, anchor_pixel=(10, 10))
        tree = SimpleNamespace(root=(10, 10))
        tool._livewire_task = task
        tool._livewire_generation = 5
        tool._livewire_tree = None
        tool._livewire_failed_anchor = None
        tool._pending_livewire_accept_point = QgsPointXY(2, 3)
        tool._pending_livewire_auto_accept = False
        tool.start_point = QgsPointXY(1, 1)
        tool.auto_path = False
        tool.use_sam = False
        tool.is_tracing = True
        tool._current_livewire_anchor_pixel = lambda: (10, 10)
        tool._recovery_preview_identity = None
        tool._present_livewire_cursor_preview = mock.Mock(return_value=False)
        tool._commit_visible_livewire_segment = mock.Mock()

        tool._on_livewire_tree_finished(task, True, tree, None)

        self.assertIs(tool._livewire_tree, tree)
        self.assertIsNone(tool._pending_livewire_accept_point)
        tool._present_livewire_cursor_preview.assert_called_once_with(
            tool.start_point,
            global_mode=False,
            request_tree=False,
            schedule_recovery=True,
        )
        tool._commit_visible_livewire_segment.assert_not_called()

    def test_deferred_click_waits_when_smart_recovery_actually_starts(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        identity = (3, ((1.0, 1.0), (8.0, 8.0)), (8.0, 8.0))
        task = SimpleNamespace(generation=5, anchor_pixel=(10, 10))
        tree = SimpleNamespace(root=(10, 10))
        pending = QgsPointXY(8, 8)
        tool._livewire_task = task
        tool._livewire_generation = 5
        tool._livewire_tree = None
        tool._livewire_failed_anchor = None
        tool._pending_livewire_accept_point = pending
        tool._pending_livewire_auto_accept = True
        tool._pending_livewire_recovery_identity = None
        tool._recovery_preview_identity = identity
        tool.start_point = QgsPointXY(1, 1)
        tool.auto_path = False
        tool.use_sam = False
        tool.is_tracing = True
        tool._current_livewire_anchor_pixel = lambda: (10, 10)
        tool._present_livewire_cursor_preview = mock.Mock(return_value=True)
        tool._commit_visible_livewire_segment = mock.Mock()

        tool._on_livewire_tree_finished(task, True, tree, None)

        self.assertEqual(tool._pending_livewire_accept_point, pending)
        self.assertTrue(tool._pending_livewire_auto_accept)
        self.assertEqual(tool._pending_livewire_recovery_identity, identity)
        tool._commit_visible_livewire_segment.assert_not_called()

    def test_deferred_click_commits_immediately_when_recovery_does_not_start(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        task = SimpleNamespace(generation=5, anchor_pixel=(10, 10))
        tree = SimpleNamespace(root=(10, 10))
        pending = QgsPointXY(8, 8)
        tool._livewire_task = task
        tool._livewire_generation = 5
        tool._livewire_tree = None
        tool._livewire_failed_anchor = None
        tool._pending_livewire_accept_point = pending
        tool._pending_livewire_auto_accept = True
        tool._pending_livewire_recovery_identity = None
        tool._recovery_preview_identity = ("confident",)
        tool.start_point = QgsPointXY(1, 1)
        tool.auto_path = False
        tool.use_sam = False
        tool.is_tracing = True
        tool._current_livewire_anchor_pixel = lambda: (10, 10)
        tool._present_livewire_cursor_preview = mock.Mock(return_value=False)
        tool._commit_visible_livewire_segment = mock.Mock()

        tool._on_livewire_tree_finished(task, True, tree, None)

        self.assertIsNone(tool._pending_livewire_accept_point)
        self.assertIsNone(tool._pending_livewire_recovery_identity)
        tool._commit_visible_livewire_segment.assert_called_once_with(pending)

    def test_pending_recovery_error_commits_the_unchanged_ink_champion(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        identity = (4, ((1.0, 1.0), (8.0, 8.0)), (8.0, 8.0))
        pending = QgsPointXY(8, 8)
        task = SimpleNamespace(
            request_generation=7,
            cache_generation=4,
            preview_identity=identity,
            encoding=object(),
        )
        tool._recovery_task = task
        tool._recovery_request = {
            "request_generation": 7,
            "cache_generation": 4,
        }
        tool._recovery_generation = 7
        tool._recovery_preview_identity = identity
        tool._cache_generation = 4
        tool._disposed = False
        tool.is_tracing = True
        tool.smart_recovery_enabled = True
        tool.preview_path = ["immutable Ink champion"]
        tool._pending_livewire_accept_point = pending
        tool._pending_livewire_auto_accept = True
        tool._pending_livewire_recovery_identity = identity
        tool._emit_recovery_state = mock.Mock()
        committed = []
        tool._commit_visible_livewire_segment = mock.Mock(
            side_effect=lambda point: committed.append((point, list(tool.preview_path)))
        )

        tool._on_recovery_preview_finished(
            task,
            False,
            None,
            None,
            None,
            RuntimeError("injected ONNX failure"),
        )

        self.assertEqual(committed, [(pending, ["immutable Ink champion"])])
        self.assertIsNone(tool._pending_livewire_accept_point)
        self.assertIsNone(tool._pending_livewire_recovery_identity)

    def test_pending_recovery_accepts_challenger_before_automatic_commit(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        identity = (6, ((1.0, 1.0), (8.0, 8.0)), (8.0, 8.0))
        pending = QgsPointXY(8, 8)
        task = SimpleNamespace(
            request_generation=9,
            cache_generation=6,
            preview_identity=identity,
            encoding=object(),
        )
        tool._recovery_task = task
        tool._recovery_request = {
            "request_generation": 9,
            "cache_generation": 6,
            "target_map": pending,
        }
        tool._recovery_generation = 9
        tool._recovery_preview_identity = identity
        tool._cache_generation = 6
        tool._disposed = False
        tool.is_tracing = True
        tool.smart_recovery_enabled = True
        tool.auto_path = False
        tool.preview_path = [QgsPointXY(2, 2), pending]
        tool._pending_livewire_accept_point = pending
        tool._pending_livewire_auto_accept = True
        tool._pending_livewire_recovery_identity = identity
        tool.pixel_to_map = lambda x, y: QgsPointXY(x, y)
        tool._render_preview = mock.Mock()
        tool._emit_recovery_state = mock.Mock()
        committed = []
        tool._commit_visible_livewire_segment = mock.Mock(
            side_effect=lambda point: committed.append((point, list(tool.preview_path)))
        )

        tool._on_recovery_preview_finished(
            task,
            True,
            SimpleNamespace(reached_target=True),
            ((1.0, 1.0), (4.0, 5.0), (8.0, 8.0)),
            SimpleNamespace(accepted=True),
            None,
        )

        self.assertEqual(
            committed,
            [(pending, [QgsPointXY(4, 5), QgsPointXY(8, 8)])],
        )
        self.assertIsNone(tool._pending_livewire_accept_point)

    def test_pending_recovery_updates_close_preview_without_auto_saving(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        identity = (6, ((8.0, 8.0), (1.0, 1.0)), (1.0, 1.0))
        start = QgsPointXY(1, 1)
        task = SimpleNamespace(
            request_generation=9,
            cache_generation=6,
            preview_identity=identity,
            encoding=object(),
        )
        tool._recovery_task = task
        tool._recovery_request = {
            "request_generation": 9,
            "cache_generation": 6,
            "target_map": start,
        }
        tool._recovery_generation = 9
        tool._recovery_preview_identity = identity
        tool._cache_generation = 6
        tool._disposed = False
        tool.is_tracing = True
        tool.smart_recovery_enabled = True
        tool.auto_path = False
        tool.preview_path = [start]
        tool._pending_livewire_accept_point = start
        tool._pending_livewire_auto_accept = False
        tool._pending_livewire_recovery_identity = identity
        tool.pixel_to_map = lambda x, y: QgsPointXY(x, y)
        tool._render_preview = mock.Mock()
        tool._emit_recovery_state = mock.Mock()
        tool._commit_visible_livewire_segment = mock.Mock()

        tool._on_recovery_preview_finished(
            task,
            True,
            SimpleNamespace(reached_target=True),
            ((8.0, 8.0), (4.0, 5.0), (1.0, 1.0)),
            SimpleNamespace(accepted=True),
            None,
        )

        self.assertEqual(tool.preview_path, [QgsPointXY(4, 5), start])
        self.assertIsNone(tool._pending_livewire_accept_point)
        tool._commit_visible_livewire_segment.assert_not_called()

    def test_stale_pending_recovery_fails_closed_to_ink(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        identity = (4, ((1.0, 1.0), (8.0, 8.0)), (8.0, 8.0))
        pending = QgsPointXY(8, 8)
        task = SimpleNamespace(
            request_generation=7,
            cache_generation=4,
            preview_identity=identity,
            encoding=object(),
        )
        tool._recovery_task = task
        tool._recovery_request = None
        tool._recovery_generation = 8
        tool._recovery_preview_identity = ("newer",)
        tool._cache_generation = 4
        tool._disposed = False
        tool.is_tracing = True
        tool.smart_recovery_enabled = True
        tool.preview_path = ["immutable Ink champion"]
        tool._pending_livewire_accept_point = pending
        tool._pending_livewire_auto_accept = True
        tool._pending_livewire_recovery_identity = identity
        tool._emit_recovery_state = mock.Mock()
        tool._commit_visible_livewire_segment = mock.Mock()

        tool._on_recovery_preview_finished(
            task,
            False,
            None,
            None,
            None,
            None,
        )

        tool._commit_visible_livewire_segment.assert_called_once_with(pending)
        self.assertIsNone(tool._pending_livewire_accept_point)
        self.assertIn(
            "stale",
            tool._emit_recovery_state.call_args.args[1].lower(),
        )

    def test_ctrl_z_cancels_pending_recovery_before_clearing_preview(self):
        from ai_vectorizer.tools.smart_trace_tool import _qt_value

        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        tool.is_tracing = True
        tool._pending_livewire_accept_point = QgsPointXY(8, 8)
        tool._pending_livewire_auto_accept = True
        tool._pending_livewire_recovery_identity = ("pending",)
        tool._livewire_request_point = QgsPointXY(8, 8)
        tool._invalidate_recovery = mock.Mock()
        tool._clear_preview = mock.Mock()
        event = SimpleNamespace(
            key=lambda: _qt_value("Key_Z", "Key"),
            modifiers=lambda: _qt_value("ControlModifier", "KeyboardModifier"),
            accept=mock.Mock(),
        )

        tool.keyPressEvent(event)

        tool._invalidate_recovery.assert_called_once()
        self.assertIsNone(tool._pending_livewire_accept_point)
        self.assertIsNone(tool._pending_livewire_recovery_identity)
        self.assertIsNone(tool._livewire_request_point)
        tool._clear_preview.assert_called_once_with(stop_timer=False)
        event.accept.assert_called_once_with()

    def test_ink_task_failure_resolves_a_pending_click_as_exact_cursor(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        task = SimpleNamespace(generation=3, cache_identity=("request",))
        tool._ink_evidence_task = task
        tool._ink_evidence_generation = 3
        tool._pending_cache_identity = ("request",)
        tool._disposed = False
        tool._pending_livewire_accept_point = QgsPointXY(4, 5)
        tool._current_ink_cache_identity = lambda: ("request",)
        tool._resolve_pending_livewire_fallback = mock.Mock(return_value=True)
        tool._emit_recovery_state = mock.Mock()

        error = RuntimeError("injected evidence failure")
        tool._on_ink_evidence_finished(
            task,
            False,
            None,
            None,
            None,
            None,
            error,
        )

        tool._resolve_pending_livewire_fallback.assert_called_once()
        self.assertFalse(
            tool._resolve_pending_livewire_fallback.call_args.kwargs["nearby"]
        )
        self.assertIsNone(tool._ink_evidence_task)
        tool._emit_recovery_state.assert_called_once()

    def test_checkpoint_undo_discards_stale_recovery_and_livewire_state(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        start = QgsPointXY(1, 1)
        middle = QgsPointXY(2, 2)
        anchor = QgsPointXY(3, 3)
        tool.path_points = [start, middle, anchor]
        tool.checkpoints = [0, 2]
        tool._livewire_generation = 9
        tool._livewire_tree = object()
        tool._livewire_anchor_pixel = (3, 3)
        tool._livewire_request_point = anchor
        tool._livewire_failed_anchor = (3, 3)
        tool._pending_livewire_accept_point = QgsPointXY(4, 4)
        tool._pending_livewire_auto_accept = True
        tool.last_map_point = anchor
        tool.last_input_point = anchor
        tool.last_hover_pos = anchor
        tool.last_sample_pos = object()
        tool.last_preview_pos = object()
        tool._invalidate_recovery = mock.Mock()
        tool._clear_preview = mock.Mock()
        tool._cancel_livewire_task = mock.Mock()
        tool._request_livewire_tree = mock.Mock(return_value=True)
        tool.redraw_confirmed_path = mock.Mock()
        tool.checkpoint_markers = SimpleNamespace(
            reset=mock.Mock(),
            addPoint=mock.Mock(),
        )

        tool.undo_to_checkpoint()

        self.assertEqual(tool.path_points, [start])
        self.assertEqual(tool.checkpoints, [0])
        tool._invalidate_recovery.assert_called_once()
        tool._clear_preview.assert_called_once_with()
        tool._cancel_livewire_task.assert_called_once_with()
        self.assertEqual(tool._livewire_generation, 10)
        self.assertIsNone(tool._livewire_tree)
        self.assertIsNone(tool._livewire_anchor_pixel)
        self.assertIsNone(tool._livewire_request_point)
        self.assertIsNone(tool._livewire_failed_anchor)
        self.assertIsNone(tool._pending_livewire_accept_point)
        self.assertFalse(tool._pending_livewire_auto_accept)
        self.assertEqual(tool.last_map_point, start)
        self.assertEqual(tool.last_input_point, start)
        self.assertIsNone(tool.last_hover_pos)
        self.assertIsNone(tool.last_sample_pos)
        self.assertIsNone(tool.last_preview_pos)
        tool.redraw_confirmed_path.assert_called_once_with()
        tool._request_livewire_tree.assert_called_once_with(force=True)

    def test_closed_candidate_save_is_transactional_on_failure_and_exception(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        start = QgsPointXY(0, 0)
        anchor = QgsPointXY(5, 0)
        bend = QgsPointXY(5, 5)
        confirmed = [start, anchor]
        tool.path_points = confirmed
        observed = []

        def fail_save(*, closed, elevation):
            observed.append((list(tool.path_points), closed, elevation))
            return False

        tool.save_to_layer = fail_save
        self.assertFalse(
            tool._save_closed_path_candidate([bend, start], 12.5)
        )
        self.assertIs(tool.path_points, confirmed)
        self.assertEqual(observed[0][0], [start, anchor, bend])
        self.assertTrue(observed[0][1])
        self.assertEqual(observed[0][2], 12.5)

        def raise_save(**_kwargs):
            self.assertIsNot(tool.path_points, confirmed)
            raise RuntimeError("injected layer failure")

        tool.save_to_layer = raise_save
        with self.assertRaisesRegex(RuntimeError, "injected layer failure"):
            tool._save_closed_path_candidate([bend, start], 20.0)
        self.assertIs(tool.path_points, confirmed)

    def test_cancelled_enhanced_close_never_mutates_confirmed_path(self):
        from ai_vectorizer.tools.smart_trace_tool import _qt_value

        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        start = QgsPointXY(0, 0)
        anchor = QgsPointXY(5, 0)
        enhanced = QgsPointXY(3, 4)
        confirmed = [start, anchor]
        tool.path_points = confirmed
        tool.is_tracing = True
        tool.start_point = start
        tool._pending_livewire_accept_point = None
        tool.auto_path = False
        tool.use_sam = False
        tool.preview_path = [enhanced, start]
        tool.toMapCoordinates = mock.Mock(return_value=start)
        tool.is_near_start = mock.Mock(return_value=True)
        tool._defer_click_until_livewire_ready = mock.Mock(return_value=False)
        tool._visible_enhanced_preview_matches = mock.Mock(return_value=True)
        tool._invalidate_recovery = mock.Mock()
        tool._livewire_tree_is_ready = mock.Mock(return_value=True)
        tool.ask_elevation = mock.Mock(return_value=None)
        tool._save_closed_path_candidate = mock.Mock()
        event = SimpleNamespace(
            button=lambda: _qt_value("LeftButton", "MouseButton"),
            pos=lambda: object(),
        )

        tool.canvasPressEvent(event)

        self.assertIs(tool.path_points, confirmed)
        self.assertEqual(tool.preview_path, [enhanced, start])
        tool._save_closed_path_candidate.assert_not_called()
        tool._invalidate_recovery.assert_not_called()

    def test_plain_close_invalidates_old_recovery_before_ink_preview(self):
        from ai_vectorizer.tools.smart_trace_tool import _qt_value

        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        start = QgsPointXY(0, 0)
        anchor = QgsPointXY(5, 0)
        confirmed = [start, anchor]
        tool.path_points = confirmed
        tool.is_tracing = True
        tool.start_point = start
        tool._pending_livewire_accept_point = None
        tool.auto_path = False
        tool.use_sam = False
        tool.preview_path = [QgsPointXY(4, 2), start]
        tool.toMapCoordinates = mock.Mock(return_value=start)
        tool.is_near_start = mock.Mock(return_value=True)
        tool._defer_click_until_livewire_ready = mock.Mock(return_value=False)
        tool._visible_enhanced_preview_matches = mock.Mock(return_value=False)
        calls = mock.Mock()
        invalidate = mock.Mock()
        present = mock.Mock()
        calls.attach_mock(invalidate, "invalidate")
        calls.attach_mock(present, "present")
        tool._invalidate_recovery = invalidate
        tool._present_livewire_cursor_preview = present
        tool._livewire_tree_is_ready = mock.Mock(return_value=True)
        tool.ask_elevation = mock.Mock(return_value=None)
        tool._save_closed_path_candidate = mock.Mock()
        event = SimpleNamespace(
            button=lambda: _qt_value("LeftButton", "MouseButton"),
            pos=lambda: object(),
        )

        tool.canvasPressEvent(event)

        self.assertIs(tool.path_points, confirmed)
        self.assertEqual(
            [call[0] for call in calls.mock_calls],
            ["invalidate", "present"],
        )
        invalidate.assert_called_once_with(
            "Segment accepted; Ink is the new champion."
        )
        present.assert_called_once_with(
            start,
            global_mode=False,
            request_tree=False,
            schedule_recovery=False,
        )
        tool._save_closed_path_candidate.assert_not_called()

    def test_recovery_prompts_do_not_relabel_prior_vertices_as_positive(self):
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        previous = QgsPointXY(4, 8)
        anchor = QgsPointXY(20, 20)
        target = QgsPointXY(60, 20)
        tool.path_points = [previous, anchor]
        tool.cache_transform = {"ready": True}
        tool.cached_rgb_image = __import__("numpy").zeros(
            (100, 100, 3),
            dtype="uint8",
        )
        coordinates = {
            id(previous): (4, 8),
            id(anchor): (20, 20),
            id(target): (60, 20),
        }
        tool.map_to_pixel = lambda point: coordinates[id(point)]

        points, labels = tool._build_recovery_prompts(target)

        self.assertEqual(labels[:2].tolist(), [1, 1])
        self.assertEqual(points[:2].tolist(), [[20.0, 20.0], [60.0, 20.0]])
        self.assertNotIn([4.0, 8.0], points.tolist())

    def test_recovery_task_crops_to_livewire_window_and_preserves_endpoints(self):
        import numpy as np

        from ai_vectorizer.core.line_evidence import LineEvidence
        from ai_vectorizer.core.trace_kernel import TraceConfig
        from ai_vectorizer.tools.smart_trace_tool import _RecoveryPreviewTask

        score = np.ones((12, 12), dtype=np.float32)
        evidence = LineEvidence(
            center_score=score,
            centerline=score > 0.5,
            tangent_x=np.ones_like(score),
            tangent_y=np.zeros_like(score),
            coherence=np.ones_like(score),
        )

        class Engine:
            @staticmethod
            def encode(_image):
                return "encoding"

            @staticmethod
            def predict(encoding, _points, _labels):
                if encoding != "encoding":
                    raise AssertionError("unexpected encoding")
                return SimpleNamespace(mask=np.ones((12, 12), dtype=np.float32))

        task = _RecoveryPreviewTask(
            engine=Engine(),
            image=np.zeros((12, 12, 3), dtype=np.uint8),
            encoding=None,
            evidence=evidence,
            champion_path=((4.0, 5.0), (5.0, 6.0), (7.0, 7.0)),
            start_pixel=(4.0, 5.0),
            target_pixel=(7.0, 7.0),
            prompt_points=np.array([[4.0, 5.0], [7.0, 7.0]], dtype=np.float32),
            prompt_labels=np.array([1, 1], dtype=np.int32),
            window_bounds=(3, 4, 9, 9),
            cache_generation=1,
            request_generation=1,
            preview_identity=("current",),
            smooth_window_size=5,
            trace_config=TraceConfig(
                max_width=6,
                max_height=5,
                max_cells=30,
                validate_all_costs=False,
                validate_accessed_costs=False,
            ),
            callback=lambda *_args: None,
        )

        self.assertTrue(task.run(), task.error)
        self.assertTrue(task.trace_result.reached_target)
        self.assertEqual(task.challenger_path[0], (4.0, 5.0))
        self.assertEqual(task.challenger_path[-1], (7.0, 7.0))
        self.assertTrue(
            all(
                3.0 <= x < 9.0 and 4.0 <= y < 9.0
                for x, y in task.challenger_path
            )
        )

    def test_resume_updates_geometry_and_elevation_together(self):
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "contours", "memory")
        feature = QgsFeature(layer.fields())
        feature.setGeometry(
            QgsGeometry.fromPolylineXY([QgsPointXY(0, 0), QgsPointXY(1, 0)])
        )
        ok, added = layer.dataProvider().addFeatures([feature])
        self.assertTrue(ok)

        tool = self._tool(layer)
        tool.path_points = [QgsPointXY(1, 0), QgsPointXY(2, 0)]
        tool.resume_feature_id = added[0].id()
        tool.resume_at_start = False

        self.assertTrue(tool.save_to_layer(closed=False, elevation=20.0))
        self.assertTrue(layer.isEditable())
        elevation_index = layer.fields().indexOf("elevation")
        self.assertGreaterEqual(elevation_index, 0)
        updated = layer.getFeature(added[0].id())
        self.assertEqual(updated[elevation_index], 20.0)
        self.assertEqual(updated.geometry().asPolyline()[-1], QgsPointXY(2, 0))

        # Field creation, geometry and elevation are one QGIS edit command.
        layer.undoStack().undo()
        self.assertEqual(layer.fields().indexOf("elevation"), -1)
        restored = layer.getFeature(added[0].id())
        self.assertEqual(restored.geometry().asPolyline()[-1], QgsPointXY(1, 0))

    def test_new_feature_is_added_as_one_undoable_edit_command(self):
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "contours", "memory")
        tool = self._tool(layer)
        geometry = QgsGeometry.fromPolylineXY(
            [QgsPointXY(0, 0), QgsPointXY(2, 0)]
        )

        self.assertTrue(tool.save_geometry(geometry, elevation=12.0))
        self.assertTrue(layer.isEditable())
        self.assertEqual(layer.featureCount(), 1)
        self.assertGreaterEqual(layer.fields().indexOf("elevation"), 0)

        layer.undoStack().undo()
        self.assertEqual(layer.featureCount(), 0)
        self.assertEqual(layer.fields().indexOf("elevation"), -1)

    def test_failed_feature_add_rolls_back_new_elevation_field(self):
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "contours", "memory")
        tool = self._tool(layer)

        class RejectingLayer:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def __getattr__(self, name):
                return getattr(self.wrapped, name)

            def addFeature(self, _feature):
                return False

        geometry = QgsGeometry.fromPolylineXY(
            [QgsPointXY(0, 0), QgsPointXY(2, 0)]
        )
        rejecting = RejectingLayer(layer)

        self.assertFalse(
            tool._add_geometry_feature(rejecting, geometry, elevation=12.0)
        )
        layer.updateFields()
        self.assertEqual(layer.fields().indexOf("elevation"), -1)
        self.assertEqual(layer.featureCount(), 0)

    def test_idle_ctrl_z_is_delegated_to_qgis_edit_stack(self):
        from ai_vectorizer.tools.smart_trace_tool import _qt_value

        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "contours", "memory")
        tool = self._tool(layer)
        geometry = QgsGeometry.fromPolylineXY(
            [QgsPointXY(0, 0), QgsPointXY(2, 0)]
        )
        self.assertTrue(tool.save_geometry(geometry, elevation=12.0))
        self.assertEqual(layer.featureCount(), 1)

        class Event:
            ignored = False
            accepted = False

            def key(self):
                return _qt_value("Key_Z", "Key")

            def modifiers(self):
                return _qt_value("ControlModifier", "KeyboardModifier")

            def ignore(self):
                self.ignored = True

            def accept(self):
                self.accepted = True

        event = Event()
        self.assertFalse(tool.is_tracing)
        tool.keyPressEvent(event)
        self.assertTrue(event.ignored)
        self.assertFalse(event.accepted)

        # QGIS receives the ignored shortcut and invokes this layer edit stack.
        layer.undoStack().undo()
        self.assertEqual(layer.featureCount(), 0)

    def test_file_layer_write_stays_uncommitted_until_qgis_save(self):
        from ai_vectorizer.ui.main_dialog import _write_vector_layer, _writer_no_error

        source = QgsVectorLayer("LineString?crs=EPSG:3857", "source", "memory")
        source.dataProvider().addAttributes(
            [QgsField("elevation", _field_type("Double"))]
        )
        source.updateFields()
        temporary_directory = tempfile.TemporaryDirectory()
        self.__class__._temporary_directories.append(temporary_directory)
        path = str(Path(temporary_directory.name) / "contours.shp")
        self.assertEqual(
            _write_vector_layer(source, path, self.crs)[0],
            _writer_no_error(),
        )
        layer = QgsVectorLayer(path, "contours", "ogr")
        self.assertTrue(layer.isValid())
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        self.assertTrue(tool._ensure_edit_session(layer))
        feature = QgsFeature(layer.fields())
        feature.setGeometry(
            QgsGeometry.fromPolylineXY([QgsPointXY(0, 0), QgsPointXY(3, 0)])
        )
        feature[0] = 20.0

        self.assertTrue(tool._add_feature(layer, feature))
        self.assertEqual(layer.featureCount(), 1)
        on_disk = QgsVectorLayer(path, "on-disk", "ogr")
        self.assertTrue(on_disk.isValid())
        self.assertEqual(on_disk.featureCount(), 0)

        layer.undoStack().undo()
        self.assertEqual(layer.featureCount(), 0)
        del on_disk
        del tool
        del layer
        gc.collect()
        QgsApplication.processEvents()

    def test_feature_defaults_and_zm_guard(self):
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "defaults", "memory")
        layer.dataProvider().addAttributes(
            [
                QgsField("elevation", _field_type("Double")),
                QgsField("note", _field_type("String")),
                QgsField("id", _field_type("Int")),
            ]
        )
        layer.updateFields()
        note_index = layer.fields().indexOf("note")
        id_index = layer.fields().indexOf("id")
        layer.setDefaultValueDefinition(note_index, QgsDefaultValue("'default-note'"))
        layer.setDefaultValueDefinition(id_index, QgsDefaultValue("1000"))
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        tool.language = "en"
        geometry = QgsGeometry.fromPolylineXY(
            [QgsPointXY(0, 0), QgsPointXY(1, 1)]
        )

        feature = tool._build_feature(layer, geometry, 7.0)
        self.assertEqual(feature[note_index], "default-note")
        self.assertEqual(feature[id_index], 1000)

        multi_layer = QgsVectorLayer(
            "MultiLineString?crs=EPSG:3857",
            "multi",
            "memory",
        )
        multi_feature = tool._build_feature(multi_layer, geometry)
        self.assertTrue(multi_feature.geometry().isMultipart())

        z_layer = QgsVectorLayer("LineStringZ?crs=EPSG:3857", "z", "memory")
        m_layer = QgsVectorLayer("LineStringM?crs=EPSG:3857", "m", "memory")
        self.assertEqual(tool.unsupported_output_reason(z_layer), "z_or_m")
        self.assertEqual(tool.unsupported_output_reason(m_layer), "z_or_m")

    def test_hard_field_constraints_reject_atomic_feature_add(self):
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "required", "memory")
        layer.dataProvider().addAttributes(
            [QgsField("required", _field_type("String"))]
        )
        layer.updateFields()
        required_index = layer.fields().indexOf("required")
        layer.setFieldConstraint(
            required_index,
            _field_constraint("ConstraintNotNull"),
            _constraint_strength("ConstraintStrengthHard"),
        )
        tool = self._tool(layer)
        geometry = QgsGeometry.fromPolylineXY(
            [QgsPointXY(0, 0), QgsPointXY(2, 0)]
        )

        # Adding the elevation schema and feature is one edit command. A hard
        # failure on another required field must roll both changes back.
        self.assertFalse(tool.save_geometry(geometry, elevation=12.0))
        layer.updateFields()
        self.assertEqual(layer.featureCount(), 0)
        self.assertEqual(layer.fields().indexOf("elevation"), -1)

    def test_spot_layer_does_not_adopt_same_named_user_layer(self):
        from ai_vectorizer.config import DEFAULT_SPOT_LAYER_NAME

        project = QgsProject.instance()
        user_layer = QgsVectorLayer(
            "Point?crs=EPSG:3857",
            DEFAULT_SPOT_LAYER_NAME,
            "memory",
        )
        contour_layer = QgsVectorLayer(
            "LineString?crs=EPSG:3857",
            "contours",
            "memory",
        )
        project.addMapLayer(user_layer)
        project.addMapLayer(contour_layer)

        class Settings:
            def destinationCrs(self):
                return contour_layer.crs()

        class Canvas:
            def mapSettings(self):
                return Settings()

        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        tool.spot_height_layer = None
        tool.vector_layer = contour_layer
        tool.canvas = Canvas()
        owned_layer = tool.get_or_create_spot_layer()

        self.assertNotEqual(owned_layer.id(), user_layer.id())
        self.assertTrue(
            owned_layer.customProperty(tool.SPOT_LAYER_OWNERSHIP_PROPERTY, False)
        )

    def test_removed_owned_spot_layer_is_recreated_without_dead_wrapper(self):
        project = QgsProject.instance()
        contour_layer = QgsVectorLayer(
            "LineString?crs=EPSG:3857",
            "contours",
            "memory",
        )
        project.addMapLayer(contour_layer)

        class Settings:
            def destinationCrs(self):
                return contour_layer.crs()

        class Canvas:
            def mapSettings(self):
                return Settings()

        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        tool.spot_height_layer = None
        tool.vector_layer = contour_layer
        tool.canvas = Canvas()
        first = tool.get_or_create_spot_layer()
        first_id = first.id()
        project.removeMapLayer(first_id)
        gc.collect()
        QgsApplication.processEvents()

        second = tool.get_or_create_spot_layer()
        self.assertTrue(second.isValid())
        self.assertNotEqual(second.id(), first_id)

    def test_non_editable_layer_starts_one_atomic_edit_command(self):
        class Layer:
            def __init__(self):
                self.editable = False
                self.commands = []
                self.attributes = {}

            def isEditable(self):
                return self.editable

            def startEditing(self):
                self.editable = True
                return True

            def beginEditCommand(self, _label):
                self.commands.append("begin")

            def changeGeometry(self, _feature_id, _geometry):
                return True

            def changeAttributeValue(self, _feature_id, index, value):
                self.attributes[index] = value
                return True

            def endEditCommand(self):
                self.commands.append("end")

            def destroyEditCommand(self):
                self.commands.append("destroy")

            def updateExtents(self):
                pass

            def fields(self):
                return []

            class _Feature:
                @staticmethod
                def isValid():
                    return True

            def getFeature(self, _feature_id):
                return self._Feature()

        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        layer = Layer()

        self.assertTrue(tool._update_feature(layer, 4, object(), {2: 55.0}))
        self.assertTrue(layer.editable)
        self.assertEqual(layer.commands, ["begin", "end"])
        self.assertEqual(layer.attributes, {2: 55.0})

    def test_canvas_crs_change_resets_trace_and_refreshes_transforms(self):
        raster_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        layer = QgsVectorLayer("LineString?crs=EPSG:4326", "contours", "memory")
        tool = self._tool(layer, raster_crs=raster_crs)
        tool.activate()
        tool.is_tracing = True
        tool.path_points = [QgsPointXY(0, 0), QgsPointXY(1, 1)]

        new_canvas_crs = QgsCoordinateReferenceSystem("EPSG:32652")
        self.canvas.setDestinationCrs(new_canvas_crs)
        self.assertFalse(tool.is_tracing)
        self.assertEqual(tool.path_points, [])

        source_point = QgsPointXY(500000, 4000000)
        actual = tool._map_point_to_raster(source_point)
        expected_transform = QgsCoordinateTransform(
            new_canvas_crs,
            raster_crs,
            QgsProject.instance(),
        )
        expected = expected_transform.transform(source_point)
        self.assertAlmostEqual(actual.x(), expected.x(), places=8)
        self.assertAlmostEqual(actual.y(), expected.y(), places=8)
        tool.deactivate()

    def test_active_tool_stops_when_output_source_is_replaced(self):
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "contours", "memory")
        tool = self._tool(layer)
        emissions = []
        tool.deactivated.connect(lambda: emissions.append(True))
        self.canvas.setMapTool(tool)
        self.assertIs(self.canvas.mapTool(), tool)

        layer.dataSourceChanged.emit()

        self.assertIsNot(self.canvas.mapTool(), tool)
        self.assertEqual(emissions, [True])

    def test_loaded_output_path_detection_decodes_uri_and_aliases(self):
        from ai_vectorizer.ui import main_dialog
        from ai_vectorizer.ui.main_dialog import (
            AIVectorizerDock,
            _layer_file_path,
            _project_layers_using_path,
            _write_vector_layer,
        )

        source = QgsVectorLayer("LineString?crs=EPSG:3857", "source", "memory")
        temporary_directory = tempfile.TemporaryDirectory()
        self.__class__._temporary_directories.append(temporary_directory)
        path = str(Path(temporary_directory.name) / "contours.shp")
        alias = str(Path(temporary_directory.name) / "contours-alias.shp")
        self.assertEqual(_write_vector_layer(source, path, self.crs)[0], 0)
        os.symlink(path, alias)

        loaded = QgsVectorLayer(f"{path}|layerid=0", "loaded contours", "ogr")
        unrelated = QgsVectorLayer(
            "LineString?crs=EPSG:3857",
            "unrelated",
            "memory",
        )
        self.assertTrue(loaded.isValid())
        QgsProject.instance().addMapLayer(loaded)
        QgsProject.instance().addMapLayer(unrelated)

        matches = _project_layers_using_path(alias)
        self.assertEqual([layer.id() for layer in matches], [loaded.id()])

        class ProviderUriLayer:
            @staticmethod
            def providerType():
                return "ogr"

            @staticmethod
            def source():
                return f"{QUrl.fromLocalFile(path).toString()}|layerid=0"

        self.assertEqual(_layer_file_path(ProviderUriLayer()), path)

        warnings = []

        class PathEdit:
            @staticmethod
            def text():
                return alias

        class DockHarness:
            shp_path = PathEdit()

            @staticmethod
            def _tr(_ko, en):
                return en

        class MessageBoxHarness:
            @staticmethod
            def warning(*args):
                warnings.append(args)

        with mock.patch.object(
            main_dialog,
            "QMessageBox",
            MessageBoxHarness,
        ), mock.patch.object(main_dialog, "_write_vector_layer") as writer:
            AIVectorizerDock.create_shp_layer(DockHarness())
        writer.assert_not_called()
        self.assertEqual(len(warnings), 1)
        self.assertIn("loaded contours", warnings[0][-1])

    def test_preview_temp_directory_tracks_project_layer_lifetime(self):
        from ai_vectorizer.ui.main_dialog import _TemporaryPreviewStore

        store = _TemporaryPreviewStore(QgsProject.instance())
        preview_directory = tempfile.TemporaryDirectory()
        preview_path = Path(preview_directory.name) / "preview.tif"
        preview_path.write_bytes(b"preview")
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "preview", "memory")
        QgsProject.instance().addMapLayer(layer)
        layer_id = layer.id()
        store.track(layer, preview_directory)

        QgsProject.instance().removeMapLayer(layer_id)
        QgsApplication.processEvents()
        self.assertFalse(preview_path.exists())
        self.assertEqual(store._directories, {})

        shutdown_directory = tempfile.TemporaryDirectory()
        shutdown_path = Path(shutdown_directory.name) / "preview.tif"
        shutdown_path.write_bytes(b"preview")
        shutdown_layer = QgsVectorLayer(
            "LineString?crs=EPSG:3857",
            "preview shutdown",
            "memory",
        )
        QgsProject.instance().addMapLayer(shutdown_layer)
        shutdown_id = shutdown_layer.id()
        store.track(shutdown_layer, shutdown_directory)
        store.shutdown()
        self.assertIsNone(QgsProject.instance().mapLayer(shutdown_id))
        self.assertFalse(shutdown_path.exists())
        self.assertFalse(store._connected)

        retry_directory = tempfile.TemporaryDirectory()
        self.__class__._temporary_directories.append(retry_directory)
        retry_path = Path(retry_directory.name) / "preview.tif"
        retry_path.write_bytes(b"preview")

        class InitiallyLockedDirectory:
            name = retry_directory.name

            @staticmethod
            def cleanup():
                raise OSError("provider handle is still open")

        _TemporaryPreviewStore._cleanup_directory(InitiallyLockedDirectory())
        event_loop = QEventLoop()
        QTimer.singleShot(0, event_loop.quit)
        execute = getattr(event_loop, "exec", None) or event_loop.exec_
        execute()
        self.assertFalse(retry_path.exists())

    def test_dem_pipeline_runs_all_processing_stages_and_releases_resources(self):
        import qgis

        # The macOS app bundle keeps GDAL executables beside its Python
        # launcher. A shell-driven offscreen run does not inherit the PATH
        # adjustment made by the normal QGIS application launcher.
        executable_directory = Path(sys.executable).resolve().parent
        if (executable_directory / "gdal_translate").is_file():
            path_entries = os.environ.get("PATH", "").split(os.pathsep)
            if str(executable_directory) not in path_entries:
                os.environ["PATH"] = os.pathsep.join(
                    (str(executable_directory), *path_entries)
                )

        processing_paths = (
            Path(qgis.__file__).resolve().parent.parent / "plugins",
            Path(QgsApplication.pkgDataPath()) / "python" / "plugins",
            Path(QgsApplication.prefixPath()) / "share" / "qgis" / "python" / "plugins",
            Path(sys.executable).resolve().parent.parent
            / "Resources"
            / "qgis"
            / "python"
            / "plugins",
        )
        for processing_path in processing_paths:
            if (processing_path / "processing").is_dir():
                path_text = str(processing_path)
                if path_text not in sys.path:
                    sys.path.insert(0, path_text)
                break
        from processing.core.Processing import Processing
        from ai_vectorizer.core.dem_pipeline import (
            DemPipelineRunner,
            build_dem_request,
        )

        Processing.initialize()
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "terrain", "memory")
        layer.dataProvider().addAttributes(
            [QgsField("elevation", _field_type("Double"))]
        )
        layer.updateFields()
        features = []
        for elevation, y_coordinate in ((100.0, 0.0), (110.0, 50.0), (120.0, 100.0)):
            feature = QgsFeature(layer.fields())
            feature[0] = elevation
            feature.setGeometry(
                QgsGeometry.fromPolylineXY(
                    [
                        QgsPointXY(0.0, y_coordinate),
                        QgsPointXY(100.0, y_coordinate),
                    ]
                )
            )
            features.append(feature)
        self.assertTrue(layer.dataProvider().addFeatures(features)[0])
        QgsProject.instance().addMapLayer(layer)

        temporary_directory = tempfile.TemporaryDirectory()
        self.__class__._temporary_directories.append(temporary_directory)
        dem_path = str(Path(temporary_directory.name) / "terrain-dem.tif")
        hillshade_path = str(Path(temporary_directory.name) / "terrain-hillshade.tif")
        request = build_dem_request(
            contour_layer=layer,
            contour_field="elevation",
            pixel_size=10.0,
            dem_path=dem_path,
            hillshade_path=hillshade_path,
        )
        runner = DemPipelineRunner()
        event_loop = QEventLoop()
        terminal = []

        def finish(kind, *details):
            terminal.append((kind, details))
            event_loop.quit()

        runner.succeeded.connect(lambda *paths: finish("succeeded", *paths))
        runner.failed.connect(lambda message: finish("failed", message))
        runner.canceled.connect(lambda: finish("canceled"))
        timeout = QTimer()
        timeout.setSingleShot(True)
        timeout.timeout.connect(lambda: finish("timeout"))
        timeout.start(30_000)

        runner.start(request)
        execute = getattr(event_loop, "exec", None) or event_loop.exec_
        execute()
        timeout.stop()

        self.assertEqual(terminal[0][0], "succeeded", terminal)
        self.assertTrue(Path(dem_path).is_file())
        self.assertTrue(Path(hillshade_path).is_file())
        self.assertFalse(runner.is_running)
        self.assertEqual(runner._resources, [])

        loaded_dem = QgsRasterLayer(dem_path, "loaded terrain", "gdal")
        self.assertTrue(loaded_dem.isValid())
        QgsProject.instance().addMapLayer(loaded_dem)
        dem_alias = str(Path(temporary_directory.name) / "terrain-dem-alias.tif")
        os.symlink(dem_path, dem_alias)
        from ai_vectorizer.core.dem_pipeline import loaded_project_paths

        self.assertEqual(loaded_project_paths([dem_alias]), [dem_alias])

    def test_dem_publish_rechecks_outputs_loaded_during_async_run(self):
        from ai_vectorizer.core import dem_pipeline

        temporary_directory = tempfile.TemporaryDirectory()
        self.__class__._temporary_directories.append(temporary_directory)
        dem_path = str(Path(temporary_directory.name) / "terrain-dem.tif")
        hillshade_path = str(
            Path(temporary_directory.name) / "terrain-hillshade.tif"
        )
        dem_work_path = str(Path(temporary_directory.name) / ".dem-work.tif")
        hillshade_work_path = str(
            Path(temporary_directory.name) / ".hillshade-work.tif"
        )
        Path(dem_work_path).write_bytes(b"staged dem")
        Path(hillshade_work_path).write_bytes(b"staged hillshade")

        runner = dem_pipeline.DemPipelineRunner()
        runner._request = mock.Mock(
            dem_path=dem_path,
            hillshade_path=hillshade_path,
            crs_authid="EPSG:3857",
        )
        runner._raw_dem_path = None
        runner._dem_work_path = dem_work_path
        runner._hillshade_work_path = hillshade_work_path
        runner._dem_dimensions = (10, 10)
        failures = []
        runner.failed.connect(failures.append)

        with mock.patch.object(
            dem_pipeline,
            "loaded_project_paths",
            return_value=[dem_path],
        ), mock.patch.object(
            runner,
            "_validate_raster",
            return_value=(10, 10),
        ), mock.patch.object(runner, "_publish_outputs") as publish:
            runner._on_hillshade_finished(True, {}, mock.Mock(errors=[]))

        publish.assert_not_called()
        self.assertEqual(len(failures), 1)
        self.assertIn("loaded in QGIS", failures[0])
        self.assertFalse(Path(dem_work_path).exists())
        self.assertFalse(Path(hillshade_work_path).exists())

    def test_timers_are_tool_owned_and_deactivated_emits_once(self):
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "contours", "memory")
        tool = self._tool(layer)
        emissions = []
        tool.deactivated.connect(lambda: emissions.append(True))

        self.assertIs(tool._edge_cache_timer.parent(), tool)
        self.assertIs(tool._proposal_timer.parent(), tool)
        tool.activate()
        tool.deactivate()
        self.assertEqual(emissions, [True])

    def test_deactivated_canvas_owned_tool_is_scheduled_for_deletion(self):
        from ai_vectorizer.ui.main_dialog import AIVectorizerDock

        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "contours", "memory")
        tool = self._tool(layer)
        self.assertIs(tool.parent(), self.canvas)
        bands = [
            tool.preview_band,
            tool.confirm_band,
            tool.start_marker,
            tool.close_indicator,
            tool.checkpoint_markers,
            tool.snap_marker,
        ]
        scene_items = self.canvas.scene().items()
        self.assertTrue(all(band in scene_items for band in bands))
        destroyed = []
        tool.destroyed.connect(lambda *_args: destroyed.append(True))

        class DockHarness:
            def __init__(self, active_tool):
                self.active_tool = active_tool
                self._sender = active_tool
                self.idle = False

            def sender(self):
                return self._sender

            def _set_idle_ui(self):
                self.idle = True

        dock = DockHarness(tool)
        tool.deactivated.connect(
            lambda: AIVectorizerDock.on_tool_deactivated(dock)
        )
        self.canvas.setMapTool(tool)
        self.canvas.unsetMapTool(tool)
        self.assertIsNone(dock.active_tool)
        self.assertTrue(dock.idle)
        scene_items = self.canvas.scene().items()
        self.assertTrue(all(band not in scene_items for band in bands))

        event_loop = QEventLoop()
        QTimer.singleShot(0, event_loop.quit)
        execute = getattr(event_loop, "exec", None) or event_loop.exec_
        execute()
        self.assertEqual(destroyed, [True])

    def test_custom_toolbar_is_removed_on_unload(self):
        class Iface:
            def __init__(self):
                self.window = QMainWindow()
                self.removed_actions = []

            def mainWindow(self):
                return self.window

            def addToolBar(self, name):
                toolbar = QToolBar(name, self.window)
                self.window.addToolBar(toolbar)
                return toolbar

            def addPluginToVectorMenu(self, _menu, _action):
                pass

            def removePluginVectorMenu(self, _menu, action):
                self.removed_actions.append(action)

            def removeToolBarIcon(self, _action):
                pass

        iface = Iface()
        plugin = self.AIVectorizer(iface)
        plugin.initGui()
        toolbar = plugin.toolbar
        action = plugin.actions[0]
        self.assertIn(action, toolbar.actions())

        plugin.unload()

        self.assertIsNone(plugin.toolbar)
        self.assertEqual(plugin.actions, [])
        self.assertNotIn(action, toolbar.actions())

    def test_vector_writer_and_standard_path_compatibility(self):
        from qgis.PyQt.QtCore import QStandardPaths
        from ai_vectorizer.ui.main_dialog import (
            _standard_location,
            _write_vector_layer,
            _writer_no_error,
        )

        location = _standard_location("AppDataLocation")
        self.assertTrue(QStandardPaths.writableLocation(location))
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "output", "memory")
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "output.shp")
            result = _write_vector_layer(layer, path, self.crs)

            self.assertEqual(result[0], _writer_no_error())
            self.assertTrue(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
