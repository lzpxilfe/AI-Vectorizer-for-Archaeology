"""Production callback lifecycle logic and actual optional QgsTask checks."""

import ast
import math
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from ai_vectorizer.core.learned_prior import LearnedLinePrior


TOOL_SOURCE = Path(__file__).resolve().parents[1] / "ai_vectorizer/tools/smart_trace_tool.py"
DIALOG_SOURCE = Path(__file__).resolve().parents[1] / "ai_vectorizer/ui/main_dialog.py"


def _production_method(name, *, source=TOOL_SOURCE, owner_name="SmartTraceTool", extra=None):
    """Execute the real callback body independently of QGIS object setup.

    These callbacks use only task identity and Python state.  Extracting the
    actual source keeps ordinary CI checking their behaviour; separate tests
    below exercise the real QgsTask when QGIS is installed.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner_name)
    function = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "LearnedLinePrior": LearnedLinePrior, "np": np, "math": math,
        "__name__": "ai_vectorizer.tools.smart_trace_tool",
        "__package__": "ai_vectorizer.tools",
    }
    namespace.update(extra or {})
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[name]


def _dialog_method(name, **extra):
    return _production_method(name, source=DIALOG_SOURCE, owner_name="AIVectorizerDock", extra=extra)


class LearnedLifecycleLogicTests(unittest.TestCase):
    def setUp(self):
        self.callback = _production_method("_on_learned_prior_finished")
        self.identity = (7, 11, ("source-grid",), "a" * 64)
        self.task = SimpleNamespace(identity=self.identity)
        self.old_prior = LearnedLinePrior(np.zeros((5, 7)), np.zeros((5, 7)))
        self.new_prior = LearnedLinePrior(np.full((5, 7), 0.8), np.full((5, 7), 0.2))
        self.preview = [object()]
        self.accepted = [object(), object()]
        self.tree = object()
        self.tool = SimpleNamespace(
            _learned_task=self.task,
            _learned_cache_identity=mock.Mock(return_value=self.identity),
            _learned_cache_ready=mock.Mock(return_value=True),
            _learned_failed_cache=None,
            _learned_prior=self.old_prior,
            cached_ink_evidence=SimpleNamespace(shape=(5, 7)),
            _emit_learned_state=mock.Mock(),
            is_tracing=True,
            preview_path=self.preview,
            path_points=self.accepted,
            _livewire_tree=self.tree,
            _pending_livewire_accept_point=None,
            _manual_avoidance_draft_start=None,
            _manual_gap_bridge_preview_is_current=mock.Mock(return_value=False),
            _request_livewire_tree=mock.Mock(),
        )

    def _unchanged_geometry(self):
        self.assertIs(self.tool.preview_path, self.preview)
        self.assertIs(self.tool.path_points, self.accepted)
        self.assertIs(self.tool._livewire_tree, self.tree)
        self.tool._request_livewire_tree.assert_not_called()

    def test_obsolete_task_cannot_clear_or_publish_over_current_task(self):
        stale = SimpleNamespace(identity=self.identity)
        self.callback(self.tool, stale, True, self.new_prior, None)
        self.assertIs(self.tool._learned_task, self.task)
        self.assertIs(self.tool._learned_prior, self.old_prior)
        self.tool._emit_learned_state.assert_not_called()
        self._unchanged_geometry()

    def test_cache_model_or_generation_change_discards_late_result(self):
        for changed in (
            (8, 11, ("source-grid",), "a" * 64),
            (7, 12, ("source-grid",), "a" * 64),
            (7, 11, ("other-source",), "a" * 64),
            (7, 11, ("source-grid",), "b" * 64),
        ):
            with self.subTest(identity=changed):
                self.tool._learned_task = self.task
                self.tool._learned_cache_identity.return_value = changed
                self.callback(self.tool, self.task, True, self.new_prior, None)
                self.assertIsNone(self.tool._learned_task)
                self.assertIs(self.tool._learned_prior, self.old_prior)
                self.tool._emit_learned_state.assert_not_called()
                self._unchanged_geometry()

    def test_disposed_or_disabled_cache_discards_late_result(self):
        self.tool._learned_cache_ready.return_value = False
        self.callback(self.tool, self.task, True, self.new_prior, None)
        self.assertIs(self.tool._learned_prior, self.old_prior)
        self.tool._emit_learned_state.assert_not_called()
        self._unchanged_geometry()

    def test_failure_cancellation_or_wrong_provider_result_keeps_champion(self):
        for succeeded, prior, error in (
            (False, None, RuntimeError("runtime failed")),
            (False, self.new_prior, None),
            (True, LearnedLinePrior(np.ones((2, 2)), np.zeros((2, 2))), None),
            (True, SimpleNamespace(shape=(5, 7)), None),
            (True, object(), None),
        ):
            with self.subTest(result=(succeeded, type(prior).__name__)):
                self.tool._learned_task = self.task
                self.tool._emit_learned_state.reset_mock()
                self.callback(self.tool, self.task, succeeded, prior, error)
                self.assertIs(self.tool._learned_prior, self.old_prior)
                self.assertEqual(self.tool._learned_failed_cache, self.identity)
                self.assertEqual(self.tool._emit_learned_state.call_args.args[0], "fallback")
                self._unchanged_geometry()

    def test_current_prior_waits_while_green_preview_is_shown(self):
        self.callback(self.tool, self.task, True, self.new_prior, None)
        self.assertIs(self.tool._learned_prior, self.new_prior)
        self._unchanged_geometry()

    def test_pending_click_manual_gap_and_avoidance_draft_keep_visible_snapshot(self):
        self.tool.preview_path = []
        for guard in ("pending", "gap", "avoidance"):
            with self.subTest(guard=guard):
                self.tool._learned_task = self.task
                self.tool._pending_livewire_accept_point = object() if guard == "pending" else None
                self.tool._manual_avoidance_draft_start = object() if guard == "avoidance" else None
                self.tool._manual_gap_bridge_preview_is_current.return_value = guard == "gap"
                self.callback(self.tool, self.task, True, self.new_prior, None)
                self.assertIs(self.tool._learned_prior, self.new_prior)
                self.tool._request_livewire_tree.assert_not_called()
                self.assertIs(self.tool._livewire_tree, self.tree)
                self.assertIs(self.tool.path_points, self.accepted)

    def test_safe_initial_anchor_prepares_next_tree_and_idle_never_does(self):
        self.tool.preview_path = []
        self.callback(self.tool, self.task, True, self.new_prior, None)
        self.tool._request_livewire_tree.assert_called_once_with(force=True)
        self.tool._request_livewire_tree.reset_mock()
        self.tool._learned_task = self.task
        self.tool.is_tracing = False
        self.callback(self.tool, self.task, True, self.new_prior, None)
        self.tool._request_livewire_tree.assert_not_called()

    def test_task_cancel_invalidates_generation_even_if_qt_task_is_gone(self):
        task = SimpleNamespace(cancel=mock.Mock(side_effect=RuntimeError("deleted Qt task")))
        tool = SimpleNamespace(_learned_task=task, _learned_generation=7)
        _production_method("_cancel_learned_task")(tool)
        self.assertIsNone(tool._learned_task)
        self.assertEqual(tool._learned_generation, 8)
        task.cancel.assert_called_once_with()

    def test_each_new_anchor_can_request_prior_after_reset_and_off_never_does(self):
        # An unchanged Ink cache can survive saving one contour. A new anchor
        # must still ask for the reset prior, without requiring a canvas pan.
        tool = SimpleNamespace(learned_guide_requested=True, _livewire_disabled=True,
                               _request_learned_prior=mock.Mock())
        request = _production_method("_request_livewire_tree")
        self.assertFalse(request(tool))
        tool._request_learned_prior.assert_called_once_with()
        tool.learned_guide_requested=False
        tool._request_learned_prior.reset_mock()
        self.assertFalse(request(tool))
        tool._request_learned_prior.assert_not_called()


class LearnedNativeGridGuardTests(unittest.TestCase):
    """Unsupported GDAL grids stop before provider reads or worker creation."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        local = Path(self.temporary.name) / "native.tif"
        local.write_bytes(b"grid-metadata-test")
        self.driver = SimpleNamespace(ShortName="GTiff")
        self.transform = [0, 1, 0, 1000, 0, -1]
        self.bands = [SimpleNamespace(DataType=1, GetColorInterpretation=mock.Mock(return_value=i)) for i in (3, 4, 5)]
        self.source = SimpleNamespace(
            RasterXSize=1000, RasterYSize=1000, RasterCount=3,
            GetDriver=lambda: self.driver, GetGeoTransform=lambda: self.transform,
            GetRasterBand=lambda i: self.bands[i-1],
        )
        self.gdal = SimpleNamespace(Open=mock.Mock(return_value=self.source), GA_ReadOnly=0,
                                    GDT_Byte=1, GCI_RedBand=3, GCI_GreenBand=4, GCI_BlueBand=5)
        self.provider = SimpleNamespace(bandCount=lambda: 3)
        extent = SimpleNamespace(width=lambda:1000, height=lambda:1000,
                                 xMinimum=lambda:0, yMaximum=lambda:1000)
        self.layer = SimpleNamespace(width=lambda:1000, height=lambda:1000,
                                    extent=lambda:extent, source=lambda:str(local),
                                    dataProvider=lambda:self.provider)
        self.identity = (1, 2, "model")
        self.tool = SimpleNamespace(
            learned_guide_requested=True, learned_engine=object(),
            _learned_cache_ready=mock.Mock(return_value=True),
            _learned_cache_identity=lambda:self.identity,
            _learned_failed_cache=None, _learned_identity=None, _learned_task=None,
            _learned_prior=object(), cache_tile_origin=(0,0),
            cached_ink_evidence=SimpleNamespace(shape=(5,7)), raster_layer=self.layer,
            _emit_learned_state=mock.Mock(), _tr=lambda ko,en:en,
            _on_learned_prior_finished=mock.Mock(),
        )
        self.read = mock.Mock(side_effect=lambda provider, rectangle, w, h, **kwargs:
                              (None, [np.zeros((h,w), dtype=np.uint8) for _ in range(3)], True))
        self.make_task = mock.Mock(side_effect=lambda *args:SimpleNamespace(arguments=args))
        self.manager = SimpleNamespace(addTask=mock.Mock())
        self.request = _production_method("_request_learned_prior", extra={
            "QgsRectangle":lambda *args:args, "read_raster_bands_with_native":self.read,
            "_LearnedPriorTask":self.make_task,
            "QgsApplication":SimpleNamespace(taskManager=lambda:self.manager),
        })
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict(sys.modules, {"osgeo":SimpleNamespace(gdal=self.gdal)}).start()

    def test_native_byte_rgb_snapshots_are_read_on_calling_thread_and_frozen(self):
        thread = threading.get_ident()
        self.read.side_effect = lambda provider, rectangle, w, h, **kwargs: (
            self.assertEqual(threading.get_ident(), thread),
            [np.full((h,w), i, dtype=np.uint8) for i in range(3)], True)
        self.assertTrue(self.request(self.tool))
        self.make_task.assert_called_once()
        args = self.make_task.call_args.args
        self.assertEqual(args[2:5], ((0,0,7,5),(1000,1000),self.identity))
        self.assertEqual(list(args[1]), [(0,0,448,448)])
        snapshot = args[1][(0,0,448,448)]
        self.assertEqual(snapshot.shape,(448,448,3))
        self.assertFalse(snapshot.flags.writeable)
        self.assertEqual(snapshot[0,0].tolist(),[0,1,2])
        self.manager.addTask.assert_called_once_with(self.tool._learned_task)

    def test_nonlocal_virtual_rotated_stretched_and_non_rgb_sources_fall_back(self):
        changes = {
            "remote":lambda:setattr(self.layer,"source",lambda:"https://example.invalid/map.tif"),
            "unreadable":lambda:setattr(self.gdal.Open,"return_value",None),
            "virtual":lambda:setattr(self.driver,"ShortName","VRT"),
            "rotated":lambda:self.transform.__setitem__(2,.1),
            "size":lambda:setattr(self.source,"RasterXSize",999),
            "non_byte":lambda:setattr(self.bands[0],"DataType",2),
            "palette":lambda:setattr(self.source,"RasterCount",1),
            "band_order":lambda:self.bands[0].GetColorInterpretation.configure_mock(return_value=5),
            "resolution":lambda:self.transform.__setitem__(1,2),
            "origin":lambda:self.transform.__setitem__(0,.5),
            "provider_bands":lambda:setattr(self.provider,"bandCount",lambda:4),
        }
        for label, change in changes.items():
            with self.subTest(grid=label):
                self.setUp()
                old_prior = self.tool._learned_prior
                change()
                self.assertFalse(self.request(self.tool))
                self.read.assert_not_called()
                self.make_task.assert_not_called()
                self.manager.addTask.assert_not_called()
                self.assertIs(self.tool._learned_prior,old_prior)
                self.assertEqual(self.tool._learned_failed_cache,self.identity)
                self.assertEqual(self.tool._emit_learned_state.call_args.args[0],"fallback")

    def test_render_stretch_does_not_enter_background_inference(self):
        self.read.return_value = (None,[np.zeros((448,448),dtype=np.uint8)]*3,False)
        self.read.side_effect = None
        self.assertFalse(self.request(self.tool))
        self.make_task.assert_not_called()
        self.manager.addTask.assert_not_called()

    def test_large_map_coordinates_do_not_relax_exact_grid_origin(self):
        self.layer.extent=lambda:SimpleNamespace(width=lambda:1000,height=lambda:1000,
                                                xMinimum=lambda:1e9,yMaximum=lambda:1000)
        self.transform[0]=1e9+.5
        self.assertFalse(self.request(self.tool))
        self.read.assert_not_called()
        self.make_task.assert_not_called()

    def test_off_and_repeated_identity_skip_native_source_reads(self):
        self.tool.learned_guide_requested=False
        self.assertFalse(self.request(self.tool))
        self.gdal.Open.assert_not_called()
        self.tool.learned_guide_requested=True
        self.tool._learned_identity=self.identity
        self.assertFalse(self.request(self.tool))
        self.gdal.Open.assert_not_called()


class LearnedDialogLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.task = SimpleNamespace(generation=7, bundle_path="/local/bundle", cancel=mock.Mock())
        self.engine = SimpleNamespace(manifest={"provenance":{"synthetic_diagnostic":False}})
        self.settings = SimpleNamespace(setValue=mock.Mock())
        self.dock = SimpleNamespace(
            _learned_prepare_generation=7, learned_prepare_task=self.task,
            learned_engine=None, _shutting_down=False, active_tool=None,
            learned_import_btn=SimpleNamespace(setEnabled=mock.Mock()),
            learned_guide_check=SimpleNamespace(isChecked=lambda:True),
            _on_learned_state_changed=mock.Mock(), _tr=lambda ko,en:en,
        )
        self.callback = _dialog_method("_on_learned_model_prepared",QSettings=lambda:self.settings)

    def test_cancel_import_reenables_button_and_invalidates_late_callback(self):
        _dialog_method("_cancel_learned_prepare")(self.dock)
        self.assertEqual(self.dock._learned_prepare_generation,8)
        self.assertIsNone(self.dock.learned_prepare_task)
        self.task.cancel.assert_called_once()
        self.dock.learned_import_btn.setEnabled.assert_called_once_with(True)
        self.callback(self.dock,self.task,True,self.engine,None)
        self.assertIsNone(self.dock.learned_engine)
        self.settings.setValue.assert_not_called()
        self.dock._on_learned_state_changed.assert_not_called()

    def test_shutdown_and_generation_mismatch_discard_preparation(self):
        for shutdown,generation in ((True,7),(False,8)):
            with self.subTest(shutdown=shutdown,generation=generation):
                self.dock.learned_prepare_task=self.task
                self.dock._shutting_down=shutdown
                self.dock._learned_prepare_generation=generation
                self.callback(self.dock,self.task,True,self.engine,None)
                self.assertIsNone(self.dock.learned_engine)
                self.settings.setValue.assert_not_called()

    def test_current_success_publishes_path_only_and_active_trace_keeps_button_locked(self):
        self.dock.active_tool=object()
        self.callback(self.dock,self.task,True,self.engine,None)
        self.assertIs(self.dock.learned_engine,self.engine)
        self.settings.setValue.assert_called_once_with("ArchaeoTrace/learnedBundle","/local/bundle")
        self.dock.learned_import_btn.setEnabled.assert_called_once_with(False)

    def test_zero_assistance_cancels_model_preparation_and_releases_engine(self):
        self.dock.learned_engine=self.engine
        self.dock.freedom_label=SimpleNamespace(setText=mock.Mock())
        self.dock._cancel_recovery_prepare=mock.Mock()
        self.dock._release_recovery_engine=mock.Mock()
        self.dock._refresh_recovery_availability=mock.Mock()
        self.dock._cancel_learned_prepare=lambda:_dialog_method("_cancel_learned_prepare")(self.dock)
        _dialog_method("_on_assist_strength_changed")(self.dock,0)
        self.assertIsNone(self.dock.learned_engine)
        self.assertIsNone(self.dock.learned_prepare_task)
        self.task.cancel.assert_called_once()
        self.dock._on_learned_state_changed.assert_called_once_with("off")

    def test_cancel_during_shutdown_never_reenables_import(self):
        self.dock._shutting_down=True
        self.task.cancel.side_effect=RuntimeError("deleted Qt task")
        _dialog_method("_cancel_learned_prepare")(self.dock)
        self.assertIsNone(self.dock.learned_prepare_task)
        self.dock.learned_import_btn.setEnabled.assert_called_once_with(False)


class LearnedDialogGuidancePolicyTests(unittest.TestCase):
    def setUp(self):
        self.engine = SimpleNamespace(
            guidance_parameters=(0.8, 0.0), has_selected_guidance=False,
            manifest=mock.Mock(name="manifest_not_revalidated_by_ui"),
        )
        self.status = SimpleNamespace(setText=mock.Mock(), setToolTip=mock.Mock())
        self.dock = SimpleNamespace(
            learned_engine=self.engine, learned_status=self.status,
            _tr=lambda ko, en: en,
        )

    def test_trace_parameters_use_prepared_policy_and_missing_model_is_support_only(self):
        read = _dialog_method("_learned_guidance_parameters")
        self.assertEqual(read(self.dock), (0.8, 0.0))
        self.engine.guidance_parameters = (0.4, 4.0)
        self.engine.has_selected_guidance = True
        self.assertEqual(read(self.dock), (0.4, 4.0))
        self.engine.manifest.assert_not_called()
        self.dock.learned_engine = None
        self.assertEqual(read(self.dock), (0.8, 0.0))

    def test_ready_and_active_status_and_tooltip_explain_support_only_in_both_languages(self):
        update = _dialog_method("_on_learned_state_changed")
        for language, expected in (
            ("en", "Contour support only · no text avoidance setting"),
            ("ko", "등고선 지지만 사용 · 문자 회피 설정 없음"),
        ):
            self.dock._tr = (lambda ko, en: ko) if language == "ko" else (lambda ko, en: en)
            for state in ("ready", "active"):
                with self.subTest(language=language, state=state):
                    update(self.dock, state, "pilot")
                    self.assertIn(expected, self.status.setText.call_args.args[0])
                    self.assertIn(expected, self.status.setToolTip.call_args.args[0])
        self.engine.has_selected_guidance = True
        update(self.dock, "ready")
        self.assertNotIn("문자 회피 설정 없음", self.status.setText.call_args.args[0])
        self.assertNotIn("문자 회피 설정 없음", self.status.setToolTip.call_args.args[0])
        self.engine.has_selected_guidance = False
        update(self.dock, "off")
        self.assertNotIn("문자 회피 설정 없음", self.status.setText.call_args.args[0])


try:
    from qgis.core import QgsApplication
    HAS_QGIS = True
except ImportError:
    HAS_QGIS = False


@unittest.skipUnless(HAS_QGIS, "QGIS bindings are required for real QgsTask checks")
class LearnedQgsTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.owns_app = QgsApplication.instance() is None
        cls.application = QgsApplication.instance() or QgsApplication([], False)
        if cls.owns_app:
            cls.application.initQgis()
        from ai_vectorizer.tools.smart_trace_tool import _LearnedPriorTask
        cls.Task = _LearnedPriorTask

    @classmethod
    def tearDownClass(cls):
        if cls.owns_app:
            cls.application.exitQgis()

    def test_worker_reads_only_frozen_snapshots_and_publishes_on_finished(self):
        image = np.zeros((5, 7, 3), dtype=np.uint8)
        image.setflags(write=False)
        bounds = (0, 0, 7, 5)
        prior = LearnedLinePrior(np.ones((5, 7)), np.zeros((5, 7)))
        thread_ids = []
        def predict(read_tile, received_bounds, raster_size, cancel_check):
            thread_ids.append(threading.get_ident())
            self.assertIs(read_tile(bounds), image)
            self.assertFalse(read_tile(bounds).flags.writeable)
            self.assertEqual(received_bounds, bounds)
            self.assertEqual(raster_size, (7, 5))
            self.assertFalse(cancel_check())
            return prior
        callback = mock.Mock()
        task = self.Task(SimpleNamespace(predict_crop=predict), {bounds: image}, bounds, (7, 5), (3, "model"), callback)
        results = []
        worker = threading.Thread(target=lambda: results.append(task.run()))
        worker.start()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results, [True])
        self.assertNotEqual(thread_ids, [threading.get_ident()])
        callback.assert_not_called()
        task.finished(True)
        callback.assert_called_once_with(task, True, prior, None)

    def test_pre_cancelled_task_skips_provider_and_late_cancel_is_not_success(self):
        engine = SimpleNamespace(predict_crop=mock.Mock())
        callback = mock.Mock()
        task = self.Task(engine, {}, (0, 0, 7, 5), (7, 5), (3, "model"), callback)
        task.cancel()
        self.assertFalse(task.run())
        engine.predict_crop.assert_not_called()
        task.finished(False)
        callback.assert_called_once_with(task, False, None, None)
        prior = LearnedLinePrior(np.ones((5, 7)), np.zeros((5, 7)))
        current = self.Task(engine, {}, (0, 0, 7, 5), (7, 5), (3, "model"), mock.Mock())
        def cancel_after_native_call(*args, **kwargs):
            current.cancel()
            return prior
        engine.predict_crop.side_effect = cancel_after_native_call
        self.assertFalse(current.run())
        self.assertIs(current.prior, prior)

    def test_support_only_model_keeps_user_avoidance_in_actual_livewire_worker(self):
        from ai_vectorizer.core.learned_model import model_guidance_parameters
        from ai_vectorizer.core.line_evidence import LineEvidence
        from ai_vectorizer.core.livewire import LiveWireConfig, is_livewire_available
        from ai_vectorizer.core.trace_guidance import TraceGuidance
        from ai_vectorizer.tools.smart_trace_tool import _LiveWireTreeTask

        if not is_livewire_available():
            self.skipTest("Actual Live-Wire worker requires SciPy")
        shape = (64, 64)
        centerline = np.zeros(shape, dtype=bool)
        centerline[20, 8:57] = True
        centerline[40, 8:57] = True
        centerline[20:41, (8, 56)] = True
        evidence = LineEvidence(centerline.astype(np.float32), centerline)
        image = np.full(shape + (3,), 245, dtype=np.uint8)
        image[centerline] = 25
        avoidance = np.zeros(shape, dtype=np.float32)
        avoidance[:36, 27:38] = 1.0
        manual = TraceGuidance(avoidance)
        # An unconfigured model can emit high other-ink scores; it cannot
        # suppress or replace the user's independent avoidance selection.
        prior = LearnedLinePrior(np.zeros(shape), np.ones(shape))
        support_weight, avoidance_weight = model_guidance_parameters(None)
        config = LiveWireConfig(
            max_window_size=64, target_snap_radius=0,
            direction_cost_weight=0, heading_cost_weight=0,
            learned_support_cap=support_weight,
            learned_avoidance_cost_weight=avoidance_weight,
        )
        paths = []
        for guidance in (None, manual):
            task = _LiveWireTreeTask(
                image=image, edges=centerline.astype(np.uint8) * 255,
                evidence=evidence, guidance=guidance, anchor_pixel=(8, 20),
                incoming_direction=None, strength=1.0, generation=1,
                config=config, callback=mock.Mock(), learned_prior=prior,
            )
            self.assertTrue(task.run(), task.error)
            self.assertIs(task.guidance, guidance)
            paths.append(task.tree.trace((56, 20)))
        self.assertEqual(max(y for x, y in paths[0]), 20)
        self.assertGreaterEqual(max(y for x, y in paths[1]), 36)
        np.testing.assert_array_equal(manual.avoidance_score, avoidance)

    def test_actual_map_tool_infers_bundle_policy_and_preserves_explicit_diagnostic_costs(self):
        from qgis.core import QgsCoordinateReferenceSystem, QgsVectorLayer
        from qgis.gui import QgsMapCanvas
        from ai_vectorizer.core.learned_model import build_bundle_manifest
        from ai_vectorizer.tools.smart_trace_tool import SmartTraceTool

        provenance = {
            "dataset_manifest_sha256": "a" * 64,
            "training_code_commit": "b" * 40, "export_code_commit": "c" * 40,
            "architecture": "policy-diagnostic", "seed": 0, "license_spdx": "MIT",
        }
        unconfigured = build_bundle_manifest(b"policy-only-fixture", provenance)
        configured = build_bundle_manifest(b"policy-only-fixture", dict(
            provenance,
            selected_guidance={
                "support": 0.4, "avoidance": 4,
                "calibration_sha256": "d" * 64,
                "holdout_report_sha256": "e" * 64, "pilot_gates_passed": True,
            },
        ))
        canvas = QgsMapCanvas()
        crs = QgsCoordinateReferenceSystem("EPSG:3857")
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "policy test", "memory")
        try:
            for manifest, overrides, expected in (
                (unconfigured, {}, (0.8, 0.0)),
                (configured, {}, (0.4, 4.0)),
                (unconfigured, {"learned_avoidance_weight": 6.0}, (0.8, 6.0)),
            ):
                with self.subTest(expected=expected):
                    tool = SmartTraceTool(
                        canvas, SimpleNamespace(crs=lambda: crs), layer,
                        freehand=True, learned_engine=SimpleNamespace(manifest=manifest),
                        **overrides,
                    )
                    try:
                        config = tool._livewire_config()
                        self.assertEqual(
                            (config.learned_support_cap, config.learned_avoidance_cost_weight),
                            expected,
                        )
                    finally:
                        tool.dispose()
                        tool.deleteLater()
        finally:
            canvas.deleteLater()
            self.application.processEvents()


if __name__ == "__main__":
    unittest.main()
