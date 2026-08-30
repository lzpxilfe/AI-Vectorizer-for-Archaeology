"""QGIS-free source contracts for the optional Smart Recovery UI path."""

from pathlib import Path
import ast
import unittest

from ai_vectorizer.recovery import (
    RECOVERY_STATE_ENHANCED,
    RECOVERY_STATE_INK,
    RECOVERY_STATE_INK_FALLBACK,
    RECOVERY_STATE_RECOVERING,
    RECOVERY_STATES,
    require_recovery_state,
)


ROOT = Path(__file__).resolve().parents[1]
DIALOG_PATH = ROOT / "ai_vectorizer" / "ui" / "main_dialog.py"
TOOL_PATH = ROOT / "ai_vectorizer" / "tools" / "smart_trace_tool.py"


def _source(path):
    return path.read_text(encoding="utf-8")


def _function_source(path, function_name, class_name=None):
    source = _source(path)
    tree = ast.parse(source)
    nodes = tree.body
    if class_name is not None:
        owner = next(
            node
            for node in nodes
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        nodes = owner.body
    node = next(
        node
        for node in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    return ast.get_source_segment(source, node)


class RecoveryStateTests(unittest.TestCase):
    def test_public_state_vocabulary_is_exact(self):
        self.assertEqual(
            RECOVERY_STATES,
            {
                "Ink",
                "Recovering",
                "Enhanced",
                "Ink fallback",
            },
        )
        self.assertEqual(require_recovery_state(RECOVERY_STATE_INK), "Ink")
        self.assertEqual(
            require_recovery_state(RECOVERY_STATE_RECOVERING),
            "Recovering",
        )
        self.assertEqual(
            require_recovery_state(RECOVERY_STATE_ENHANCED),
            "Enhanced",
        )
        self.assertEqual(
            require_recovery_state(RECOVERY_STATE_INK_FALLBACK),
            "Ink fallback",
        )
        with self.assertRaises(ValueError):
            require_recovery_state("model wins")


class SmartRecoverySourceContractTests(unittest.TestCase):
    def test_basic_ui_defaults_to_ink_and_hides_legacy_models(self):
        setup = _function_source(DIALOG_PATH, "setup_ui", "AIVectorizerDock")
        model_items = _function_source(
            DIALOG_PATH,
            "_model_items",
            "AIVectorizerDock",
        )

        self.assertIn("self.smart_recovery_check.setChecked(False)", setup)
        self.assertIn("self.freehand_check.toggled.connect", setup)
        self.assertIn("self.advanced_check.setChecked(False)", setup)
        self.assertIn("self.advanced_group.setVisible(False)", setup)
        self.assertIn("advanced_layout.addLayout(model_layout)", setup)
        for index_name in (
            "MODEL_IDX_INK",
            "MODEL_IDX_LSD",
            "MODEL_IDX_HED",
            "MODEL_IDX_MOBILE_SAM",
            "MODEL_IDX_SAM",
            "MODEL_IDX_LEGACY_CANNY",
        ):
            self.assertIn(index_name, model_items)

    def test_only_explicit_install_task_calls_network_fetch(self):
        source = _source(DIALOG_PATH)
        tree = ast.parse(source)
        owners = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            segment = ast.get_source_segment(source, node) or ""
            if "fetch_bundle" in segment:
                owners.append(node.name)

        self.assertEqual(owners, ["run"])
        install = _function_source(
            DIALOG_PATH,
            "install_recovery_model",
            "AIVectorizerDock",
        )
        load = _function_source(
            DIALOG_PATH,
            "_load_recovery_engine_offline",
            "AIVectorizerDock",
        )
        toggle = _function_source(
            DIALOG_PATH,
            "_on_smart_recovery_toggled",
            "AIVectorizerDock",
        )
        self.assertIn("_RecoveryInstallTask", install)
        self.assertNotIn("fetch_bundle", load)
        self.assertNotIn("fetch_bundle", toggle)

    def test_model_hashing_and_onnx_initialization_run_only_in_prepare_task(self):
        source = _source(DIALOG_PATH)
        prepare_run = _function_source(
            DIALOG_PATH,
            "run",
            "_RecoveryPrepareTask",
        )
        load = _function_source(
            DIALOG_PATH,
            "_load_recovery_engine_offline",
            "AIVectorizerDock",
        )
        refresh = _function_source(
            DIALOG_PATH,
            "_refresh_recovery_availability",
            "AIVectorizerDock",
        )

        self.assertIn("EfficientSAMRecoveryEngine.inspect", prepare_run)
        self.assertIn("EfficientSAMRecoveryEngine(self.cache_root)", prepare_run)
        self.assertNotIn("EfficientSAMRecoveryEngine", load)
        self.assertNotIn("EfficientSAMRecoveryEngine", refresh)
        self.assertIn("_start_recovery_prepare", refresh)
        self.assertEqual(source.count("EfficientSAMRecoveryEngine.inspect"), 1)

    def test_runtime_and_model_install_are_separate_explicit_guidance(self):
        source = _source(DIALOG_PATH)
        refresh = _function_source(
            DIALOG_PATH,
            "_refresh_recovery_availability",
            "AIVectorizerDock",
        )
        self.assertIn('find_spec("onnxruntime")', source)
        self.assertIn('onnxruntime>=1.17,<2', source)
        self.assertIn("recovery_runtime_cmd.setVisible(True)", refresh)
        self.assertIn("recovery_install_btn.setVisible", refresh)

    def test_zero_percent_cancels_optional_model_preparation_and_stays_cursor_only(self):
        assist = _function_source(
            DIALOG_PATH,
            "_on_assist_strength_changed",
            "AIVectorizerDock",
        )
        refresh = _function_source(
            DIALOG_PATH,
            "_refresh_recovery_availability",
            "AIVectorizerDock",
        )
        trace = _function_source(
            DIALOG_PATH,
            "toggle_trace_tool",
            "AIVectorizerDock",
        )
        needs_cache = _function_source(
            TOOL_PATH,
            "_needs_edge_cache",
            "SmartTraceTool",
        )

        self.assertIn("int(value) <= 0", assist)
        self.assertIn("self._cancel_recovery_prepare()", assist)
        self.assertIn("self._release_recovery_engine()", assist)
        self.assertIn("self.freedom_slider.value() <= 0", refresh)
        self.assertIn("exact cursor", refresh)
        self.assertIn("self._cancel_recovery_prepare()", trace)
        self.assertIn("self.edge_weight > 0.0", needs_cache)

    def test_missing_recovery_model_does_not_block_ink_tool_start(self):
        toggle = _function_source(
            DIALOG_PATH,
            "toggle_trace_tool",
            "AIVectorizerDock",
        )
        missing = toggle.index("if recovery_engine is None:")
        construction = toggle.index("tool = SmartTraceTool(")
        self.assertLess(missing, construction)
        self.assertIn("RECOVERY_STATE_INK_FALLBACK", toggle)
        self.assertIn("recovery_engine=recovery_engine", toggle)
        self.assertIn(
            "recovery_state_callback=self._on_recovery_state_changed",
            toggle,
        )

    def test_ink_evidence_runs_in_cancelable_task_and_publishes_current_only(self):
        task_run = _function_source(TOOL_PATH, "run", "_InkEvidenceTask")
        update = _function_source(
            TOOL_PATH,
            "update_edge_cache",
            "SmartTraceTool",
        )
        finished = _function_source(
            TOOL_PATH,
            "_on_ink_evidence_finished",
            "SmartTraceTool",
        )
        self.assertLess(
            task_run.index("self.detector.detect_edges"),
            task_run.index("detect_ink_evidence"),
        )
        self.assertIn("cancel_check=self.isCanceled", task_run)
        self.assertIn("except InkEvidenceCancelled", task_run)
        self.assertIn("_InkEvidenceTask(", update)
        self.assertIn("QgsApplication.taskManager().addTask(task)", update)
        self.assertNotIn("detect_ink_evidence(", update)
        self.assertIn("task.generation == self._ink_evidence_generation", finished)
        self.assertIn("task.cache_identity == self._pending_cache_identity", finished)
        self.assertIn("task.cache_identity == current_identity", finished)
        self.assertIn("fallback_edges", finished)
        self.assertIn("RECOVERY_STATE_INK_FALLBACK", finished)

    def test_ink_task_call_supplies_every_required_cache_snapshot(self):
        """Keep constructor growth from silently disabling Ink at runtime."""

        source = _source(TOOL_PATH)
        tree = ast.parse(source)
        task_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "_InkEvidenceTask"
        )
        initializer = next(
            node
            for node in task_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        required_keywords = {
            argument.arg
            for argument, default in zip(
                initializer.args.kwonlyargs,
                initializer.args.kw_defaults,
            )
            if default is None
        }
        update = next(
            node
            for node in next(
                owner
                for owner in tree.body
                if isinstance(owner, ast.ClassDef)
                and owner.name == "SmartTraceTool"
            ).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "update_edge_cache"
        )
        task_call = next(
            node
            for node in ast.walk(update)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_InkEvidenceTask"
        )
        supplied_keywords = {keyword.arg for keyword in task_call.keywords}

        self.assertEqual(required_keywords - supplied_keywords, set())
        self.assertTrue(
            {
                "fallback_image",
                "fallback_rgb_image",
                "fallback_cache_identity",
                "fallback_cache_extent",
                "fallback_output_size",
                "evidence_image",
                "recovery_image",
                "recovery_compatible",
                "recovery_disabled_reason",
            }.issubset(supplied_keywords)
        )

    def test_smart_recovery_is_gated_to_native_byte_cache_and_raster_changes(self):
        update = _function_source(
            TOOL_PATH,
            "update_edge_cache",
            "SmartTraceTool",
        )
        schedule = _function_source(
            TOOL_PATH,
            "_schedule_smart_recovery",
            "SmartTraceTool",
        )
        listeners = _function_source(
            TOOL_PATH,
            "_set_source_lifecycle_listeners",
            "SmartTraceTool",
        )

        self.assertIn("np.dtype(np.uint8)", update)
        self.assertIn("recovery_compatible=recovery_compatible", update)
        self.assertIn("not self._recovery_cache_compatible", schedule)
        self.assertIn('getattr(self.raster_layer, "dataChanged"', listeners)
        self.assertIn("self._on_raster_data_changed", listeners)

    def test_ink_task_cannot_publish_after_deactivation_or_source_change(self):
        deactivate = _function_source(
            TOOL_PATH,
            "deactivate",
            "SmartTraceTool",
        )
        invalidated = _function_source(
            TOOL_PATH,
            "_on_source_layer_invalidated",
            "SmartTraceTool",
        )
        current = _function_source(
            TOOL_PATH,
            "_current_ink_cache_identity",
            "SmartTraceTool",
        )

        self.assertLess(
            deactivate.index("self._is_active = False"),
            deactivate.index("self._clear_edge_cache()"),
        )
        self.assertIn("self._clear_edge_cache()", invalidated)
        self.assertIn("not self._is_active", current)
        self.assertIn("self.raster_layer.isValid()", current)
        self.assertIn("self._canvas_extent_in_raster_crs()", current)
        self.assertIn("self._cache_request_identity", current)

    def test_ink_v2_requires_an_exact_bounded_source_grid(self):
        extent = _function_source(
            TOOL_PATH,
            "_ink_evidence_extent_and_origin",
            "SmartTraceTool",
        )
        plan = _function_source(
            TOOL_PATH,
            "_ink_evidence_sampling_plan",
            "SmartTraceTool",
        )
        update = _function_source(
            TOOL_PATH,
            "update_edge_cache",
            "SmartTraceTool",
        )
        task_run = _function_source(TOOL_PATH, "run", "_InkEvidenceTask")

        self.assertIn("visible_x0 // tile_size", extent)
        self.assertIn("visible_x1 + tile_size - 1", extent)
        self.assertIn("tile_x0 - halo", extent)
        self.assertIn("tile_x1 + halo", extent)
        self.assertIn("tile_y0 - halo", extent)
        self.assertIn("tile_y1 + halo", extent)
        self.assertIn("exact_width = int(round", plan)
        self.assertIn("exact_height = int(round", plan)
        self.assertIn("exact_width <= self.CACHE_MAX_DIMENSION", plan)
        self.assertIn("exact_height <= self.CACHE_MAX_DIMENSION", plan)
        self.assertIn("return (exact_width, exact_height), True", plan)
        self.assertIn("Zoom in to enable continuous Ink evidence", plan)
        self.assertIn("enable_evidence=enable_ink_evidence", update)
        self.assertIn(
            "evidence_disabled_reason=evidence_disabled_reason",
            update,
        )
        self.assertLess(
            task_run.index("if not self.enable_evidence"),
            task_run.index("detect_evidence("),
        )

    def test_ink_v1_fallback_keeps_the_frozen_visible_extent_contract(self):
        update = _function_source(
            TOOL_PATH,
            "update_edge_cache",
            "SmartTraceTool",
        )
        finished = _function_source(
            TOOL_PATH,
            "_on_ink_evidence_finished",
            "SmartTraceTool",
        )

        self.assertIn("visible_ext = extent.intersect(raster_ext)", update)
        self.assertIn("fallback_bands = read_raster_bands(", update)
        fallback_read = update.index("fallback_bands = read_raster_bands(")
        task_creation = update.index("task = _InkEvidenceTask(")
        self.assertLess(fallback_read, task_creation)
        self.assertIn("fallback_cache_extent=(", update)
        self.assertIn("fallback_output_size=(fallback_width, fallback_height)", update)
        self.assertIn("fallback_rgb_image=fallback_rgb_image", update)
        self.assertIn("published_image = task.fallback_rgb_image", finished)
        self.assertIn("QgsRectangle(*task.fallback_cache_extent)", finished)
        self.assertIn("output_size = task.fallback_output_size", finished)
        self.assertIn("published_identity = task.fallback_cache_identity", finished)

    def test_livewire_receives_evidence_and_recovery_uses_conservative_arbiter(self):
        livewire_run = _function_source(TOOL_PATH, "run", "_LiveWireTreeTask")
        recovery_run = _function_source(TOOL_PATH, "run", "_RecoveryPreviewTask")
        schedule = _function_source(
            TOOL_PATH,
            "_schedule_smart_recovery",
            "SmartTraceTool",
        )
        callback = _function_source(
            TOOL_PATH,
            "_on_recovery_preview_finished",
            "SmartTraceTool",
        )
        self.assertIn("evidence=self.evidence", livewire_run)
        self.assertIn("build_corridor_cost_map", recovery_run)
        self.assertIn("crop_line_evidence", recovery_run)
        self.assertIn("self.window_bounds", recovery_run)
        self.assertIn("local_start", recovery_run)
        self.assertIn("raw_challenger", recovery_run)
        self.assertIn("for value in self.start_pixel", recovery_run)
        self.assertIn("for value in self.target_pixel", recovery_run)
        self.assertLess(
            recovery_run.index("smooth_pixel_path("),
            recovery_run.index("arbitrate_routes("),
        )
        self.assertIn("arbitrate_routes", recovery_run)
        self.assertIn("recovery_gate", _source(TOOL_PATH))
        self.assertIn("force=False", schedule)
        self.assertLess(
            schedule.index("self._recovery_generation += 1"),
            schedule.index("self._build_recovery_request("),
        )
        self.assertIn("self._recovery_preview_identity = preview_identity", schedule)
        self.assertIn("Ink evidence is confident; recovery was not run.", schedule)
        self.assertIn(
            "task.preview_identity == self._recovery_preview_identity",
            callback,
        )
        self.assertIn("selection.accepted", callback)
        self.assertNotIn("_pixel_path_to_map", callback)
        self.assertIn("for x, y in challenger_path[1:]", callback)
        self.assertIn("RECOVERY_STATE_ENHANCED", callback)
        self.assertIn("RECOVERY_STATE_INK_FALLBACK", callback)

    def test_recovery_prompts_use_only_current_anchor_and_target_positives(self):
        prompts = _function_source(
            TOOL_PATH,
            "_build_recovery_prompts",
            "SmartTraceTool",
        )
        request = _function_source(
            TOOL_PATH,
            "_build_recovery_request",
            "SmartTraceTool",
        )
        self.assertIn("self.path_points[-1]", prompts)
        self.assertIn("build_recovery_prompt_tensors", prompts)
        self.assertNotIn("recent_points", prompts)
        self.assertNotIn("self.path_points[:-1]", prompts)
        self.assertIn("_build_recovery_prompts", request)
        self.assertNotIn("_build_sam_prompts", request)

    def test_retry_is_explicit_and_engine_lifecycle_is_bounded(self):
        retry = _function_source(
            TOOL_PATH,
            "retry_current_segment",
            "SmartTraceTool",
        )
        toggle = _function_source(
            DIALOG_PATH,
            "_on_smart_recovery_toggled",
            "AIVectorizerDock",
        )
        cleanup = _function_source(
            DIALOG_PATH,
            "cleanup",
            "AIVectorizerDock",
        )
        install_finished = _function_source(
            DIALOG_PATH,
            "_on_recovery_install_finished",
            "AIVectorizerDock",
        )
        self.assertIn("force=True", retry)
        self.assertIn("RECOVERY_STATE_ENHANCED", retry)
        self.assertIn("already enhanced", retry)
        self.assertIn("self._release_recovery_engine()", toggle)
        freehand_toggle = _function_source(
            DIALOG_PATH,
            "_on_freehand_toggled",
            "AIVectorizerDock",
        )
        self.assertIn("self.smart_recovery_check.setChecked(False)", freehand_toggle)
        trace_toggle = _function_source(
            DIALOG_PATH,
            "toggle_trace_tool",
            "AIVectorizerDock",
        )
        recovery_branch = trace_toggle.index("if smart_recovery:")
        release = trace_toggle.index(
            "self._release_recovery_engine()",
            recovery_branch,
        )
        self.assertGreater(release, recovery_branch)
        self.assertIn("install_task.cancel()", cleanup)
        self.assertIn("self._shutting_down", install_finished)


if __name__ == "__main__":
    unittest.main()
