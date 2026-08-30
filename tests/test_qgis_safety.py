"""QGIS integration source contracts runnable outside a QGIS installation.

These checks intentionally parse, rather than import, the plugin modules so
ordinary Python CI can guard the safety-critical QGIS paths.
"""

from pathlib import Path
import ast
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _source(path):
    return (ROOT / path).read_text(encoding="utf-8")


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


class QgisSafetySourceTests(unittest.TestCase):
    def test_custom_toolbar_is_removed_and_actions_are_disposed(self):
        unload = _function_source(
            "ai_vectorizer/plugin.py",
            "unload",
            "AIVectorizer",
        )

        self.assertIn("self.toolbar.removeAction(action)", unload)
        self.assertIn("removeToolBar(self.toolbar)", unload)
        self.assertIn("action.deleteLater()", unload)
        self.assertIn("self.actions.clear()", unload)

    def test_active_trace_uses_current_selector_and_locks_configuration(self):
        toggle = _function_source(
            "ai_vectorizer/ui/main_dialog.py",
            "toggle_trace_tool",
            "AIVectorizerDock",
        )
        lock = _function_source(
            "ai_vectorizer/ui/main_dialog.py",
            "_set_trace_configuration_enabled",
            "AIVectorizerDock",
        )

        self.assertIn("output_layer = self.vector_combo.currentLayer()", toggle)
        self.assertIn("unsupported_output_reason(output_layer)", toggle)
        self.assertIn("self._set_trace_configuration_enabled(False)", toggle)
        self.assertIn(
            "layerChanged.connect(self.on_raster_layer_selected)",
            _source("ai_vectorizer/ui/main_dialog.py"),
        )
        for control in ("layer_combo", "vector_combo", "model_combo", "freedom_slider"):
            self.assertIn(f"self.{control}", lock)

    def test_qt6_standard_path_uses_scoped_enum_fallback(self):
        dialog_source = _source("ai_vectorizer/ui/main_dialog.py")
        helper = _function_source(
            "ai_vectorizer/ui/main_dialog.py",
            "_standard_location",
        )

        self.assertIn("QStandardPaths.StandardLocation", helper)
        self.assertNotIn(
            "QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)",
            dialog_source,
        )

    def test_trace_feature_creation_keeps_layer_defaults(self):
        build_feature = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "_build_feature",
            "SmartTraceTool",
        )

        self.assertIn("QgsVectorLayerUtils.createFeature", build_feature)
        self.assertIn("and feature[id_idx] is None", build_feature)
        self.assertNotIn("setAttributes", build_feature)
        self.assertNotIn("[None]", build_feature)

    def test_hard_field_constraints_are_checked_before_feature_save(self):
        add_geometry = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "_add_geometry_feature",
            "SmartTraceTool",
        )
        validate = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "_hard_constraint_failures",
            "SmartTraceTool",
        )

        self.assertIn("self._validate_feature_constraints", add_geometry)
        self.assertIn("QgsVectorLayerUtils.validateAttribute", validate)
        self.assertIn("HARD_CONSTRAINT", validate)

    def test_resume_updates_geometry_and_elevation_together(self):
        save = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "save_to_layer",
            "SmartTraceTool",
        )

        self.assertIn("attribute_changes[elev_idx] = float(elevation)", save)
        self.assertIn("self._run_edit_command(", save)
        self.assertIn("self._ensure_field_in_edit_buffer(", save)
        self.assertIn("self._update_feature_in_edit_buffer(", save)
        self.assertNotIn("self._ensure_field(", save)

        update = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "_update_feature_in_edit_buffer",
            "SmartTraceTool",
        )
        self.assertIn("layer.changeGeometry(", update)
        self.assertIn("layer.changeAttributeValue(", update)
        self.assertNotIn("dataProvider()", update)

    def test_user_feature_writes_use_qgis_edit_buffer_only(self):
        add = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "_add_feature",
            "SmartTraceTool",
        )
        ensure_field = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "_ensure_field_in_edit_buffer",
            "SmartTraceTool",
        )
        add_geometry = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "_add_geometry_feature",
            "SmartTraceTool",
        )
        constructor = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "__init__",
            "SmartTraceTool",
        )

        self.assertIn("self._run_edit_command(", add)
        self.assertIn("layer.addFeature(", add)
        self.assertNotIn("dataProvider()", add)
        self.assertIn("layer.addAttribute(", ensure_field)
        self.assertNotIn("dataProvider()", ensure_field)
        self.assertIn("self._run_edit_command(", add_geometry)
        self.assertIn("self._ensure_field_in_edit_buffer(", add_geometry)
        self.assertIn("layer.addFeature(", add_geometry)
        self.assertIn("self._ensure_edit_session(self.vector_layer)", constructor)

    def test_idle_shortcuts_leave_qgis_undo_reachable(self):
        source = _source("ai_vectorizer/tools/smart_trace_tool.py")
        key_press = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "keyPressEvent",
            "SmartTraceTool",
        )
        activate = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "activate",
            "SmartTraceTool",
        )

        self.assertIn("event.ignore()", key_press)
        self.assertIn("if self.is_tracing:", key_press)
        self.assertNotIn("_set_undo_enabled", activate)
        self.assertNotIn("setEnabled(False)", source)

    def test_hed_assets_are_redirected_to_persistent_profile_storage(self):
        constructor = _function_source(
            "ai_vectorizer/ui/main_dialog.py",
            "__init__",
            "AIVectorizerDock",
        )
        configure = _function_source(
            "ai_vectorizer/ui/main_dialog.py",
            "_configure_hed_storage",
            "AIVectorizerDock",
        )

        self.assertIn("self._configure_hed_storage()", constructor)
        self.assertIn("EdgeDetector.configure_hed_storage", configure)
        self.assertIn("cls._sam_models_dir()", configure)

    def test_diagnostic_report_uses_an_exclusive_temp_descriptor(self):
        export = _function_source(
            "ai_vectorizer/ui/main_dialog.py",
            "export_sam_report",
            "AIVectorizerDock",
        )

        self.assertIn("tempfile.mkstemp(", export)
        self.assertIn("os.fdopen(", export)
        self.assertIn("os.fsync(", export)
        self.assertNotIn("tempfile.gettempdir()", export)
        self.assertNotIn("with open(out_path", export)

    def test_edge_preview_uses_a_private_unique_directory(self):
        preview = _function_source(
            "ai_vectorizer/ui/main_dialog.py",
            "preview_edges",
            "AIVectorizerDock",
        )

        self.assertIn("tempfile.TemporaryDirectory(", preview)
        self.assertIn('"preview.tif"', preview)
        self.assertIn("self._preview_store.track", preview)
        self.assertIn("finally:", preview)
        self.assertIn("_cleanup_directory", preview)
        self.assertNotIn("tempfile.gettempdir()", preview)
        self.assertNotIn("edge_preview_", preview)

    def test_sam_check_reports_pinned_integrity_and_accepts_missing_remote_data(self):
        check = _function_source(
            "ai_vectorizer/ui/main_dialog.py",
            "check_sam_update",
            "AIVectorizerDock",
        )
        dialog_source = _source("ai_vectorizer/ui/main_dialog.py")

        self.assertIn('info.get("remote") or {}', check)
        self.assertIn('status in ("invalid", "update_available")', check)
        self.assertIn("pinned-file verification", check)
        self.assertIn("Re-download", check)
        self.assertIn("Verify Selected SAM Model", dialog_source)
        self.assertNotIn("Checking latest model", check)
        self.assertNotIn("A newer", check)

    def test_loaded_shapefile_is_blocked_before_overwrite(self):
        create = _function_source(
            "ai_vectorizer/ui/main_dialog.py",
            "create_shp_layer",
            "AIVectorizerDock",
        )
        source_path = _function_source(
            "ai_vectorizer/core/dem_pipeline.py",
            "layer_file_path",
        )

        self.assertIn("_project_layers_using_path(path)", create)
        self.assertIn("return", create)
        self.assertIn("decodeUri", source_path)
        self.assertIn("QUrl", source_path)

    def test_dem_rechecks_loaded_outputs_immediately_before_publish(self):
        finish = _function_source(
            "ai_vectorizer/core/dem_pipeline.py",
            "_on_hillshade_finished",
            "DemPipelineRunner",
        )
        loaded = _function_source(
            "ai_vectorizer/ui/dem_dialog.py",
            "_loaded_output_paths",
            "DemBuildDialog",
        )

        guard_index = finish.index("loaded_project_paths(")
        publish_index = finish.index("self._publish_outputs(")
        self.assertLess(guard_index, publish_index)
        self.assertIn("raise DemInputError", finish)
        self.assertIn("loaded_project_paths(paths)", loaded)

    def test_active_tool_tracks_crs_and_layer_source_lifecycle(self):
        activate = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "activate",
            "SmartTraceTool",
        )
        crs_changed = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "_on_cache_crs_changed",
            "SmartTraceTool",
        )
        source_changed = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "_on_source_layer_invalidated",
            "SmartTraceTool",
        )

        self.assertIn("_set_coordinate_crs_listeners(True)", activate)
        self.assertIn("_set_source_lifecycle_listeners(True)", activate)
        self.assertIn("self.reset_tracing()", crs_changed)
        self.assertIn("self._refresh_crs_transforms()", crs_changed)
        self.assertIn("unsetMapTool(self)", source_changed)

    def test_z_and_m_layers_are_rejected_before_2d_geometry_creation(self):
        check = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "unsupported_output_reason",
            "SmartTraceTool",
        )

        self.assertIn("QgsWkbTypes.hasZ", check)
        self.assertIn("QgsWkbTypes.hasM", check)

    def test_spot_layer_reuse_requires_plugin_ownership_marker(self):
        getter = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "get_or_create_spot_layer",
            "SmartTraceTool",
        )

        self.assertIn("customProperty", getter)
        self.assertIn("setCustomProperty", getter)
        self.assertNotIn("layer.name()", getter)

    def test_tool_timers_are_owned_by_tool_and_deactivation_emits_once(self):
        initializer = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "__init__",
            "SmartTraceTool",
        )
        deactivate = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "deactivate",
            "SmartTraceTool",
        )

        self.assertGreaterEqual(initializer.count("QTimer(self)"), 2)
        self.assertNotIn("self.deactivated.emit", deactivate)

        dock_deactivated = _function_source(
            "ai_vectorizer/ui/main_dialog.py",
            "on_tool_deactivated",
            "AIVectorizerDock",
        )
        self.assertIn("tool = self.sender()", dock_deactivated)
        self.assertIn("tool.dispose()", dock_deactivated)
        self.assertIn("tool.deleteLater()", dock_deactivated)

        dispose = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "dispose",
            "SmartTraceTool",
        )
        self.assertIn("scene.removeItem(item)", dispose)
        self.assertIn('setattr(self, attribute_name, None)', dispose)

    def test_permanent_livewire_failure_disables_rescheduling(self):
        callback = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "_on_livewire_tree_finished",
            "SmartTraceTool",
        )
        request = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "_request_livewire_tree",
            "SmartTraceTool",
        )

        self.assertIn("isinstance(error, LiveWireUnavailable)", callback)
        self.assertIn("self._livewire_disabled = True", callback)
        self.assertIn("self._livewire_disabled", request)
        self.assertIn("self._livewire_failed_anchor == anchor_pixel", request)

    def test_fast_livewire_click_waits_for_current_anchor_tree(self):
        defer = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "_defer_click_until_livewire_ready",
            "SmartTraceTool",
        )
        callback = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "_on_livewire_tree_finished",
            "SmartTraceTool",
        )
        press = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "canvasPressEvent",
            "SmartTraceTool",
        )

        self.assertIn("self._livewire_tree_is_ready()", defer)
        self.assertIn("self.edge_method != EdgeDetector.METHOD_INK", defer)
        self.assertIn("self._pending_livewire_accept_point = target", defer)
        self.assertIn("self._pending_livewire_auto_accept = bool(auto_accept)", defer)
        self.assertIn("pending_accept = self._pending_livewire_accept_point", callback)
        self.assertIn("request_tree=False", callback)
        self.assertIn("schedule_recovery=True", callback)
        self.assertIn("self._pending_livewire_recovery_identity", callback)
        self.assertIn("recovery_started", callback)
        self.assertIn("self._commit_visible_livewire_segment", callback)
        self.assertIn("self._defer_click_until_livewire_ready", press)
        self.assertIn("The previous Ink point is still preparing", press)

    def test_acceptance_preserves_enhanced_preview_and_cache_failure_drains_click(self):
        press = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "canvasPressEvent",
            "SmartTraceTool",
        )
        present = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "_present_livewire_cursor_preview",
            "SmartTraceTool",
        )
        evidence_finished = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "_on_ink_evidence_finished",
            "SmartTraceTool",
        )

        self.assertIn("accept_visible_enhanced", press)
        self.assertIn("not accept_visible_enhanced", press)
        self.assertGreaterEqual(press.count("schedule_recovery=False"), 2)
        self.assertIn("schedule_recovery=True", present)
        self.assertIn("and schedule_recovery", present)
        self.assertIn("self._resolve_pending_livewire_fallback", evidence_finished)
        self.assertIn("nearby=False", evidence_finished)

    def test_path_rewind_invalidates_async_segment_state(self):
        refresh = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "_refresh_after_path_rewind",
            "SmartTraceTool",
        )
        undo_checkpoint = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "undo_to_checkpoint",
            "SmartTraceTool",
        )
        undo_points = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "undo_points",
            "SmartTraceTool",
        )

        self.assertIn("self._invalidate_recovery", refresh)
        self.assertIn("self._clear_preview()", refresh)
        self.assertIn("self._cancel_livewire_task()", refresh)
        self.assertIn("self._livewire_generation += 1", refresh)
        self.assertIn("self._pending_livewire_accept_point = None", refresh)
        self.assertIn("self._request_livewire_tree(force=True)", refresh)
        self.assertIn("self._refresh_after_path_rewind()", undo_checkpoint)
        self.assertIn("self._refresh_after_path_rewind()", undo_points)

    def test_polygon_close_uses_a_transactional_candidate(self):
        press = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "canvasPressEvent",
            "SmartTraceTool",
        )
        save_candidate = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "_save_closed_path_candidate",
            "SmartTraceTool",
        )
        take_auto_path = _function_source(
            "ai_vectorizer/tools/smart_trace_tool.py",
            "_take_or_prepare_auto_path",
            "SmartTraceTool",
        )

        self.assertNotIn("self.path_points.extend(closing_path)", press)
        self.assertIn(
            "if not near_start or not accept_visible_enhanced",
            press,
        )
        self.assertIn("self._save_closed_path_candidate", press)
        self.assertIn("confirmed_path = self.path_points", save_candidate)
        self.assertIn("finally:", save_candidate)
        self.assertIn("self.path_points = confirmed_path", save_candidate)
        self.assertNotIn("self.path_points.extend", take_auto_path)

    def test_dem_processing_uses_exact_selected_project_layer(self):
        reference = _function_source(
            "ai_vectorizer/core/dem_pipeline.py",
            "_layer_reference",
        )

        self.assertIn("return layer.id()", reference)
        self.assertNotIn("layer.source()", reference)


if __name__ == "__main__":
    unittest.main()
