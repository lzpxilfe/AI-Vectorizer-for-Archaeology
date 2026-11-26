# -*- coding: utf-8 -*-
"""Dock controls for named tracing presets and per-map project memory.

The format and its rules live in :mod:`ai_vectorizer.core.trace_preset`.
This mixin only moves settings between that format, the dock widgets, the
user's QGIS settings and the QGIS project.
"""

from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
)
from qgis.PyQt.QtCore import QSettings
from qgis.core import QgsProject

from ..config import (
    MODEL_IDX_HED,
    MODEL_IDX_INK,
    MODEL_IDX_LEGACY_CANNY,
    MODEL_IDX_LSD,
    MODEL_IDX_MOBILE_SAM,
    MODEL_IDX_SAM,
)
from ..core.dependencies import is_cv2_available
from ..core.trace_preset import (
    MAX_CONTOUR_INTERVAL,
    PresetError,
    TracePreset,
    find,
    library_from_json,
    library_to_json,
    plan_application,
    preset_from_json,
    remove,
    rename,
    upsert,
)


PRESET_SETTINGS_KEY = "ArchaeoTrace/tracePresets"
PRESET_BACKUP_SETTINGS_KEY = "ArchaeoTrace/tracePresetsUnreadable"
PROJECT_SCOPE = "ArchaeoTrace"
PROJECT_KEY_PREFIX = "rasterPresets/r"
MAX_PRESET_FILE_BYTES = 64 * 1024

METHOD_BY_MODEL = {
    MODEL_IDX_INK: "ink",
    MODEL_IDX_LSD: "lsd",
    MODEL_IDX_HED: "hed",
    MODEL_IDX_MOBILE_SAM: "mobile_sam",
    MODEL_IDX_SAM: "sam",
    MODEL_IDX_LEGACY_CANNY: "legacy_canny",
}
MODEL_BY_METHOD = {method: index for index, method in METHOD_BY_MODEL.items()}

_FIELD_LABELS = {
    "method": ("추적 방식", "tracing method"),
    "auto_path": ("자동 경로", "Auto Path"),
    "smart_recovery": ("Smart Recovery", "Smart Recovery"),
    "learned_guide": ("학습 가이드", "learned guide"),
}
_SKIP_REASONS = {
    "opencv_missing": ("OpenCV 없음", "OpenCV is not installed"),
    "hed_model_missing": ("HED 모델 없음", "the HED model is missing"),
    "sam_backend_missing": ("SAM 실행 환경 없음", "the SAM runtime is not installed"),
    "recovery_runtime_missing": ("ONNX Runtime 없음", "ONNX Runtime is not installed"),
    "learned_model_missing": ("가져온 학습 모델 없음", "no learned model has been imported"),
}
_ERROR_TEXT = {
    "invalid_name": ("이름이 올바르지 않습니다", "the name is not valid"),
    "newer_version": (
        "더 새로운 ArchaeoTrace에서 만든 프리셋입니다. 플러그인을 업데이트하세요",
        "it was made by a newer ArchaeoTrace; update the plugin",
    ),
    "unknown_schema": ("ArchaeoTrace 프리셋 파일이 아닙니다", "it is not an ArchaeoTrace preset"),
    "unknown_field": ("알 수 없는 항목이 있습니다", "it has fields this version does not know"),
    "missing_field": ("필수 항목이 없습니다", "required fields are missing"),
    "invalid_field": ("값이 올바르지 않습니다", "a value is not valid"),
    "conflicting_fields": ("함께 쓸 수 없는 옵션이 있습니다", "it combines options that cannot be used together"),
    "malformed": ("JSON 형식이 아닙니다", "it is not valid JSON"),
    "duplicate_name": ("같은 이름이 이미 있습니다", "that name is already used"),
    "not_found": ("프리셋을 찾지 못했습니다", "the preset was not found"),
    "too_many": ("프리셋이 너무 많습니다", "there are too many presets"),
}


def _enum_value(owner, legacy_name, scope_name):
    legacy = getattr(owner, legacy_name, None)
    if legacy is not None:
        return legacy
    return getattr(getattr(owner, scope_name), legacy_name)


def _message_button(name):
    return _enum_value(QMessageBox, name, "StandardButton")


class TracePresetMixin:
    """Preset controls mixed into :class:`AIVectorizerDock`."""

    # -- state -----------------------------------------------------------

    def _init_preset_state(self):
        self._presets = []
        self._preset_load_error = None
        self._preset_raw_unreadable = None
        self._series_ink_rgb = None
        self._elevation_history = []
        self._preset_raster_id = None
        self._preset_output_id = None
        self._preset_message = None

    def _load_presets(self):
        raw = QSettings().value(PRESET_SETTINGS_KEY, "")
        text = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
        try:
            self._presets = library_from_json(text)
            self._preset_load_error = None
            self._preset_raw_unreadable = None
        except PresetError as exc:
            # Keep the unreadable value untouched until the user saves a
            # preset; it is then copied to a backup key, never discarded.
            self._presets = []
            self._preset_load_error = exc
            self._preset_raw_unreadable = text

    def _store_presets(self, presets):
        text = library_to_json(presets)
        settings = QSettings()
        if self._preset_raw_unreadable:
            settings.setValue(PRESET_BACKUP_SETTINGS_KEY, self._preset_raw_unreadable)
            self._preset_raw_unreadable = None
            self._preset_load_error = None
        settings.setValue(PRESET_SETTINGS_KEY, text)
        self._presets = list(presets)

    # -- widgets ---------------------------------------------------------

    def _setup_preset_ui(self, parent_layout):
        self.preset_group = QGroupBox()
        layout = QVBoxLayout()
        self.preset_desc = QLabel()
        self.preset_desc.setWordWrap(True)
        self.preset_desc.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.preset_desc)

        self.preset_combo = QComboBox()
        self.preset_combo.setSizeAdjustPolicy(_enum_value(
            QComboBox, "AdjustToMinimumContentsLengthWithIcon", "SizeAdjustPolicy",
        ))
        self.preset_combo.setMinimumContentsLength(8)
        layout.addWidget(self.preset_combo)
        row = QHBoxLayout()
        self.preset_apply_btn = QPushButton()
        self.preset_apply_btn.clicked.connect(self.apply_selected_preset)
        row.addWidget(self.preset_apply_btn)
        self.preset_save_btn = QPushButton()
        self.preset_save_btn.clicked.connect(self.save_preset)
        row.addWidget(self.preset_save_btn)
        self.preset_more_btn = QToolButton()
        self.preset_more_btn.setText("⋯")
        self.preset_more_btn.setPopupMode(
            _enum_value(QToolButton, "InstantPopup", "ToolButtonPopupMode")
        )
        self.preset_menu = QMenu(self.preset_more_btn)
        self.preset_rename_action = self.preset_menu.addAction("")
        self.preset_rename_action.triggered.connect(self.rename_preset)
        self.preset_delete_action = self.preset_menu.addAction("")
        self.preset_delete_action.triggered.connect(self.delete_preset)
        self.preset_menu.addSeparator()
        self.preset_export_action = self.preset_menu.addAction("")
        self.preset_export_action.triggered.connect(self.export_preset)
        self.preset_import_action = self.preset_menu.addAction("")
        self.preset_import_action.triggered.connect(self.import_preset)
        self.preset_more_btn.setMenu(self.preset_menu)
        row.addWidget(self.preset_more_btn)
        layout.addLayout(row)

        interval_row = QHBoxLayout()
        self.contour_interval_label = QLabel()
        self.contour_interval_label.setWordWrap(True)
        interval_row.addWidget(self.contour_interval_label, 1)
        self.contour_interval_spin = QDoubleSpinBox()
        self.contour_interval_spin.setDecimals(2)
        self.contour_interval_spin.setRange(0.0, MAX_CONTOUR_INTERVAL)
        self.contour_interval_spin.setSingleStep(5.0)
        self.contour_interval_spin.setValue(0.0)
        self.contour_interval_spin.valueChanged.connect(self._on_contour_interval_changed)
        interval_row.addWidget(self.contour_interval_spin)
        layout.addLayout(interval_row)

        self.preset_status = QLabel()
        self.preset_status.setWordWrap(True)
        self.preset_status.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.preset_status)
        self.preset_group.setLayout(layout)
        parent_layout.addWidget(self.preset_group)
        self.preset_combo.currentIndexChanged.connect(self._update_preset_buttons)

        self._load_presets()
        if self._preset_load_error is not None:
            self._set_preset_message((
                "저장된 프리셋을 읽지 못했습니다 ({error}). 새로 저장하면 기존 값은 백업으로 남깁니다.",
                "Saved presets could not be read ({error}). Saving a preset keeps the old value as a backup.",
            ), "warning", error=self._preset_error_text(self._preset_load_error))

    def _preset_widgets(self):
        return (
            self.preset_combo,
            self.preset_apply_btn,
            self.preset_save_btn,
            self.preset_more_btn,
            self.contour_interval_spin,
        )

    def _apply_preset_language(self):
        self.preset_group.setTitle(self._tr("지도 시리즈 설정 (프리셋)", "Map Series Settings (Presets)"))
        self.preset_desc.setText(self._tr(
            "같은 양식으로 인쇄된 도엽에 쓸 설정을 이름 붙여 저장합니다. 선 색상과 등고선 간격도 함께 저장됩니다.",
            "Save the settings for sheets printed in the same style under a name, including the ink color and contour interval.",
        ))
        self.preset_apply_btn.setText(self._tr("적용", "Apply"))
        self.preset_save_btn.setText(self._tr("저장", "Save"))
        self.preset_more_btn.setToolTip(self._tr("프리셋 관리", "Manage presets"))
        self.preset_rename_action.setText(self._tr("이름 바꾸기…", "Rename…"))
        self.preset_delete_action.setText(self._tr("삭제…", "Delete…"))
        self.preset_export_action.setText(self._tr("파일로 내보내기…", "Export to File…"))
        self.preset_import_action.setText(self._tr("파일에서 가져오기…", "Import from File…"))
        self.contour_interval_label.setText(self._tr("등고선 간격 (m)", "Contour interval (m)"))
        self.contour_interval_spin.setSpecialValueText(self._tr("사용 안 함", "Off"))
        self.contour_interval_spin.setToolTip(self._tr(
            "새 등고선의 해발값 창에 직전 값 ± 간격을 제안합니다. 값은 항상 직접 확인합니다.",
            "Suggests the previous elevation ± this interval for each new contour. You always confirm the value.",
        ))
        self._refresh_preset_combo()
        self._render_preset_message()

    def _refresh_preset_combo(self, select_name=None):
        current = select_name if select_name is not None else self.preset_combo.currentData()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem(self._tr("(프리셋 선택)", "(choose a preset)"), None)
        for preset in self._presets:
            self.preset_combo.addItem(preset.name, preset.name)
        index = self.preset_combo.findData(current) if current else 0
        self.preset_combo.setCurrentIndex(max(0, index))
        self.preset_combo.blockSignals(False)
        self._update_preset_buttons()

    def _update_preset_buttons(self, *_args):
        selected = self.preset_combo.currentData() is not None
        idle = getattr(self, "active_tool", None) is None and self.preset_combo.isEnabled()
        self.preset_apply_btn.setEnabled(idle and selected)
        self.preset_rename_action.setEnabled(selected)
        self.preset_delete_action.setEnabled(selected)
        self.preset_export_action.setEnabled(selected)

    def _set_preset_message(self, texts, tone="info", **values):
        self._preset_message = (texts, tone, values)
        self._render_preset_message()

    def _render_preset_message(self):
        if self._preset_message is None:
            self.preset_status.setText("")
            return
        (ko, en), tone, values = self._preset_message
        rendered = {
            key: (self._tr(*value) if isinstance(value, tuple) else value)
            for key, value in values.items()
        }
        self.preset_status.setText(self._tr(ko, en).format(**rendered))
        colours = {"info": "green", "warning": "#e67e22", "error": "red"}
        self.preset_status.setStyleSheet(f"color: {colours.get(tone, 'green')}; font-size: 10px;")

    def _preset_error_text(self, exc):
        ko, en = _ERROR_TEXT.get(getattr(exc, "code", ""), ("오류", "error"))
        detail = getattr(exc, "detail", "")
        return (f"{ko}: {detail}" if detail else ko, f"{en}: {detail}" if detail else en)

    def _selected_preset(self):
        name = self.preset_combo.currentData()
        return find(self._presets, name) if name else None

    # -- moving settings -------------------------------------------------

    def _contour_interval_value(self):
        value = float(self.contour_interval_spin.value())
        return value if value > 0.0 else None

    def _on_contour_interval_changed(self, _value):
        self._update_preset_buttons()

    def _preset_from_ui(self, name):
        model_idx = self.model_combo.currentIndex()
        freehand = self.freehand_check.isChecked()
        method = METHOD_BY_MODEL.get(model_idx, "ink")
        return TracePreset(
            name=name,
            method=method,
            assist_strength=int(self.freedom_slider.value()),
            smart_recovery=bool(self.smart_recovery_check.isChecked()) and method == "ink" and not freehand,
            freehand=freehand,
            auto_path=bool(self.auto_path_check.isChecked()) and not freehand,
            learned_guide=bool(self.learned_guide_check.isChecked()) and method == "ink" and not freehand,
            ink_color_rgb=self._series_ink_rgb,
            contour_interval=self._contour_interval_value(),
        )

    def _preset_availability(self):
        availability = {
            "opencv": False, "hed_model": False, "mobile_sam_backend": False,
            "sam_backend": False, "recovery_runtime": False, "learned_model": False,
        }
        try:
            availability["opencv"] = bool(is_cv2_available())
            if availability["opencv"]:
                from ..core.edge_detector import EdgeDetector

                availability["hed_model"] = bool(EdgeDetector.get_hed_runtime_status().get("ok"))
        except Exception as exc:  # an optional runtime probe must not block presets
            self._log_nonfatal_ui_error("Preset OpenCV/HED check", exc)
        for key, model_idx in (("mobile_sam_backend", MODEL_IDX_MOBILE_SAM), ("sam_backend", MODEL_IDX_SAM)):
            try:
                availability[key] = bool(self._sam_backend_available(model_idx))
            except Exception as exc:
                self._log_nonfatal_ui_error("Preset SAM check", exc)
        try:
            availability["recovery_runtime"] = bool(self._recovery_runtime_available())
        except Exception as exc:
            self._log_nonfatal_ui_error("Preset recovery check", exc)
        availability["learned_model"] = bool(
            self.learned_engine is not None or getattr(self, "_learned_bundle_path", "")
        )
        return availability

    def _apply_preset_settings(self, preset):
        """Set the dock widgets from a preset; return the skipped options."""

        plan = plan_application(preset, self._preset_availability())
        settings = plan.settings
        # Turn the mutually exclusive options off first, then let the normal
        # toggle handlers keep the dock consistent as each value is set.
        self.freehand_check.setChecked(False)
        self.learned_guide_check.setChecked(False)
        self.smart_recovery_check.setChecked(False)
        self.model_combo.setCurrentIndex(MODEL_BY_METHOD.get(settings.method, MODEL_IDX_INK))
        self.freedom_slider.setValue(settings.assist_strength)
        self.auto_path_check.setChecked(settings.auto_path)
        self.smart_recovery_check.setChecked(settings.smart_recovery)
        self.learned_guide_check.setChecked(settings.learned_guide)
        self.freehand_check.setChecked(settings.freehand)
        if settings.method != "ink" or settings.auto_path or settings.learned_guide:
            self.advanced_check.setChecked(True)
        self.contour_interval_spin.setValue(settings.contour_interval or 0.0)
        self._series_ink_rgb = settings.ink_color_rgb
        self._on_ink_color_changed(None, False)
        return plan.skipped

    def _skipped_text(self, skipped):
        parts_ko = []
        parts_en = []
        for field, reason in skipped:
            field_ko, field_en = _FIELD_LABELS.get(field, (field, field))
            reason_ko, reason_en = _SKIP_REASONS.get(reason, (reason, reason))
            parts_ko.append(f"{field_ko}({reason_ko})")
            parts_en.append(f"{field_en} ({reason_en})")
        return "; ".join(parts_ko), "; ".join(parts_en)

    def _report_applied(self, preset, skipped, origin):
        if origin == "project":
            texts = (
                "이 지도에 저장된 설정 '{name}'을 적용했습니다.",
                "Applied the settings saved with this map: '{name}'.",
            )
        else:
            texts = ("프리셋 '{name}'을 적용했습니다.", "Applied preset '{name}'.")
        if skipped:
            texts = (
                texts[0] + " 이 컴퓨터에서 쓸 수 없어 건너뛴 항목: {skipped}",
                texts[1] + " Skipped because unavailable here: {skipped}.",
            )
            self._set_preset_message(texts, "warning", name=preset.name, skipped=self._skipped_text(skipped))
        else:
            self._set_preset_message(texts, "info", name=preset.name)

    # -- actions ---------------------------------------------------------

    def apply_selected_preset(self):
        if self.active_tool is not None:
            return False
        preset = self._selected_preset()
        if preset is None:
            return False
        skipped = self._apply_preset_settings(preset)
        self._report_applied(preset, skipped, "library")
        self._remember_project_preset(self.layer_combo.currentLayer(), preset.name)
        return True

    def _ask_preset_name(self, title, default=""):
        name, accepted = QInputDialog.getText(
            self, title, self._tr("프리셋 이름", "Preset name"), text=default,
        )
        return name if accepted else None

    def _confirm(self, title, text):
        answer = QMessageBox.question(
            self, title, text,
            _message_button("Yes") | _message_button("No"),
            _message_button("No"),
        )
        return answer == _message_button("Yes")

    def _show_preset_error(self, exc):
        ko, en = self._preset_error_text(exc)
        QMessageBox.warning(self, self._tr("프리셋", "Preset"), self._tr(ko, en))

    def save_preset(self):
        if self.active_tool is not None:
            return False
        selected = self.preset_combo.currentData() or ""
        name = self._ask_preset_name(self._tr("프리셋 저장", "Save Preset"), selected)
        if name is None:
            return False
        try:
            preset = self._preset_from_ui(name)
            if find(self._presets, preset.name) is not None and not self._confirm(
                self._tr("프리셋 저장", "Save Preset"),
                self._tr(
                    "'{name}' 프리셋을 현재 설정으로 덮어쓸까요?",
                    "Replace preset '{name}' with the current settings?",
                ).format(name=preset.name),
            ):
                return False
            self._store_presets(upsert(self._presets, preset))
        except PresetError as exc:
            self._show_preset_error(exc)
            return False
        self._refresh_preset_combo(select_name=find(self._presets, preset.name).name)
        self._set_preset_message(("프리셋 '{name}'을 저장했습니다.", "Saved preset '{name}'."), name=preset.name)
        self._remember_project_preset(self.layer_combo.currentLayer(), preset.name)
        return True

    def rename_preset(self):
        preset = self._selected_preset()
        if preset is None:
            return False
        name = self._ask_preset_name(self._tr("프리셋 이름 바꾸기", "Rename Preset"), preset.name)
        if name is None:
            return False
        try:
            renamed = rename(self._presets, preset.name, name)
            self._store_presets(renamed)
        except PresetError as exc:
            self._show_preset_error(exc)
            return False
        new_name = find(self._presets, name).name
        self._refresh_preset_combo(select_name=new_name)
        self._set_preset_message(
            ("프리셋 이름을 '{name}'(으)로 바꿨습니다.", "Renamed the preset to '{name}'."), name=new_name,
        )
        return True

    def delete_preset(self):
        preset = self._selected_preset()
        if preset is None:
            return False
        if not self._confirm(
            self._tr("프리셋 삭제", "Delete Preset"),
            self._tr(
                "'{name}' 프리셋을 삭제할까요? 지도에 이미 기억된 설정은 그대로 남습니다.",
                "Delete preset '{name}'? Settings already remembered with a map stay as they are.",
            ).format(name=preset.name),
        ):
            return False
        try:
            self._store_presets(remove(self._presets, preset.name))
        except PresetError as exc:
            self._show_preset_error(exc)
            return False
        self._refresh_preset_combo(select_name="")
        self._set_preset_message(("프리셋 '{name}'을 삭제했습니다.", "Deleted preset '{name}'."), name=preset.name)
        return True

    def export_preset(self):
        preset = self._selected_preset()
        if preset is None:
            return False
        path, _filter = QFileDialog.getSaveFileName(
            self, self._tr("프리셋 내보내기", "Export Preset"),
            f"{preset.name}.archaeotrace-preset.json", "JSON (*.json)",
        )
        if not path:
            return False
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(preset.to_json())
        except OSError as exc:
            QMessageBox.warning(self, self._tr("프리셋 내보내기", "Export Preset"), str(exc))
            return False
        self._set_preset_message(
            ("프리셋 '{name}'을 파일로 내보냈습니다.", "Exported preset '{name}' to a file."), name=preset.name,
        )
        return True

    def import_preset(self):
        if self.active_tool is not None:
            return False
        path, _filter = QFileDialog.getOpenFileName(
            self, self._tr("프리셋 가져오기", "Import Preset"), "", "JSON (*.json)",
        )
        if not path:
            return False
        try:
            with open(path, "rb") as handle:
                data = handle.read(MAX_PRESET_FILE_BYTES + 1)
            if len(data) > MAX_PRESET_FILE_BYTES:
                raise PresetError("malformed", "the file is larger than a preset can be")
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PresetError("malformed", "the file is not UTF-8 text") from exc
            preset = preset_from_json(text)
            if find(self._presets, preset.name) is not None and not self._confirm(
                self._tr("프리셋 가져오기", "Import Preset"),
                self._tr(
                    "같은 이름의 프리셋 '{name}'이 있습니다. 가져온 설정으로 덮어쓸까요?",
                    "A preset named '{name}' exists. Replace it with the imported settings?",
                ).format(name=preset.name),
            ):
                return False
            self._store_presets(upsert(self._presets, preset))
        except OSError as exc:
            QMessageBox.warning(self, self._tr("프리셋 가져오기", "Import Preset"), str(exc))
            return False
        except PresetError as exc:
            self._show_preset_error(exc)
            return False
        self._refresh_preset_combo(select_name=find(self._presets, preset.name).name)
        self._set_preset_message((
            "프리셋 '{name}'을 가져왔습니다. 쓰려면 '적용'을 누르세요.",
            "Imported preset '{name}'. Press Apply to use it.",
        ), name=preset.name)
        return True

    # -- project memory --------------------------------------------------

    @staticmethod
    def _project_preset_key(layer):
        return PROJECT_KEY_PREFIX + str(layer.id()).encode("utf-8").hex()

    def _remember_project_preset(self, layer, name=None):
        """Store the current settings with this raster in the QGIS project."""

        if layer is None:
            return False
        try:
            layer_id = layer.id()
        except RuntimeError:
            return False
        if not layer_id:
            return False
        selected = self._selected_preset()
        if name is None:
            name = selected.name if selected is not None else self._tr("이 지도의 설정", "Settings for this map")
        try:
            preset = self._preset_from_ui(name)
            if selected is not None and name != selected.name and selected.same_settings(preset):
                preset = preset.renamed(selected.name)
        except PresetError:
            return False
        project = QgsProject.instance()
        key = self._project_preset_key(layer)
        previous, _found = project.readEntry(PROJECT_SCOPE, key, "")
        text = preset.to_json()
        if previous == text:
            return True
        project.writeEntry(PROJECT_SCOPE, key, text)
        return True

    def _restore_project_preset(self, layer):
        """Apply the settings remembered with this raster, if any."""

        if layer is None or self.active_tool is not None:
            return False
        try:
            text, found = QgsProject.instance().readEntry(PROJECT_SCOPE, self._project_preset_key(layer), "")
        except RuntimeError:
            return False
        if not found or not text:
            return False
        try:
            preset = preset_from_json(text)
        except PresetError as exc:
            self._set_preset_message((
                "이 지도에 저장된 설정을 읽지 못했습니다 ({error}). 현재 설정을 유지합니다.",
                "The settings saved with this map could not be read ({error}). Current settings were kept.",
            ), "warning", error=self._preset_error_text(exc))
            return False
        skipped = self._apply_preset_settings(preset)
        if find(self._presets, preset.name) is not None:
            self._refresh_preset_combo(select_name=find(self._presets, preset.name).name)
        self._report_applied(preset, skipped, "project")
        return True

    def _on_preset_raster_changed(self, layer):
        layer_id = None
        if layer is not None:
            try:
                layer_id = layer.id()
            except RuntimeError:
                layer_id = None
        if layer_id == self._preset_raster_id:
            return
        self._preset_raster_id = layer_id
        if layer_id is not None:
            self._restore_project_preset(layer)

    def _on_preset_output_changed(self, layer):
        layer_id = None
        if layer is not None:
            try:
                layer_id = layer.id()
            except RuntimeError:
                layer_id = None
        if layer_id != self._preset_output_id:
            # Elevation suggestions follow the contours of one output layer.
            self._preset_output_id = layer_id
            del self._elevation_history[:]

    # -- ink colour choice -------------------------------------------------

    def _on_ink_color_choice(self, rgb):
        self._series_ink_rgb = tuple(int(v) for v in rgb) if rgb is not None else None

    def _series_ink_text(self):
        if self._series_ink_rgb is None:
            return None
        return self._tr(
            "선 색상: RGB {rgb} (다음 추적에 적용)",
            "Ink color: RGB {rgb} (used when tracing starts)",
        ).format(rgb=", ".join(str(value) for value in self._series_ink_rgb))


__all__ = ["TracePresetMixin", "METHOD_BY_MODEL", "MODEL_BY_METHOD"]
