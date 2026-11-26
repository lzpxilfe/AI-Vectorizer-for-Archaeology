"""Pure-Python contract for named tracing presets (archaeotrace-trace-preset/1)."""

import json
import unittest

from ai_vectorizer.core.trace_preset import (
    LIBRARY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SKIP_HED_MODEL,
    SKIP_LEARNED_MODEL,
    SKIP_OPENCV,
    SKIP_RECOVERY_RUNTIME,
    SKIP_SAM_BACKEND,
    PresetError,
    TracePreset,
    find,
    library_from_json,
    library_to_json,
    plan_application,
    preset_from_dict,
    preset_from_json,
    remove,
    rename,
    suggest_contour_elevation,
    upsert,
)


def _full():
    return TracePreset(
        name="AMS Korea L751",
        method="ink",
        assist_strength=45,
        smart_recovery=True,
        freehand=False,
        auto_path=False,
        learned_guide=False,
        ink_color_rgb=(150, 90, 50),
        contour_interval=20.0,
    )


class PresetFormatTests(unittest.TestCase):
    def test_round_trip_keeps_every_field_including_ink_colour(self):
        preset = _full()
        loaded = preset_from_json(preset.to_json())
        self.assertEqual(loaded, preset)
        self.assertEqual(loaded.ink_color_rgb, (150, 90, 50))
        self.assertEqual(loaded.contour_interval, 20.0)
        payload = json.loads(preset.to_json())
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(payload["ink_color_rgb"], [150, 90, 50])

    def test_defaults_match_the_dock_defaults(self):
        preset = TracePreset(name="x")
        self.assertEqual(preset.method, "ink")
        self.assertEqual(preset.assist_strength, 30)
        self.assertIsNone(preset.ink_color_rgb)
        self.assertIsNone(preset.contour_interval)

    def test_names_are_trimmed_and_validated(self):
        self.assertEqual(TracePreset(name="  USGS   HTMC ").name, "USGS HTMC")
        for bad in ("", "   ", "a" * 81, "bell\x07here", None, 3):
            with self.subTest(bad=bad), self.assertRaises(PresetError) as caught:
                TracePreset(name=bad)
            self.assertEqual(caught.exception.code, "invalid_name")

    def test_newer_schema_version_is_refused_by_name(self):
        payload = _full().to_dict()
        payload["schema_version"] = "archaeotrace-trace-preset/2"
        with self.assertRaises(PresetError) as caught:
            preset_from_dict(payload)
        self.assertEqual(caught.exception.code, "newer_version")

    def test_unknown_schema_unknown_field_and_missing_field_are_refused(self):
        cases = []
        payload = _full().to_dict()
        payload["schema_version"] = "something-else/1"
        cases.append((payload, "unknown_schema"))
        payload = _full().to_dict()
        payload["colour"] = "brown"
        cases.append((payload, "unknown_field"))
        payload = _full().to_dict()
        del payload["method"]
        cases.append((payload, "missing_field"))
        cases.append(([], "malformed"))
        for payload, code in cases:
            with self.subTest(code=code), self.assertRaises(PresetError) as caught:
                preset_from_dict(payload)
            self.assertEqual(caught.exception.code, code)

    def test_malformed_json_is_refused(self):
        with self.assertRaises(PresetError) as caught:
            preset_from_json("{not json")
        self.assertEqual(caught.exception.code, "malformed")

    def test_invalid_values_are_refused_not_coerced(self):
        base = _full().to_dict()
        bad_values = {
            "method": "canny",
            "assist_strength": 101,
            "freehand": 1,
            "ink_color_rgb": [128, 128, 128],
            "contour_interval": 0,
        }
        for field, value in bad_values.items():
            payload = dict(base, **{field: value})
            if field == "freehand":
                payload["smart_recovery"] = False
            with self.subTest(field=field), self.assertRaises(PresetError) as caught:
                preset_from_dict(payload)
            self.assertEqual(caught.exception.code, "invalid_field")
        payload = dict(base, assist_strength=True)
        with self.assertRaises(PresetError):
            preset_from_dict(payload)
        payload = dict(base, contour_interval=float("inf"))
        with self.assertRaises(PresetError):
            preset_from_dict(payload)

    def test_combinations_the_dock_never_produces_are_refused(self):
        with self.assertRaises(PresetError) as caught:
            TracePreset(name="x", freehand=True, smart_recovery=True)
        self.assertEqual(caught.exception.code, "conflicting_fields")
        with self.assertRaises(PresetError):
            TracePreset(name="x", method="lsd", learned_guide=True)
        with self.assertRaises(PresetError):
            TracePreset(name="x", freehand=True, auto_path=True)


class PresetLibraryTests(unittest.TestCase):
    def test_library_round_trip_and_empty_values(self):
        presets = [_full(), TracePreset(name="USGS 1890", method="legacy_canny")]
        text = library_to_json(presets)
        self.assertEqual(json.loads(text)["schema_version"], LIBRARY_SCHEMA_VERSION)
        self.assertEqual(library_from_json(text), presets)
        self.assertEqual(library_from_json(None), [])
        self.assertEqual(library_from_json("  "), [])

    def test_names_are_unique_ignoring_case(self):
        with self.assertRaises(PresetError) as caught:
            library_to_json([TracePreset(name="A"), TracePreset(name="a")])
        self.assertEqual(caught.exception.code, "duplicate_name")

    def test_upsert_replaces_same_name_and_appends_new(self):
        first = TracePreset(name="Series", assist_strength=10)
        presets = upsert([], first)
        presets = upsert(presets, TracePreset(name="series", assist_strength=60))
        self.assertEqual(len(presets), 1)
        self.assertEqual(presets[0].assist_strength, 60)
        presets = upsert(presets, TracePreset(name="Other"))
        self.assertEqual([p.name for p in presets], ["series", "Other"])

    def test_rename_remove_and_find(self):
        presets = [TracePreset(name="A"), TracePreset(name="B")]
        renamed = rename(presets, "a", "C")
        self.assertEqual([p.name for p in renamed], ["C", "B"])
        with self.assertRaises(PresetError) as caught:
            rename(renamed, "C", "b")
        self.assertEqual(caught.exception.code, "duplicate_name")
        self.assertEqual(rename(renamed, "C", "c")[0].name, "c")
        self.assertEqual([p.name for p in remove(renamed, "b")], ["C"])
        with self.assertRaises(PresetError):
            remove(renamed, "missing")
        self.assertIsNone(find(renamed, "A"))
        self.assertEqual(find(renamed, " c ").name, "C")

    def test_same_settings_ignores_only_the_name(self):
        self.assertTrue(_full().same_settings(_full().renamed("Other")))
        self.assertFalse(_full().same_settings(TracePreset(name=_full().name)))


class PresetApplicationTests(unittest.TestCase):
    ALL = {
        "opencv": True, "hed_model": True, "mobile_sam_backend": True,
        "sam_backend": True, "recovery_runtime": True, "learned_model": True,
    }

    def test_everything_is_applied_when_available(self):
        plan = plan_application(_full(), self.ALL)
        self.assertEqual(plan.settings, _full())
        self.assertEqual(plan.skipped, ())

    def test_unavailable_options_are_left_off_and_reported(self):
        preset = TracePreset(name="x", method="ink", smart_recovery=True, learned_guide=True,
                             ink_color_rgb=(150, 90, 50), contour_interval=10)
        plan = plan_application(preset, {})
        self.assertFalse(plan.settings.smart_recovery)
        self.assertFalse(plan.settings.learned_guide)
        self.assertEqual(plan.settings.ink_color_rgb, (150, 90, 50))
        self.assertEqual(plan.settings.contour_interval, 10.0)
        self.assertEqual(plan.skipped, (
            ("smart_recovery", SKIP_RECOVERY_RUNTIME),
            ("learned_guide", SKIP_LEARNED_MODEL),
        ))

    def test_methods_that_could_not_start_fall_back_to_ink(self):
        plan = plan_application(TracePreset(name="x", method="lsd"), {})
        self.assertEqual(plan.settings.method, "ink")
        self.assertEqual(plan.skipped, (("method", SKIP_OPENCV),))
        plan = plan_application(TracePreset(name="x", method="hed"), {"opencv": True})
        self.assertEqual(plan.skipped, (("method", SKIP_HED_MODEL),))
        plan = plan_application(TracePreset(name="x", method="sam", auto_path=True), {})
        self.assertEqual(plan.settings.method, "sam")
        self.assertFalse(plan.settings.auto_path)
        self.assertEqual(plan.skipped, (("auto_path", SKIP_SAM_BACKEND),))
        plan = plan_application(TracePreset(name="x", method="sam"), {})
        self.assertEqual(plan.skipped, ())


class ContourElevationSuggestionTests(unittest.TestCase):
    def test_no_suggestion_without_interval_or_history(self):
        self.assertIsNone(suggest_contour_elevation([], 20))
        self.assertIsNone(suggest_contour_elevation([100.0], None))
        self.assertIsNone(suggest_contour_elevation([100.0], 0))
        self.assertIsNone(suggest_contour_elevation([float("nan")], 20))

    def test_steps_up_first_then_keeps_the_last_direction(self):
        self.assertEqual(suggest_contour_elevation([100.0], 20), 120.0)
        self.assertEqual(suggest_contour_elevation([100.0, 120.0], 20), 140.0)
        self.assertEqual(suggest_contour_elevation([120.0, 100.0], 20), 80.0)
        self.assertEqual(suggest_contour_elevation([100.0, 100.0], 20), 120.0)


if __name__ == "__main__":
    unittest.main()
