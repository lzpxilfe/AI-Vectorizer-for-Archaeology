import unittest

from ai_vectorizer.core.interaction_policy import (
    MODE_AUTO_PATH,
    MODE_FREEHAND,
    MODE_MOUSE_ASSIST,
    resolve_interaction_mode,
    uses_global_path_search,
)


class InteractionPolicyTests(unittest.TestCase):
    def test_default_is_mouse_led_assist(self):
        self.assertEqual(resolve_interaction_mode(), MODE_MOUSE_ASSIST)
        self.assertFalse(uses_global_path_search(MODE_MOUSE_ASSIST))

    def test_auto_path_requires_explicit_opt_in(self):
        self.assertEqual(resolve_interaction_mode(auto_path=True), MODE_AUTO_PATH)
        self.assertTrue(uses_global_path_search(MODE_AUTO_PATH))

    def test_freehand_and_manual_override_win_over_auto_path(self):
        self.assertEqual(
            resolve_interaction_mode(freehand=True, auto_path=True),
            MODE_FREEHAND,
        )
        self.assertEqual(
            resolve_interaction_mode(auto_path=True, manual_override=True),
            MODE_FREEHAND,
        )


if __name__ == "__main__":
    unittest.main()
