from dataclasses import asdict
from types import SimpleNamespace
import unittest

import benchmarks.worker as worker

try:
    import numpy as np
    from ai_vectorizer.core import recovery_prompts
    from ai_vectorizer.core.line_evidence import LineEvidence
    from ai_vectorizer.core.smart_recovery import DEFAULT_RECOVERY_CONFIG
except ImportError:  # The dependency-light benchmark test environment is valid.
    np = None
    LineEvidence = None
    DEFAULT_RECOVERY_CONFIG = None
    recovery_prompts = None


def _ink_configuration():
    return {
        "edge_weight": 1.0,
        "livewire_window_px": 320,
        "target_snap_radius_px": 6,
        "smoothing_window_px": 5,
        "smoothing_profile": worker.INK_LIVEWIRE_SMOOTHING_PROFILE,
    }


def _recovery_configuration():
    return {
        **_ink_configuration(),
        "recovery_policy_id": DEFAULT_RECOVERY_CONFIG.policy_id,
        "recovery_provisional": True,
        "recovery_thresholds": asdict(DEFAULT_RECOVERY_CONFIG),
        "recovery_configuration_sha256": DEFAULT_RECOVERY_CONFIG.sha256,
    }


class _FakeTree:
    origin = (0, 0)
    shape = (12, 12)

    def trace(self, target):
        return [(1.0, 2.0), (2.0, 2.0), (3.0, 2.0), (4.0, 2.0), (5.0, 2.0), target]


class _FakeLiveWire:
    calls = []

    class LiveWireConfig:
        def __init__(self, **values):
            self.values = values

    @classmethod
    def build_livewire_tree(cls, image, edges, root, **kwargs):
        cls.calls.append((image, edges, root, kwargs))
        return _FakeTree()


class _FakeTraceKernel:
    @staticmethod
    def smooth_pixel_path(points, *, window_size):
        result = list(points)
        result[0] = (99.0, 99.0)
        result[-1] = (98.0, 98.0)
        return tuple(result)


@unittest.skipIf(np is None, "Ink adapter contract tests require NumPy")
class InkAdapterContractTests(unittest.TestCase):
    def _pipeline(self, backend, detector):
        pipeline = worker._InkLiveWirePipeline.__new__(worker._InkLiveWirePipeline)
        pipeline._backend = backend
        pipeline._np = np
        pipeline._detector = detector
        pipeline._livewire = _FakeLiveWire
        pipeline._trace_kernel = _FakeTraceKernel
        pipeline.info = SimpleNamespace(runtime_details={})
        return pipeline

    def setUp(self):
        _FakeLiveWire.calls.clear()
        self.image = {
            "rgb": np.zeros((12, 12, 3), dtype=np.uint8),
            "edges": None,
            "evidence": None,
            "source_tile_origin_xy": (128, 64),
        }
        self.prompt = worker.TracePrompt(
            (1.25, 2.5),
            (8.5, 2.25),
            previous_xy=(-1.75, 0.5),
        )

    def test_ink_v1_uses_only_legacy_detect_edges(self):
        class Detector:
            def detect_edges(self, image):
                return np.ones(image.shape[:2], dtype=np.uint8) * 255

            def detect_ink_evidence(self, _image):
                raise AssertionError("v1 must not request LineEvidence")

        pipeline = self._pipeline(worker.INK_LIVEWIRE_V1_BACKEND, Detector())
        path, evidence, tree = pipeline._champion_path(
            self.image, self.prompt, _ink_configuration()
        )

        self.assertIsNone(evidence)
        self.assertIsInstance(tree, _FakeTree)
        self.assertIsNone(_FakeLiveWire.calls[0][3]["evidence"])
        self.assertEqual(path[0], self.prompt.start_xy)
        self.assertEqual(path[-1], self.prompt.end_xy)

    def test_ink_v2_consumes_centerline_and_passes_exact_evidence(self):
        observed_origin = {}
        evidence = LineEvidence(
            center_score=np.ones((12, 12), dtype=np.float32),
            centerline=np.eye(12, dtype=bool),
            tangent_x=np.ones((12, 12), dtype=np.float32),
            tangent_y=np.zeros((12, 12), dtype=np.float32),
            coherence=np.ones((12, 12), dtype=np.float32),
        )

        class Detector:
            def detect_edges(self, _image):
                raise AssertionError("v2 must not disguise itself as detect_edges")

            def detect_ink_evidence(self, _image, *, tile_origin):
                observed_origin["value"] = tile_origin
                return evidence

        pipeline = self._pipeline(worker.INK_LIVEWIRE_V2_BACKEND, Detector())
        path, observed, tree = pipeline._champion_path(
            self.image, self.prompt, _ink_configuration()
        )

        _image, edges, root, kwargs = _FakeLiveWire.calls[0]
        self.assertIs(observed, evidence)
        self.assertIsInstance(tree, _FakeTree)
        self.assertIs(kwargs["evidence"], evidence)
        self.assertEqual(observed_origin["value"], (128, 64))
        self.assertTrue(np.array_equal(edges > 0, evidence.centerline))
        self.assertEqual(root, (1, 2))
        self.assertEqual(kwargs["incoming_direction"], (2, 2))
        self.assertEqual(kwargs["config"].values["max_window_size"], 320)
        self.assertEqual(kwargs["config"].values["target_snap_radius"], 6)
        self.assertEqual(path[0], self.prompt.start_xy)
        self.assertEqual(path[-1], self.prompt.end_xy)

    def test_recovery_does_not_touch_model_when_ink_gate_is_confident(self):
        centerline = np.zeros((12, 12), dtype=bool)
        centerline[2, :] = True
        evidence = LineEvidence(
            center_score=np.ones((12, 12), dtype=np.float32),
            centerline=centerline,
            tangent_x=np.ones((12, 12), dtype=np.float32),
            tangent_y=np.zeros((12, 12), dtype=np.float32),
            coherence=np.ones((12, 12), dtype=np.float32),
        )
        champion = [(1.0, 2.0), (8.0, 2.0)]
        pipeline = worker._InkV2EfficientSAMRecoveryPipeline.__new__(
            worker._InkV2EfficientSAMRecoveryPipeline
        )
        pipeline._champion_path = lambda *_args: (champion, evidence, _FakeTree())
        from ai_vectorizer.core import smart_recovery

        pipeline._smart_recovery = smart_recovery
        pipeline._latest_prediction_evidence = None
        pipeline._engine = SimpleNamespace(
            encode=lambda _image: (_ for _ in ()).throw(
                AssertionError("encoder must stay off before a low-quality gate")
            )
        )
        prompt = worker.TracePrompt((1.0, 2.0), (8.0, 2.0))

        selected = pipeline.predict(
            {"encoding": None}, prompt, _recovery_configuration()
        )
        record = worker._validated_recovery_prediction_evidence(
            pipeline.prediction_evidence()
        )

        self.assertEqual(selected, champion)
        self.assertFalse(record["gate"]["trigger"])
        self.assertEqual(record["selected_route"], "champion")
        self.assertIsNone(record["challenger_sha256"])
        self.assertIsNone(record["segmentation_evidence"])
        self.assertEqual(record["rejection_reason"], "recovery_not_triggered")

    def test_low_quality_gate_runs_and_can_select_one_challenger(self):
        evidence = LineEvidence(
            center_score=np.ones((12, 12), dtype=np.float32),
            centerline=np.ones((12, 12), dtype=bool),
            tangent_x=np.ones((12, 12), dtype=np.float32),
            tangent_y=np.zeros((12, 12), dtype=np.float32),
            coherence=np.ones((12, 12), dtype=np.float32),
        )
        quality = SimpleNamespace(
            as_dict=lambda: {
                "support_q10": 0.1,
                "mean_support": 0.2,
                "longest_unsupported_run": 10,
                "mean_coherence": 0.1,
                "detour_ratio": 1.0,
                "branch_density": 0.0,
                "endpoint_error": 0.0,
                "sample_count": 8,
            }
        )
        champion = [(1.0, 2.0), (8.0, 2.0)]
        challenger = [(1, 2), (4, 3), (8, 2)]
        calls = {"encode": 0, "predict": 0, "corridor": 0, "arbitrate": 0}
        model_input = {}

        class FakeRecovery:
            @staticmethod
            def recovery_gate(*_args, **_kwargs):
                return SimpleNamespace(
                    trigger=True,
                    reason="unsupported_gap",
                    quality=quality,
                    policy_id=DEFAULT_RECOVERY_CONFIG.policy_id,
                    configuration_sha256=DEFAULT_RECOVERY_CONFIG.sha256,
                )

            @staticmethod
            def build_corridor_cost_map(_evidence, corridor, *, config):
                calls["corridor"] += 1
                self.assertEqual(corridor.shape, (12, 12))
                self.assertEqual(config.sha256, DEFAULT_RECOVERY_CONFIG.sha256)
                return np.ones((12, 12), dtype=np.float32)

            @staticmethod
            def arbitrate_routes(*_args, **_kwargs):
                calls["arbitrate"] += 1
                return SimpleNamespace(
                    selected="challenger",
                    accepted=True,
                    reason="accepted",
                    champion_quality=quality,
                    challenger_quality=quality,
                    strong_ink_retention=1.0,
                    route_separation_p95=1.0,
                )

        class FakeTraceKernel:
            class TraceConfig:
                def __init__(self, **_kwargs):
                    pass

            @staticmethod
            def trace_path(*_args, **_kwargs):
                return SimpleNamespace(status="complete", points_xy=challenger)

        class FakeEngine:
            @staticmethod
            def encode(_rgb):
                calls["encode"] += 1
                return "encoding"

            @staticmethod
            def predict(_encoding, points, labels):
                calls["predict"] += 1
                model_input["points"] = np.asarray(points).tolist()
                model_input["labels"] = np.asarray(labels).tolist()
                return SimpleNamespace(
                    mask=np.ones((12, 12), dtype=bool),
                    selected_logits=np.ones((12, 12), dtype=np.float32),
                    iou_predictions=np.array([0.1, 0.9, 0.2], dtype=np.float32),
                    selected_index=1,
                    metadata={"timing_ms": {"decoder": 1.0}},
                )

        pipeline = worker._InkV2EfficientSAMRecoveryPipeline.__new__(
            worker._InkV2EfficientSAMRecoveryPipeline
        )
        pipeline._champion_path = lambda *_args: (champion, evidence, _FakeTree())
        pipeline._smart_recovery = FakeRecovery
        pipeline._recovery_trace_kernel = FakeTraceKernel
        pipeline._engine = FakeEngine
        pipeline._recovery_prompts = recovery_prompts
        pipeline._np = np
        pipeline._latest_prediction_evidence = None
        pipeline.info = SimpleNamespace(runtime_details={})
        prompt = worker.TracePrompt(
            (1.0, 2.0),
            (8.0, 2.0),
            positive_xy=((3.0, 5.0),),
            negative_xy=((4.0, 4.0),),
            previous_xy=(0.0, 2.0),
        )
        image = {"rgb": np.zeros((12, 12, 3), dtype=np.uint8), "encoding": None}

        selected = pipeline.predict(image, prompt, _recovery_configuration())
        record = worker._validated_recovery_prediction_evidence(
            pipeline.prediction_evidence()
        )

        self.assertEqual(selected, challenger)
        self.assertEqual(calls, {"encode": 1, "predict": 1, "corridor": 1, "arbitrate": 1})
        self.assertEqual(model_input["points"], [[1.0, 2.0], [8.0, 2.0]])
        self.assertEqual(model_input["labels"], [1, 1])
        self.assertTrue(record["gate"]["trigger"])
        self.assertEqual(record["selected_route"], "challenger")
        self.assertIsNotNone(record["challenger_sha256"])
        self.assertEqual(record["selection"]["reason"], "accepted")
        self.assertIsNotNone(record["segmentation_evidence"])

    def test_recovery_a_star_is_cropped_to_exact_champion_tree_bounds(self):
        evidence = LineEvidence(
            center_score=np.ones((12, 12), dtype=np.float32),
            centerline=np.ones((12, 12), dtype=bool),
            tangent_x=np.ones((12, 12), dtype=np.float32),
            tangent_y=np.zeros((12, 12), dtype=np.float32),
            coherence=np.ones((12, 12), dtype=np.float32),
        )
        quality = SimpleNamespace(
            as_dict=lambda: {
                "support_q10": 0.1,
                "mean_support": 0.2,
                "longest_unsupported_run": 10,
                "mean_coherence": 0.1,
                "detour_ratio": 1.0,
                "branch_density": 0.0,
                "endpoint_error": 0.0,
                "sample_count": 8,
            }
        )
        tree = SimpleNamespace(origin=(3, 4), shape=(5, 6))
        champion = [(4.0, 5.0), (7.0, 7.0)]
        observed = {}

        class FakeRecovery:
            @staticmethod
            def recovery_gate(*_args, **_kwargs):
                return SimpleNamespace(
                    trigger=True,
                    reason="unsupported_gap",
                    quality=quality,
                    policy_id=DEFAULT_RECOVERY_CONFIG.policy_id,
                    configuration_sha256=DEFAULT_RECOVERY_CONFIG.sha256,
                )

            @staticmethod
            def build_corridor_cost_map(bounded_evidence, corridor, *, config):
                observed["evidence_shape"] = bounded_evidence.shape
                observed["corridor_shape"] = corridor.shape
                return np.ones(corridor.shape, dtype=np.float32)

            @staticmethod
            def arbitrate_routes(champion_path, challenger_path, *_args, **_kwargs):
                observed["challenger"] = list(challenger_path)
                return SimpleNamespace(
                    selected="challenger",
                    accepted=True,
                    reason="accepted",
                    champion_quality=quality,
                    challenger_quality=quality,
                    strong_ink_retention=1.0,
                    route_separation_p95=1.0,
                )

        class FakeTraceKernel:
            class TraceConfig:
                def __init__(self, **values):
                    observed["trace_limits"] = values

            @staticmethod
            def trace_path(cost_map, start, end, **_kwargs):
                observed["cost_shape"] = cost_map.shape
                observed["local_start"] = start
                observed["local_end"] = end
                return SimpleNamespace(
                    status="complete",
                    points_xy=(
                        (1, 1),
                        (1, 2),
                        (2, 2),
                        (3, 2),
                        (4, 2),
                        (4, 3),
                    ),
                )

            @staticmethod
            def smooth_pixel_path(points, *, window_size):
                observed["smoothed_input"] = list(points)
                observed["smooth_window"] = window_size
                result = list(points)
                result[0] = (99.0, 99.0)
                result[2] = (5.5, 6.5)
                result[-1] = (98.0, 98.0)
                return tuple(result)

        class FakeEngine:
            @staticmethod
            def encode(_rgb):
                return "encoding"

            @staticmethod
            def predict(_encoding, _points, _labels):
                return SimpleNamespace(
                    mask=np.ones((12, 12), dtype=bool),
                    selected_logits=np.ones((12, 12), dtype=np.float32),
                    iou_predictions=np.array([0.1, 0.9, 0.2], dtype=np.float32),
                    selected_index=1,
                    metadata={"timing_ms": {"decoder": 1.0}},
                )

        pipeline = worker._InkV2EfficientSAMRecoveryPipeline.__new__(
            worker._InkV2EfficientSAMRecoveryPipeline
        )
        pipeline._champion_path = lambda *_args: (champion, evidence, tree)
        pipeline._smart_recovery = FakeRecovery
        pipeline._recovery_trace_kernel = FakeTraceKernel
        pipeline._engine = FakeEngine
        pipeline._recovery_prompts = recovery_prompts
        pipeline._np = np
        pipeline._latest_prediction_evidence = None
        pipeline.info = SimpleNamespace(runtime_details={})
        prompt = worker.TracePrompt((4.0, 5.0), (7.0, 7.0))
        image = {"rgb": np.zeros((12, 12, 3), dtype=np.uint8), "encoding": None}

        selected = pipeline.predict(image, prompt, _recovery_configuration())

        self.assertEqual(observed["evidence_shape"], (5, 6))
        self.assertEqual(observed["corridor_shape"], (5, 6))
        self.assertEqual(observed["cost_shape"], (5, 6))
        self.assertEqual(observed["local_start"], (1.0, 1.0))
        self.assertEqual(observed["local_end"], (4.0, 3.0))
        self.assertEqual(observed["trace_limits"]["max_width"], 6)
        self.assertEqual(observed["trace_limits"]["max_height"], 5)
        self.assertEqual(observed["trace_limits"]["max_cells"], 30)
        self.assertEqual(observed["smooth_window"], 5)
        self.assertEqual(observed["challenger"][0], prompt.start_xy)
        self.assertEqual(observed["challenger"][-1], prompt.end_xy)
        self.assertEqual(observed["challenger"][2], (5.5, 6.5))
        self.assertEqual(selected, observed["challenger"])

    def test_recovery_thresholds_are_explicit_and_hash_bound(self):
        configuration = _recovery_configuration()
        self.assertEqual(
            worker._recovery_contract(configuration).sha256,
            DEFAULT_RECOVERY_CONFIG.sha256,
        )
        configuration["recovery_thresholds"]["trigger_support_quantile"] = 0.5
        with self.assertRaisesRegex(worker.WorkerRequestError, "does not match"):
            worker._recovery_contract(configuration)

    def test_new_method_ids_are_registered_without_removing_efficientsam(self):
        self.assertTrue(
            {
                worker.INK_LIVEWIRE_V1_BACKEND,
                worker.INK_LIVEWIRE_V2_BACKEND,
                worker.EFFICIENTSAM_BACKEND,
                worker.INK_V2_EFFICIENTSAM_RECOVERY_BACKEND,
            }.issubset(worker.SUPPORTED_BACKENDS)
        )


if __name__ == "__main__":
    unittest.main()
