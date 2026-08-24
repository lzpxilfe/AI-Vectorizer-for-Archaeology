from __future__ import annotations

import configparser
import io
import json
import os
import re
import subprocess
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote
from unittest import mock

import litmus_sam_status as litmus
import package_plugin
from scripts import package_release


ROOT = Path(__file__).resolve().parents[1]
TEST_VERSION = "9.8.7"


class DependencyDeclarationTests(unittest.TestCase):
    def test_requirement_include_graph_is_complete_and_acyclic(self):
        requirement_files = sorted(ROOT.glob("requirements*.txt"))
        self.assertGreaterEqual(len(requirement_files), 6)
        includes = {path.resolve(): [] for path in requirement_files}
        for path in requirement_files:
            for line in path.read_text(encoding="utf-8").splitlines():
                fields = line.strip().split(maxsplit=1)
                if not fields or fields[0] not in ("-r", "--requirement"):
                    continue
                self.assertEqual(len(fields), 2, f"Malformed include in {path}")
                target = (path.parent / fields[1]).resolve()
                self.assertIn(target, includes, f"Missing requirement include: {target}")
                includes[path.resolve()].append(target)

        visiting = set()
        visited = set()

        def visit(path):
            if path in visiting:
                self.fail(f"Cyclic requirement include at {path.name}")
            if path in visited:
                return
            visiting.add(path)
            for target in includes[path]:
                visit(target)
            visiting.remove(path)
            visited.add(path)

        for path in includes:
            visit(path)

    def test_optional_sam_backends_are_split_and_commit_pinned(self):
        base = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        mobile = (ROOT / "requirements-sam-mobile.txt").read_text(encoding="utf-8")
        full = (ROOT / "requirements-sam-full.txt").read_text(encoding="utf-8")
        config = (ROOT / "ai_vectorizer" / "config.py").read_text(encoding="utf-8")
        dependencies = (
            ROOT / "ai_vectorizer" / "core" / "dependencies.py"
        ).read_text(encoding="utf-8")
        dialog = (
            ROOT / "ai_vectorizer" / "ui" / "main_dialog.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("mobile-sam>=", base)
        self.assertNotIn("torch", base)
        self.assertIn("mobile-sam @ git+https://", mobile)
        self.assertIn(
            "@f706ad9c4eb7f219c00d9050e46328518ffb65d2",
            mobile,
        )
        self.assertIn("segment-anything @ git+https://", full)
        self.assertIn(
            "@dca509fe793f601edb92606367a655c15ac00fdf",
            full,
        )
        self.assertIn('"opencv-python-headless>=4.8,<4.12"', config)
        self.assertIn('"opencv-python-headless>=4.8,<4.12"', dependencies)
        self.assertIn("OpenCV 4.8–4.11", dialog)
        self.assertIn("requests torch torchvision", config)
        opencv = (ROOT / "requirements-opencv.txt").read_text(encoding="utf-8")
        self.assertIn("opencv-python-headless>=4.8,<4.12", opencv)
        self.assertIn("does not pin QGIS' own NumPy ABI", opencv)
        self.assertIn("Pillow>=12.3.0", base)
        development = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
        self.assertIn("pytest>=9.0.3,<10", development)
        self.assertIn("Development dependencies require Python 3.10+", development)
        for relative_path in ("README.md", "ai_vectorizer/README.md"):
            document = (ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn("opencv-python-headless>=4.8,<4.12", document)
            self.assertIn("OpenCV 4.8–4.11", document)
        self.assertIn("f706ad9c4eb7f219c00d9050e46328518ffb65d2", config)
        self.assertIn("dca509fe793f601edb92606367a655c15ac00fdf", config)
        self.assertNotIn("MobileSAM.git\"", config)
        self.assertNotIn("segment-anything.git\"", config)

    def test_release_metadata_and_citation_are_consistent(self):
        metadata = (ROOT / "ai_vectorizer" / "metadata.txt").read_text(
            encoding="utf-8"
        )
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn("qgisMinimumVersion=3.22", metadata)
        self.assertIn("qgisMaximumVersion=4.99", metadata)
        email_match = re.search(r"(?m)^email=([^\r\n]+)$", metadata)
        self.assertIsNotNone(email_match)
        email = email_match.group(1)
        self.assertEqual(email, "lzpxilfe@gmail.com")
        self.assertRegex(email, r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
        self.assertNotIn("your-email@example.com", metadata)
        parser = configparser.ConfigParser()
        parser.optionxform = str
        parser.read_string(metadata)
        parsed_metadata = dict(parser.items("general"))
        version = parsed_metadata["version"]
        self.assertRegex(version, r"^[0-9]+\.[0-9]+\.[0-9]+$")
        self.assertIn("0-100% cursor/path", parsed_metadata["about"])
        self.assertIn("External dependencies:", parsed_metadata["about"])
        self.assertIn("pip stacks require Python 3.10", parsed_metadata["about"])
        self.assertEqual(parsed_metadata["changelog"].splitlines()[0], version)
        citation_version = re.search(
            r"(?m)^version:\s*([0-9]+\.[0-9]+\.[0-9]+)\s*$",
            citation,
        )
        self.assertIsNotNone(citation_version)
        self.assertEqual(citation_version.group(1), version)
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn(f"## {version} — release candidate", changelog)
        if f"## {version} — release candidate" in changelog:
            self.assertNotIn("date-released:", citation)

        for relative_path in (
            "README.md",
            "README.en.md",
            "ai_vectorizer/README.md",
            "ROADMAP.md",
            "docs/FEATURES_AND_ARCHITECTURE.md",
            "docs/OPEN_SOURCE_DEVELOPMENT_PLAN.md",
        ):
            document = (ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn(version, document)

        release_record = ROOT / "docs" / f"RELEASE_READINESS_{version}.md"
        self.assertTrue(release_record.is_file())
        self.assertFalse((ROOT / "docs" / "COMPETITIVE_EXECUTION_PLAN.md").exists())

    def test_public_docs_have_resolvable_local_links_and_no_commercial_scorecard(self):
        documents = (
            ROOT / "README.md",
            ROOT / "README.en.md",
            ROOT / "ROADMAP.md",
            ROOT / "CHANGELOG.md",
            ROOT / "CONTRIBUTING.md",
            ROOT / "SECURITY.md",
            ROOT / "ai_vectorizer" / "README.md",
            ROOT / "benchmarks" / "README.md",
        ) + tuple(sorted((ROOT / "docs").glob("*.md")))
        commercial_terms = ("bunting labs", "/pricing", "가격", "adoption gate")
        for document_path in documents:
            document = document_path.read_text(encoding="utf-8")
            lowered = document.lower()
            for term in commercial_terms:
                self.assertNotIn(term, lowered, f"{term!r} remains in {document_path}")
            for raw_target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", document):
                target = raw_target.strip().strip("<>").split("#", 1)[0]
                if not target or target.startswith(("http://", "https://", "mailto:")):
                    continue
                resolved = (document_path.parent / unquote(target)).resolve()
                self.assertTrue(
                    resolved.exists(),
                    f"Broken local Markdown link in {document_path}: {raw_target}",
                )

    def test_git_attributes_preserve_cross_platform_release_bytes(self):
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("* text=auto eol=lf", attributes)
        self.assertIn("*.bat text eol=crlf", attributes)
        self.assertIn("*.caffemodel binary", attributes)
        self.assertIn("*.png binary", attributes)
        self.assertIn(
            "ai_vectorizer/core/models/hed_deploy.prototxt binary",
            attributes,
        )
        ignores = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".ruff_cache/", ignores)
        self.assertIn("!ai_vectorizer/**/*.png", ignores)


class ReleasePackagingTests(unittest.TestCase):
    def test_plugin_icon_stays_under_qgis_resource_limit(self):
        icon = ROOT / "ai_vectorizer" / "icon.png"
        self.assertTrue(icon.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertLess(icon.stat().st_size, 1_000_000)

    def test_package_limit_matches_official_qgis_decimal_guideline(self):
        self.assertEqual(package_release.MAX_UPLOAD_BYTES, 20_000_000)

    def test_zip_is_independent_of_source_mtime_and_has_normalized_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin_dir = root / "ai_vectorizer"
            plugin_dir.mkdir()
            (plugin_dir / "metadata.txt").write_text(
                f"[general]\nversion={TEST_VERSION}\n",
                encoding="utf-8",
            )
            config_path = plugin_dir / "config.py"
            config_path.write_text("VALUE = 1\n", encoding="utf-8")

            replacements = {
                "ROOT": root,
                "PLUGIN_DIR": plugin_dir,
                "DIST_DIR": root / "dist",
                "TOP_LEVEL_ITEMS": ("metadata.txt", "config.py"),
            }
            with mock.patch.multiple(package_release, **replacements), mock.patch.dict(
                os.environ,
                {"SOURCE_DATE_EPOCH": "315532800"},
            ):
                os.utime(config_path, (1_600_000_000, 1_600_000_000))
                first_path = package_release.build_release_zip(TEST_VERSION)
                first = first_path.read_bytes()
                os.utime(config_path, (1_700_000_000, 1_700_000_000))
                second_path = package_release.build_release_zip(TEST_VERSION)
                second = second_path.read_bytes()

                self.assertEqual(first, second)
                package_release.build_release_tree(TEST_VERSION)
                self.assertEqual(package_release.run_check(TEST_VERSION), 0)
                with zipfile.ZipFile(io.BytesIO(second), "r") as archive:
                    self.assertEqual(archive.namelist(), sorted(archive.namelist()))
                    for info in archive.infolist():
                        self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0))
                        self.assertEqual(
                            info.compress_type,
                            package_release.ZIP_COMPRESSION,
                        )
                        self.assertEqual(
                            (info.external_attr >> 16) & 0xFFFF,
                            package_release.ZIP_FILE_MODE,
                        )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_source_file_and_directory_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin_dir = root / "ai_vectorizer"
            plugin_dir.mkdir()
            (plugin_dir / "metadata.txt").write_text(
                f"[general]\nversion={TEST_VERSION}\n",
                encoding="utf-8",
            )
            external_file = root / "outside.py"
            external_file.write_text("SECRET = True\n", encoding="utf-8")
            try:
                (plugin_dir / "config.py").symlink_to(external_file)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")

            with mock.patch.multiple(
                package_release,
                PLUGIN_DIR=plugin_dir,
                TOP_LEVEL_ITEMS=("metadata.txt", "config.py"),
            ):
                with self.assertRaisesRegex(ValueError, "symlinks are not allowed"):
                    package_release.iter_source_files()

            (plugin_dir / "config.py").unlink()
            external_directory = root / "outside-core"
            external_directory.mkdir()
            (external_directory / "secret.py").write_text(
                "SECRET = True\n",
                encoding="utf-8",
            )
            try:
                (plugin_dir / "core").symlink_to(
                    external_directory,
                    target_is_directory=True,
                )
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"directory symlink creation is unavailable: {exc}")
            with mock.patch.multiple(
                package_release,
                PLUGIN_DIR=plugin_dir,
                TOP_LEVEL_ITEMS=("metadata.txt", "core"),
            ):
                with self.assertRaisesRegex(ValueError, "symlinks are not allowed"):
                    package_release.iter_source_files()

            linked_plugin_dir = root / "linked-plugin"
            try:
                linked_plugin_dir.symlink_to(plugin_dir, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"directory symlink creation is unavailable: {exc}")
            with mock.patch.object(
                package_release,
                "PLUGIN_DIR",
                linked_plugin_dir,
            ):
                with self.assertRaisesRegex(ValueError, "symlinks are not allowed"):
                    package_release.iter_source_files()

    def test_windows_reparse_attribute_is_treated_as_link_like(self):
        fake_stat = SimpleNamespace(
            st_file_attributes=getattr(
                package_release.stat,
                "FILE_ATTRIBUTE_REPARSE_POINT",
                0x400,
            )
        )
        path = Path("junction")
        with mock.patch.object(Path, "is_symlink", return_value=False), mock.patch.object(
            Path,
            "lstat",
            return_value=fake_stat,
        ):
            self.assertTrue(package_release.is_link_like(path))

    @unittest.skipUnless(os.name == "nt", "Windows junction regression")
    def test_windows_directory_junction_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin_dir = root / "ai_vectorizer"
            plugin_dir.mkdir()
            (plugin_dir / "metadata.txt").write_text(
                f"[general]\nversion={TEST_VERSION}\n",
                encoding="utf-8",
            )
            external = root / "outside-core"
            external.mkdir()
            (external / "secret.py").write_text("SECRET = True\n", encoding="utf-8")
            junction = plugin_dir / "core"
            created = subprocess.run(
                f'mklink /J "{junction}" "{external}"',
                shell=True,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(created.returncode, 0, created.stderr or created.stdout)
            try:
                with mock.patch.multiple(
                    package_release,
                    PLUGIN_DIR=plugin_dir,
                    TOP_LEVEL_ITEMS=("metadata.txt", "core"),
                ):
                    with self.assertRaisesRegex(ValueError, "junctions/reparse points"):
                        package_release.iter_source_files()
            finally:
                os.rmdir(junction)

    def test_hed_download_and_rollback_residue_is_not_packaged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin_dir = root / "ai_vectorizer"
            models_dir = plugin_dir / "core" / "models"
            models_dir.mkdir(parents=True)
            canonical = models_dir / "hed_deploy.prototxt"
            canonical.write_text("name: pinned\n", encoding="utf-8")
            residue_names = (
                "hed_prototxt_deadbeef.prototxt",
                "hed_weights_deadbeef.caffemodel",
                ".hed_deploy.prototxt.deadbeef.rollback",
                ".hed_pretrained_bsds.caffemodel.deadbeef.rollback",
                "mobile_sam.pt.deadbeef.download",
                "mobile_sam.pt.deadbeef.migration",
                "mobile_sam.meta.json",
                "sam_vit_b_01ec64.meta.json",
            )
            for name in residue_names:
                (models_dir / name).write_bytes(b"crash residue")

            with mock.patch.multiple(
                package_release,
                PLUGIN_DIR=plugin_dir,
                TOP_LEVEL_ITEMS=("core",),
            ):
                selected = {
                    relative.as_posix()
                    for _source, relative in package_release.iter_source_files()
                }

            self.assertIn("core/models/hed_deploy.prototxt", selected)
            for name in residue_names:
                self.assertNotIn(f"core/models/{name}", selected)

    def test_hidden_files_and_native_binaries_are_rejected(self):
        for relative_path, message in (
            (Path("core") / ".secret", "Hidden files are not allowed"),
            (Path("core") / "native.so.1", "Native binaries are not allowed"),
        ):
            with self.subTest(relative_path=relative_path), tempfile.TemporaryDirectory() as temporary:
                plugin_dir = Path(temporary) / "ai_vectorizer"
                target = plugin_dir / relative_path
                target.parent.mkdir(parents=True)
                target.write_bytes(b"release residue")
                with mock.patch.multiple(
                    package_release,
                    PLUGIN_DIR=plugin_dir,
                    TOP_LEVEL_ITEMS=("core",),
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        package_release.iter_source_files()

    def test_atomic_publish_failure_preserves_existing_zip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin_dir = root / "ai_vectorizer"
            plugin_dir.mkdir()
            (plugin_dir / "metadata.txt").write_text(
                f"[general]\nversion={TEST_VERSION}\n",
                encoding="utf-8",
            )
            dist_dir = root / "dist"
            dist_dir.mkdir()
            target_zip = dist_dir / f"ai_vectorizer-{TEST_VERSION}.zip"
            original = b"existing verified release"
            target_zip.write_bytes(original)

            with mock.patch.multiple(
                package_release,
                ROOT=root,
                PLUGIN_DIR=plugin_dir,
                DIST_DIR=dist_dir,
                TOP_LEVEL_ITEMS=("metadata.txt",),
            ), mock.patch.object(
                package_release.os,
                "replace",
                side_effect=OSError("simulated publish failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated publish failure"):
                    package_release.build_release_zip(TEST_VERSION)

            self.assertEqual(target_zip.read_bytes(), original)
            self.assertEqual(list(dist_dir.glob("*.tmp")), [])

    def test_upload_limit_and_source_date_range_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin_dir = root / "ai_vectorizer"
            plugin_dir.mkdir()
            (plugin_dir / "metadata.txt").write_text(
                f"[general]\nversion={TEST_VERSION}\n",
                encoding="utf-8",
            )
            with mock.patch.multiple(
                package_release,
                ROOT=root,
                PLUGIN_DIR=plugin_dir,
                DIST_DIR=root / "dist",
                TOP_LEVEL_ITEMS=("metadata.txt",),
                MAX_UPLOAD_BYTES=1,
            ):
                with self.assertRaisesRegex(ValueError, "upload limit"):
                    package_release.build_release_zip(TEST_VERSION)

        invalid_epochs = (
            (
                package_release.DEFAULT_SOURCE_DATE_EPOCH - 1,
                "predates the ZIP timestamp range",
            ),
            (
                package_release.MAX_SOURCE_DATE_EPOCH + 1,
                "exceeds the ZIP timestamp range",
            ),
        )
        for value, message in invalid_epochs:
            with self.subTest(value=value), mock.patch.dict(
                os.environ,
                {"SOURCE_DATE_EPOCH": str(value)},
            ):
                with self.assertRaisesRegex(ValueError, message):
                    package_release.source_date_epoch()

    def test_metadata_version_cannot_escape_release_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            plugin_dir = Path(temporary) / "ai_vectorizer"
            plugin_dir.mkdir()
            (plugin_dir / "metadata.txt").write_text(
                "[general]\nversion=../../outside\n",
                encoding="utf-8",
            )
            with mock.patch.object(package_release, "PLUGIN_DIR", plugin_dir):
                with self.assertRaisesRegex(ValueError, "Invalid plugin metadata version"):
                    package_release.load_version()

    def test_legacy_entry_point_delegates_arguments(self):
        with mock.patch.object(package_plugin, "release_main", return_value=7) as delegated:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = package_plugin.create_zip(["--check"])
        self.assertEqual(result, 7)
        delegated.assert_called_once_with(["--check"])
        self.assertIn("deprecated", stderr.getvalue())


class LitmusTests(unittest.TestCase):
    def test_current_engine_api_models_dir_and_single_remote_check(self):
        class FakeEngine:
            check_calls = 0
            DOWNLOAD_TIMEOUT_SECONDS = 60

            def __init__(self, *, backend, models_dir):
                self.backend = backend
                self.models_dir = Path(models_dir)
                self.weights_path = self.models_dir / "mobile_sam.pt"
                self.weights_meta_path = self.models_dir / "mobile_sam.meta.json"
                self.model_spec = {"weights_url": "https://example.invalid/model.pt"}

            @classmethod
            def is_backend_available(cls, backend):
                return backend == "mobile_sam"

            def get_local_weights_info(self):
                return {"exists": False}

            def get_remote_weights_info(self):
                raise AssertionError("litmus must not perform a second remote request")

            def check_weights_update(self):
                type(self).check_calls += 1
                return {
                    "ok": True,
                    "status": "not_installed",
                    "local": {"exists": False},
                    "remote": {"etag": "test"},
                }

        fake_module = SimpleNamespace(SAMEngine=FakeEngine, __file__="fake_sam_engine.py")
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            litmus.importlib,
            "import_module",
            return_value=fake_module,
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = litmus.main(["--models-dir", temporary])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(result, litmus.EXIT_OK)
        self.assertEqual(FakeEngine.check_calls, 1)
        self.assertIn("segment_anything", payload["modules"])
        self.assertNotIn("sam3", payload["modules"])
        self.assertEqual(payload["sam_engine"]["models_dir_source"], "command_line")
        self.assertEqual(
            payload["sam_engine"]["models_dir"],
            str(Path(temporary).resolve()),
        )
        self.assertEqual(
            payload["sam_engine"]["weights_url"],
            "https://example.invalid/model.pt",
        )

    def test_stale_engine_api_returns_nonzero(self):
        fake_module = SimpleNamespace(SAMEngine=type("StaleEngine", (), {}), __file__="old.py")
        with mock.patch.object(
            litmus.importlib,
            "import_module",
            return_value=fake_module,
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = litmus.main([])

        self.assertEqual(result, litmus.EXIT_DIAGNOSTIC_ERROR)
        self.assertTrue(json.loads(stdout.getvalue())["sam_engine"]["missing_methods"])


@unittest.skipIf(os.name == "nt", "shell-link contract is POSIX-only")
class DevLinkTests(unittest.TestCase):
    def test_macos_link_script_rejects_a_non_plugin_source(self):
        script = ROOT / "scripts" / "setup_dev_link.sh"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "not-a-plugin"
            source.mkdir()
            plugins = root / "plugins"
            environment = {
                **os.environ,
                "ARCHAEOTRACE_PLUGIN_SOURCE": str(source),
                "ARCHAEOTRACE_QGIS_PLUGINS_DIR": str(plugins),
            }

            refused = subprocess.run(
                ["bash", str(script)],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("missing metadata.txt", refused.stderr)
            self.assertFalse((plugins / "ai_vectorizer").exists())

    def test_macos_link_script_refuses_real_directory_then_links_clean_target(self):
        script = ROOT / "scripts" / "setup_dev_link.sh"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "metadata.txt").write_text("[general]\n", encoding="utf-8")
            plugins = root / "plugins"
            destination = plugins / "ai_vectorizer"
            destination.mkdir(parents=True)
            environment = {
                **os.environ,
                "ARCHAEOTRACE_PLUGIN_SOURCE": str(source),
                "ARCHAEOTRACE_QGIS_PLUGINS_DIR": str(plugins),
            }

            refused = subprocess.run(
                ["bash", str(script)],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertFalse((destination / source.name).exists())

            destination.rmdir()
            linked = subprocess.run(
                ["bash", str(script)],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(linked.returncode, 0, linked.stderr)
            self.assertTrue(destination.is_symlink())
            expected_source = str(source.resolve())
            self.assertEqual(os.readlink(destination), expected_source)

            repeated = subprocess.run(
                ["bash", str(script)],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertEqual(os.readlink(destination), expected_source)
            self.assertIn("Already linked", repeated.stdout)

            destination.unlink()
            other_source = root / "other-source"
            other_source.mkdir()
            destination.symlink_to(other_source, target_is_directory=True)
            conflicting = subprocess.run(
                ["bash", str(script)],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(conflicting.returncode, 0)
            self.assertEqual(os.readlink(destination), str(other_source))

    def test_windows_script_checks_failure_not_nonnegative_errorlevel(self):
        script = (ROOT / "setup_dev_link.bat").read_text(encoding="utf-8")
        self.assertIn("if errorlevel 1", script.lower())
        self.assertNotIn("if errorlevel 0", script.lower())
        self.assertIn("exit /b 1", script.lower())


class ContinuousIntegrationTests(unittest.TestCase):
    def test_ci_covers_supported_qgis_imports(self):
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("qgis/qgis:3.22.16@sha256:", workflow)
        self.assertIn("qgis/qgis:3.44.13@sha256:", workflow)
        self.assertIn("qgis/qgis:4.2.1@sha256:", workflow)
        self.assertIn("scripts/qgis_import_smoke.py", workflow)
        self.assertIn('--plugin-root "$ARCHAEOTRACE_RELEASE_ROOT"', workflow)
        self.assertIn("tests.test_qgis_runtime_safety", workflow)
        self.assertIn('ARCHAEOTRACE_REQUIRE_QGIS: "1"', workflow)
        self.assertIn('python: ["3.10", "3.12"]', workflow)
        self.assertIn("python38-compatibility:", workflow)
        self.assertIn('python-version: "3.8"', workflow)
        self.assertIn("test_count < 150", workflow)
        self.assertIn("python -m compileall -q", workflow)
        self.assertIn("dependency-audit:", workflow)
        self.assertIn("pip-audit==2.10.1", workflow)
        self.assertIn("python -m pip_audit", workflow)
        self.assertIn("--strict", workflow)
        self.assertIn("requirements-sam-common.txt", workflow)
        self.assertIn("Verify immutable non-PyPI backend pins", workflow)
        self.assertNotIn("requirements-sam-mobile.txt", workflow)
        self.assertNotIn("requirements-sam-full.txt", workflow)
        self.assertIn("cancel-in-progress: true", workflow)
        self.assertIn("release_windows:", workflow)
        self.assertIn("needs.release_linux.outputs.zip_sha256", workflow)
        self.assertIn("qgis-import:\n    name:", workflow)
        self.assertIn("RELEASE_ZIP_SHA256:", workflow)
        self.assertIn('test "$actual_sha" = "$RELEASE_ZIP_SHA256"', workflow)
        self.assertIn("persist-credentials: false", workflow)


if __name__ == "__main__":
    unittest.main()
