from __future__ import annotations

from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

from scripts import qgis_clean_profile_smoke as clean_smoke


class CleanProfileInstallerTests(unittest.TestCase):
    @staticmethod
    def _write_archive(path: Path, *, version: str = "0.1.5") -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "ai_vectorizer/metadata.txt",
                "[general]\n"
                "name=ArchaeoTrace\n"
                f"version={version}\n",
            )
            archive.writestr("ai_vectorizer/__init__.py", "# installed smoke\n")

    def test_installs_only_inside_empty_named_profile_and_disables_keychain(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "plugin.zip"
            profile_root = root / "isolated"
            self._write_archive(archive)

            plugin_dir, version = clean_smoke.install_archive(
                archive,
                profile_root,
            )

            self.assertEqual(version, "0.1.5")
            self.assertEqual(
                plugin_dir,
                (
                    profile_root
                    / "profiles"
                    / clean_smoke.PROFILE_NAME
                    / "python"
                    / "plugins"
                    / "ai_vectorizer"
                ).resolve(),
            )
            self.assertTrue((plugin_dir / "metadata.txt").is_file())
            settings = (
                profile_root
                / "profiles"
                / clean_smoke.PROFILE_NAME
                / "qgis.org"
                / "QGIS3.ini"
            ).read_text(encoding="utf-8")
            self.assertEqual(settings, "[auth]\nuse_password_helper=false\n")

    def test_refuses_nonempty_profile_without_changing_sentinel(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "plugin.zip"
            profile_root = root / "existing"
            profile_root.mkdir()
            sentinel = profile_root / "user-setting.ini"
            sentinel.write_bytes(b"do not touch")
            self._write_archive(archive)

            with self.assertRaisesRegex(clean_smoke.SmokeFailure, "non-empty"):
                clean_smoke.install_archive(archive, profile_root)

            self.assertEqual(sentinel.read_bytes(), b"do not touch")
            self.assertEqual(list(profile_root.iterdir()), [sentinel])

    def test_rejects_path_traversal_and_removes_only_partial_smoke_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "plugin.zip"
            profile_root = root / "isolated"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr(
                    "ai_vectorizer/metadata.txt",
                    "[general]\nversion=0.1.5\n",
                )
                output.writestr("ai_vectorizer/../escape.py", "bad")

            with self.assertRaisesRegex(clean_smoke.SmokeFailure, "traversal"):
                clean_smoke.install_archive(archive, profile_root)

            self.assertFalse(profile_root.exists())
            self.assertFalse((root / "escape.py").exists())

    def test_rejects_symlink_and_case_insensitive_collision(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            symlink_archive = root / "symlink.zip"
            with zipfile.ZipFile(symlink_archive, "w") as output:
                output.writestr(
                    "ai_vectorizer/metadata.txt",
                    "[general]\nversion=0.1.5\n",
                )
                member = zipfile.ZipInfo("ai_vectorizer/link")
                member.create_system = 3
                member.external_attr = (stat.S_IFLNK | 0o777) << 16
                output.writestr(member, "../../outside")
            with self.assertRaisesRegex(clean_smoke.SmokeFailure, "symbolic"):
                clean_smoke.install_archive(
                    symlink_archive,
                    root / "symlink-profile",
                )

            collision_archive = root / "collision.zip"
            with zipfile.ZipFile(collision_archive, "w") as output:
                output.writestr(
                    "ai_vectorizer/metadata.txt",
                    "[general]\nversion=0.1.5\n",
                )
                output.writestr("ai_vectorizer/Module.py", "first")
                output.writestr("ai_vectorizer/module.py", "second")
            with self.assertRaisesRegex(clean_smoke.SmokeFailure, "collision"):
                clean_smoke.install_archive(
                    collision_archive,
                    root / "collision-profile",
                )

    def test_rejects_wrong_metadata_version(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "plugin.zip"
            self._write_archive(archive, version="9.9.9")

            with self.assertRaisesRegex(
                clean_smoke.SmokeFailure,
                "Expected metadata version 0.1.5",
            ):
                clean_smoke.install_archive(archive, root / "isolated")


if __name__ == "__main__":
    unittest.main()
