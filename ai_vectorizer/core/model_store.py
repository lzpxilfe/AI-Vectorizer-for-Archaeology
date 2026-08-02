"""Verified, content-addressed storage for pinned local model artifacts.

Only :func:`fetch_bundle` may open the network.  Inspection, resolution, and
loading are deliberately offline operations so benchmark execution cannot
silently change its model inputs.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import errno
import hashlib
import math
import os
from pathlib import Path
import stat
import tempfile
import time
from typing import Callable, Optional, Tuple
import urllib.error
import urllib.request
from urllib.parse import quote, urlsplit

from .efficientsam_spec import (
    ArtifactSpec,
    EFFICIENTSAM_TI_SPLIT,
    ModelBundleSpec,
    bundle_fingerprint,
)


MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 120.0
_ALLOWED_DOWNLOAD_HOSTS = frozenset({"raw.githubusercontent.com"})
_USER_AGENT = "ArchaeoTrace-model-store/1"

STATE_READY = "ready"
STATE_MISSING = "missing"
STATE_CORRUPT = "corrupt"
STATE_UNSAFE = "unsafe"


class ModelStoreError(RuntimeError):
    """Base class for model-store failures."""


class ModelNotFoundError(ModelStoreError):
    """Raised when a required offline artifact is absent."""


class ModelIntegrityError(ModelStoreError):
    """Raised when artifact size or SHA-256 evidence does not match."""


class ModelCacheSafetyError(ModelStoreError):
    """Raised when a managed path is a symlink or another unsafe file type."""


class ModelDownloadError(ModelStoreError):
    """Raised when an explicit network fetch cannot be completed safely."""


@dataclass(frozen=True)
class ArtifactStatus:
    spec: ArtifactSpec
    path: Path
    state: str
    detail: Optional[str] = None

    @property
    def id(self) -> str:
        return self.spec.identifier

    @property
    def ready(self) -> bool:
        return self.state == STATE_READY

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "state": self.state,
            "path": str(self.path),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class BundleStatus:
    root: Path
    spec: ModelBundleSpec
    artifacts: Tuple[ArtifactStatus, ...]

    @property
    def ready(self) -> bool:
        return all(artifact.ready for artifact in self.artifacts)

    def artifact(self, identifier: str) -> ArtifactStatus:
        for artifact in self.artifacts:
            if artifact.spec.identifier == identifier:
                return artifact
        raise KeyError(identifier)

    def as_dict(self) -> dict:
        return {
            "bundle_id": self.spec.identifier,
            "ready": self.ready,
            "root": str(self.root),
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
        }


@dataclass(frozen=True)
class VerifiedArtifact:
    spec: ArtifactSpec
    path: Path


@dataclass(frozen=True)
class VerifiedBundle:
    root: Path
    spec: ModelBundleSpec
    artifacts: Tuple[VerifiedArtifact, ...]

    def artifact(self, identifier: str) -> VerifiedArtifact:
        for artifact in self.artifacts:
            if artifact.spec.identifier == identifier:
                return artifact
        raise KeyError(identifier)

    def path(self, identifier: str) -> Path:
        return self.artifact(identifier).path

    def read_bytes(self, identifier: str) -> bytes:
        # Re-open and re-verify so a path replacement after resolve_bundle()
        # cannot reach ONNX Runtime as unverified bytes.
        return read_verified_bytes(self.root, self.artifact(identifier).spec)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Turn every redirect into an HTTP error instead of following it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


def _default_transport(request: urllib.request.Request, timeout: float):
    opener = urllib.request.build_opener(_NoRedirectHandler())
    return opener.open(request, timeout=timeout)


def _normal_root(cache_root) -> Path:
    try:
        raw = os.fspath(cache_root)
    except TypeError as exc:
        raise ModelCacheSafetyError("Model cache root must be path-like.") from exc
    return Path(os.path.abspath(os.path.expanduser(raw)))


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _require_safe_directory(path: Path, label: str) -> os.stat_result:
    try:
        information = os.lstat(path)
    except FileNotFoundError as exc:
        raise ModelNotFoundError(f"{label} is missing: {path}") from exc
    except OSError as exc:
        raise ModelCacheSafetyError(f"Could not inspect {label} {path}: {exc}") from exc
    if stat.S_ISLNK(information.st_mode):
        raise ModelCacheSafetyError(f"{label} must not be a symbolic link: {path}")
    if not stat.S_ISDIR(information.st_mode):
        raise ModelCacheSafetyError(f"{label} is not a directory: {path}")
    return information


def _create_safe_directory(path: Path, label: str, *, parents: bool = False) -> os.stat_result:
    if not _lexists(path):
        try:
            path.mkdir(mode=0o700, parents=parents, exist_ok=False)
        except FileExistsError:
            pass
        except OSError as exc:
            raise ModelCacheSafetyError(f"Could not create {label} {path}: {exc}") from exc
    return _require_safe_directory(path, label)


def _artifact_path(root: Path, artifact: ArtifactSpec) -> Path:
    return root / "objects" / "sha256" / artifact.sha256[:2] / artifact.sha256


def _managed_parent(root: Path, artifact: ArtifactSpec, *, create: bool) -> Path:
    components = (
        (root, "model cache root"),
        (root / "objects", "model cache objects directory"),
        (root / "objects" / "sha256", "model cache SHA-256 directory"),
        (
            root / "objects" / "sha256" / artifact.sha256[:2],
            "model cache digest directory",
        ),
    )
    for index, (component, label) in enumerate(components):
        if create:
            _create_safe_directory(component, label, parents=index == 0)
        else:
            _require_safe_directory(component, label)
    return components[-1][0]


def _validate_bundle_contract(spec: ModelBundleSpec) -> None:
    if not isinstance(spec, ModelBundleSpec):
        raise TypeError("spec must be a ModelBundleSpec.")
    repository = urlsplit(spec.source_repository)
    if (
        repository.scheme != "https"
        or repository.hostname != "github.com"
        or repository.username is not None
        or repository.password is not None
        or repository.query
        or repository.fragment
    ):
        raise ModelDownloadError("The model source repository must be a fixed GitHub HTTPS URL.")
    try:
        repository_port = repository.port
    except ValueError as exc:
        raise ModelDownloadError("The model source repository has an invalid port.") from exc
    if repository_port not in (None, 443):
        raise ModelDownloadError("The model source repository must use the default HTTPS port.")
    repository_path = repository.path.rstrip("/")
    if not repository_path or repository_path.count("/") != 2:
        raise ModelDownloadError("The model source repository path must identify owner/repository.")

    for artifact in spec.artifacts:
        if artifact.size_bytes > MAX_ARTIFACT_BYTES:
            raise ModelDownloadError(
                f"Artifact {artifact.identifier!r} exceeds the {MAX_ARTIFACT_BYTES}-byte limit."
            )
        parsed = urlsplit(artifact.url)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ModelDownloadError(
                f"Artifact {artifact.identifier!r} has an invalid URL port."
            ) from exc
        expected_path = (
            f"{repository_path}/{spec.source_commit}/weights/"
            f"{quote(artifact.filename, safe='._-')}"
        )
        if (
            parsed.scheme != "https"
            or parsed.hostname not in _ALLOWED_DOWNLOAD_HOSTS
            or port not in (None, 443)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path != expected_path
        ):
            raise ModelDownloadError(
                f"Artifact {artifact.identifier!r} does not use its fixed upstream HTTPS URL."
            )


def _read_verified_file(path: Path, artifact: ArtifactSpec, *, collect: bool) -> Optional[bytes]:
    parent_before = _safe_parent_stat(path.parent)
    directory_descriptor = None
    use_directory_descriptor = (
        os.open in getattr(os, "supports_dir_fd", set())
        and os.stat in getattr(os, "supports_dir_fd", set())
    )
    if use_directory_descriptor:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            directory_descriptor = os.open(path.parent, directory_flags)
            if not os.path.samestat(parent_before, os.fstat(directory_descriptor)):
                raise ModelCacheSafetyError(
                    f"Model cache directory changed while opening {path}."
                )
        except ModelStoreError:
            if directory_descriptor is not None:
                os.close(directory_descriptor)
            raise
        except OSError as exc:
            if directory_descriptor is not None:
                os.close(directory_descriptor)
            raise ModelCacheSafetyError(
                f"Could not safely open the model cache directory for {path}: {exc}"
            ) from exc

    try:
        if directory_descriptor is None:
            before = os.lstat(path)
        else:
            before = os.stat(
                path.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
    except FileNotFoundError as exc:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        raise ModelNotFoundError(f"Model artifact is missing: {path}") from exc
    except OSError as exc:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        raise ModelCacheSafetyError(f"Could not inspect model artifact {path}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode):
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        raise ModelCacheSafetyError(f"Model artifact must not be a symbolic link: {path}")
    if not stat.S_ISREG(before.st_mode):
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        raise ModelCacheSafetyError(f"Model artifact is not a regular file: {path}")

    flags = os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        if directory_descriptor is None:
            descriptor = os.open(path, flags)
        else:
            descriptor = os.open(path.name, flags, dir_fd=directory_descriptor)
    except FileNotFoundError as exc:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        raise ModelNotFoundError(f"Model artifact disappeared before opening: {path}") from exc
    except OSError as exc:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        if exc.errno in {getattr(errno, "ELOOP", -1), getattr(errno, "EMLINK", -1)}:
            raise ModelCacheSafetyError(
                f"Model artifact became a symbolic link before opening: {path}"
            ) from exc
        raise ModelCacheSafetyError(f"Could not open model artifact {path}: {exc}") from exc

    try:
        opened = os.fstat(descriptor)
        if directory_descriptor is None and not os.path.samestat(
            parent_before,
            _safe_parent_stat(path.parent),
        ):
            raise ModelCacheSafetyError(
                f"Model cache directory changed while opening {path}."
            )
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(before, opened):
            raise ModelCacheSafetyError(f"Model artifact changed while it was being opened: {path}")
        if opened.st_size != artifact.size_bytes:
            raise ModelIntegrityError(
                f"Model artifact {artifact.identifier!r} has size {opened.st_size}; "
                f"expected {artifact.size_bytes}."
            )

        digest = hashlib.sha256()
        chunks = [] if collect else None
        total = 0
        while total <= artifact.size_bytes:
            requested = min(DOWNLOAD_CHUNK_BYTES, artifact.size_bytes + 1 - total)
            if requested <= 0:
                break
            chunk = os.read(descriptor, requested)
            if not chunk:
                break
            total += len(chunk)
            if total > artifact.size_bytes:
                raise ModelIntegrityError(
                    f"Model artifact {artifact.identifier!r} exceeds its fixed size."
                )
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)

        after = os.fstat(descriptor)
        if (
            not os.path.samestat(opened, after)
            or after.st_size != opened.st_size
            or getattr(after, "st_mtime_ns", None) != getattr(opened, "st_mtime_ns", None)
        ):
            raise ModelCacheSafetyError(f"Model artifact changed while it was being read: {path}")
        if total != artifact.size_bytes:
            raise ModelIntegrityError(
                f"Model artifact {artifact.identifier!r} ended at {total} bytes; "
                f"expected {artifact.size_bytes}."
            )
        actual_digest = digest.hexdigest()
        if actual_digest != artifact.sha256:
            raise ModelIntegrityError(
                f"Model artifact {artifact.identifier!r} SHA-256 is {actual_digest}; "
                f"expected {artifact.sha256}."
            )
        if chunks is None:
            return None
        return b"".join(chunks)
    finally:
        os.close(descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def _artifact_status(root: Path, artifact: ArtifactSpec) -> ArtifactStatus:
    path = _artifact_path(root, artifact)
    try:
        _managed_parent(root, artifact, create=False)
        _read_verified_file(path, artifact, collect=False)
    except ModelNotFoundError as exc:
        return ArtifactStatus(artifact, path, STATE_MISSING, str(exc))
    except ModelIntegrityError as exc:
        return ArtifactStatus(artifact, path, STATE_CORRUPT, str(exc))
    except (ModelCacheSafetyError, OSError) as exc:
        return ArtifactStatus(artifact, path, STATE_UNSAFE, str(exc))
    return ArtifactStatus(artifact, path, STATE_READY)


def inspect_bundle(
    cache_root,
    spec: ModelBundleSpec = EFFICIENTSAM_TI_SPLIT,
) -> BundleStatus:
    """Inspect all bundle objects without creating files or opening the network."""

    _validate_bundle_contract(spec)
    root = _normal_root(cache_root)
    artifacts = tuple(_artifact_status(root, artifact) for artifact in spec.artifacts)
    return BundleStatus(root=root, spec=spec, artifacts=artifacts)


def resolve_bundle(
    cache_root,
    spec: ModelBundleSpec = EFFICIENTSAM_TI_SPLIT,
) -> VerifiedBundle:
    """Resolve a complete verified bundle using only local cache bytes."""

    status = inspect_bundle(cache_root, spec)
    failures = [artifact for artifact in status.artifacts if not artifact.ready]
    if failures:
        detail = "; ".join(
            f"{artifact.spec.identifier}={artifact.state}: {artifact.detail}"
            for artifact in failures
        )
        if any(artifact.state == STATE_UNSAFE for artifact in failures):
            raise ModelCacheSafetyError(detail)
        if any(artifact.state == STATE_CORRUPT for artifact in failures):
            raise ModelIntegrityError(detail)
        raise ModelNotFoundError(detail)
    return VerifiedBundle(
        root=status.root,
        spec=spec,
        artifacts=tuple(
            VerifiedArtifact(artifact.spec, artifact.path)
            for artifact in status.artifacts
        ),
    )


def read_verified_bytes(cache_root, artifact: ArtifactSpec) -> bytes:
    """Return bounded bytes read from a regular, hash-verified cache object."""

    if not isinstance(artifact, ArtifactSpec):
        raise TypeError("artifact must be an ArtifactSpec.")
    if artifact.size_bytes > MAX_ARTIFACT_BYTES:
        raise ModelIntegrityError(
            f"Artifact {artifact.identifier!r} exceeds the {MAX_ARTIFACT_BYTES}-byte limit."
        )
    root = _normal_root(cache_root)
    _managed_parent(root, artifact, create=False)
    result = _read_verified_file(_artifact_path(root, artifact), artifact, collect=True)
    if result is None:  # Defensive: collect=True always returns bytes.
        raise ModelIntegrityError(f"Could not read artifact {artifact.identifier!r}.")
    return result


def _header(headers, name: str) -> Optional[str]:
    if headers is None:
        return None
    value = headers.get(name)
    if value is None:
        value = headers.get(name.lower())
    return None if value is None else str(value)


def _validate_response(response, artifact: ArtifactSpec) -> None:
    status = getattr(response, "status", None)
    if status is None and hasattr(response, "getcode"):
        status = response.getcode()
    if status != 200:
        raise ModelDownloadError(
            f"Artifact {artifact.identifier!r} returned HTTP status {status!r}, not 200."
        )

    final_url = response.geturl() if hasattr(response, "geturl") else artifact.url
    if final_url != artifact.url:
        raise ModelDownloadError(
            f"Artifact {artifact.identifier!r} redirected away from its fixed URL."
        )

    content_encoding = _header(getattr(response, "headers", None), "Content-Encoding")
    if content_encoding is not None and content_encoding.strip().lower() != "identity":
        raise ModelDownloadError(
            f"Artifact {artifact.identifier!r} used unsupported Content-Encoding "
            f"{content_encoding!r}."
        )

    content_length = _header(getattr(response, "headers", None), "Content-Length")
    if content_length is not None:
        try:
            parsed_length = int(content_length.strip(), 10)
        except (TypeError, ValueError) as exc:
            raise ModelDownloadError(
                f"Artifact {artifact.identifier!r} returned an invalid Content-Length."
            ) from exc
        if parsed_length != artifact.size_bytes:
            raise ModelDownloadError(
                f"Artifact {artifact.identifier!r} advertised {parsed_length} bytes; "
                f"expected {artifact.size_bytes}."
            )


def _safe_parent_stat(parent: Path) -> os.stat_result:
    return _require_safe_directory(parent, "model cache digest directory")


def _publish_no_replace(
    temporary: Path,
    destination: Path,
    expected_parent: os.stat_result,
) -> None:
    """Atomically publish a regular file by creating a no-replace hard link."""

    if temporary.parent != destination.parent:
        raise ModelCacheSafetyError("Model temporary file is not beside its destination.")
    current_parent = _safe_parent_stat(destination.parent)
    if not os.path.samestat(expected_parent, current_parent):
        raise ModelCacheSafetyError("Model cache directory changed before publishing.")

    directory_descriptor = None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_descriptor = os.open(destination.parent, flags)
        opened_parent = os.fstat(directory_descriptor)
        if not os.path.samestat(expected_parent, opened_parent):
            raise ModelCacheSafetyError("Model cache directory changed while publishing.")

        supports_link_dir_fd = os.link in getattr(os, "supports_dir_fd", set())
        supports_unlink_dir_fd = os.unlink in getattr(os, "supports_dir_fd", set())
        if supports_link_dir_fd and supports_unlink_dir_fd:
            os.link(
                temporary.name,
                destination.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            os.unlink(temporary.name, dir_fd=directory_descriptor)
        else:  # pragma: no cover - exercised on platforms without dir_fd support.
            if not os.path.samestat(expected_parent, _safe_parent_stat(destination.parent)):
                raise ModelCacheSafetyError("Model cache directory changed while publishing.")
            os.link(temporary, destination, follow_symlinks=False)
            os.unlink(temporary)

        if os.name == "posix":
            os.fsync(directory_descriptor)
    except FileExistsError:
        raise
    except OSError as exc:
        raise ModelStoreError(f"Could not atomically publish model artifact: {exc}") from exc
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def _download_artifact(
    root: Path,
    artifact: ArtifactSpec,
    *,
    transport: Callable,
    timeout_seconds: float,
) -> Path:
    parent = _managed_parent(root, artifact, create=True)
    parent_information = _safe_parent_stat(parent)
    destination = _artifact_path(root, artifact)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{artifact.identifier}.",
        suffix=".partial",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        request = urllib.request.Request(
            artifact.url,
            headers={
                "Accept-Encoding": "identity",
                "User-Agent": _USER_AGENT,
            },
            method="GET",
        )
        deadline = time.monotonic() + timeout_seconds
        try:
            response = transport(request, timeout_seconds)
        except ModelStoreError:
            raise
        except Exception as exc:
            raise ModelDownloadError(
                f"Could not open artifact {artifact.identifier!r}: {exc}"
            ) from exc

        digest = hashlib.sha256()
        total = 0
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            with closing(response):
                _validate_response(response, artifact)
                while True:
                    if time.monotonic() > deadline:
                        raise ModelDownloadError(
                            f"Artifact {artifact.identifier!r} exceeded its download deadline."
                        )
                    remaining_with_sentinel = artifact.size_bytes + 1 - total
                    if remaining_with_sentinel <= 0:
                        break
                    try:
                        chunk = response.read(
                            min(DOWNLOAD_CHUNK_BYTES, remaining_with_sentinel)
                        )
                    except Exception as exc:
                        raise ModelDownloadError(
                            f"Could not read artifact {artifact.identifier!r}: {exc}"
                        ) from exc
                    if not chunk:
                        break
                    if not isinstance(chunk, bytes):
                        raise ModelDownloadError("Model download returned a non-bytes chunk.")
                    total += len(chunk)
                    if total > artifact.size_bytes:
                        raise ModelDownloadError(
                            f"Artifact {artifact.identifier!r} exceeded its fixed "
                            f"{artifact.size_bytes}-byte size."
                        )
                    digest.update(chunk)
                    output.write(chunk)

            if total != artifact.size_bytes:
                raise ModelDownloadError(
                    f"Artifact {artifact.identifier!r} ended at {total} bytes; "
                    f"expected {artifact.size_bytes}."
                )
            actual_digest = digest.hexdigest()
            if actual_digest != artifact.sha256:
                raise ModelIntegrityError(
                    f"Artifact {artifact.identifier!r} SHA-256 is {actual_digest}; "
                    f"expected {artifact.sha256}."
                )
            output.flush()
            os.fsync(output.fileno())

        current_parent = _safe_parent_stat(parent)
        if not os.path.samestat(parent_information, current_parent):
            raise ModelCacheSafetyError("Model cache directory changed during download.")
        if not stat.S_ISREG(os.lstat(temporary).st_mode):
            raise ModelCacheSafetyError("Model temporary path is no longer a regular file.")

        try:
            _publish_no_replace(temporary, destination, parent_information)
        except FileExistsError:
            # A concurrent fetch may have won.  It is acceptable only when the
            # winner is exactly the same content-addressed object.
            _read_verified_file(destination, artifact, collect=False)

        _read_verified_file(destination, artifact, collect=False)
        return destination
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if _lexists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


def fetch_bundle(
    cache_root,
    spec: ModelBundleSpec = EFFICIENTSAM_TI_SPLIT,
    *,
    transport: Optional[Callable] = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> VerifiedBundle:
    """Explicitly fetch missing artifacts, then return the verified bundle.

    Existing corrupt or unsafe objects are never overwritten.  Callers must
    make a separate, deliberate repair decision for such cache contents.
    """

    _validate_bundle_contract(spec)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
    ):
        raise ValueError("timeout_seconds must be finite and positive.")
    selected_transport = transport or _default_transport
    if not callable(selected_transport):
        raise TypeError("transport must be callable.")

    root = _normal_root(cache_root)
    _create_safe_directory(root, "model cache root", parents=True)
    for artifact in spec.artifacts:
        status = _artifact_status(root, artifact)
        if status.state == STATE_READY:
            continue
        if status.state == STATE_CORRUPT:
            raise ModelIntegrityError(status.detail or "Cached model artifact is corrupt.")
        if status.state == STATE_UNSAFE:
            raise ModelCacheSafetyError(status.detail or "Cached model artifact is unsafe.")
        _download_artifact(
            root,
            artifact,
            transport=selected_transport,
            timeout_seconds=float(timeout_seconds),
        )
    return resolve_bundle(root, spec)


__all__ = [
    "ArtifactStatus",
    "BundleStatus",
    "DEFAULT_TIMEOUT_SECONDS",
    "DOWNLOAD_CHUNK_BYTES",
    "MAX_ARTIFACT_BYTES",
    "ModelCacheSafetyError",
    "ModelDownloadError",
    "ModelIntegrityError",
    "ModelNotFoundError",
    "ModelStoreError",
    "STATE_CORRUPT",
    "STATE_MISSING",
    "STATE_READY",
    "STATE_UNSAFE",
    "VerifiedArtifact",
    "VerifiedBundle",
    "bundle_fingerprint",
    "fetch_bundle",
    "inspect_bundle",
    "read_verified_bytes",
    "resolve_bundle",
]
