"""
acsys compatibility check.

The ACNET backend (backend/acnet_scanner.py) relies on the acsys v1 DPM reply
API, specifically that reply objects yielded by ``DPMContext.process()`` expose
the boolean discriminators ``reply.isReading`` and ``reply.isStatus``. Newer or
stub builds of acsys expose a different API (e.g. ``is_reading_for`` /
``is_status_for``), which would silently break read/set status handling.

``assert_compatible_api()`` is the single gate the backend calls at import time
to fail loudly and early on an incompatible install, naming the expected pinned
version (see ``EXPECTED_ACSYS_VERSION`` in config/settings.py) versus what was
actually found.

This module is importable and side-effect-free until ``assert_compatible_api()``
is explicitly called.
"""

from config.settings import EXPECTED_ACSYS_VERSION

# Attribute names that make up the v1 reply API this codebase depends on.
# acnet_scanner.py branches on ``reply.isReading`` and ``reply.isStatus``.
_REQUIRED_REPLY_ATTRS = ("isReading", "isStatus")

# Reply classes that, in the v1 API, carry the boolean discriminators above.
# We look these up by name so that a missing class is treated the same as a
# missing attribute (i.e. an incompatible API).
_REPLY_CLASS_NAMES = ("ScalarReply", "StatusReply")


def _installed_version():
    """Return the installed acsys version string, or None if unavailable."""
    try:
        import acsys
    except Exception:
        return None
    return getattr(acsys, "__version__", None)


def _has_v1_reply_api():
    """True if the installed acsys exposes the v1 ``isReading``/``isStatus`` API.

    The discriminators may be defined as plain attributes or as property
    descriptors anywhere in a reply class's MRO, so we scan the MRO of each
    candidate reply class rather than relying on ``hasattr`` against the leaf
    class alone.
    """
    try:
        import acsys.dpm as dpm
    except Exception:
        return False

    for cls_name in _REPLY_CLASS_NAMES:
        cls = getattr(dpm, cls_name, None)
        if cls is None:
            return False
        mro = getattr(cls, "__mro__", (cls,))
        for attr in _REQUIRED_REPLY_ATTRS:
            if not any(attr in vars(klass) for klass in mro):
                return False
    return True


def assert_compatible_api():
    """Verify the installed acsys is compatible with this code's reply API.

    Passes if EITHER the installed acsys exposes the v1 reply API
    (``reply.isReading`` / ``reply.isStatus``) OR the installed
    ``acsys.__version__`` matches ``EXPECTED_ACSYS_VERSION``.

    Raises:
        RuntimeError: if acsys is not importable, or the installed version does
            not match ``EXPECTED_ACSYS_VERSION`` and the v1 reply API is absent.
            The message names the expected pinned version and what was found.
    """
    version = _installed_version()

    if version is None:
        try:
            import acsys  # noqa: F401  (presence check only)
        except Exception as exc:
            raise RuntimeError(
                "acsys is not importable; this code requires acsys "
                "==%s (the pinned production version). Import error: %s"
                % (EXPECTED_ACSYS_VERSION, exc)
            )

    if version == EXPECTED_ACSYS_VERSION:
        return

    if _has_v1_reply_api():
        return

    raise RuntimeError(
        "Incompatible acsys install: this code requires the v1 DPM reply API "
        "(reply.isReading / reply.isStatus) or acsys==%s, but found version "
        "%r without the expected reply API. Install acsys==%s (the pinned "
        "production version recorded in the monitor logs)."
        % (
            EXPECTED_ACSYS_VERSION,
            version if version is not None else "<no __version__>",
            EXPECTED_ACSYS_VERSION,
        )
    )
