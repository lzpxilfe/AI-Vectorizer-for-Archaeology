"""Shared UI states for the optional Smart Recovery workflow.

The recovery model is a challenger only.  These labels describe what happened
to the current, uncommitted Ink segment; they never imply that a model result
was committed without the user's normal click.
"""

RECOVERY_STATE_INK = "Ink"
RECOVERY_STATE_RECOVERING = "Recovering"
RECOVERY_STATE_ENHANCED = "Enhanced"
RECOVERY_STATE_INK_FALLBACK = "Ink fallback"

RECOVERY_STATES = frozenset(
    {
        RECOVERY_STATE_INK,
        RECOVERY_STATE_RECOVERING,
        RECOVERY_STATE_ENHANCED,
        RECOVERY_STATE_INK_FALLBACK,
    }
)


def require_recovery_state(state):
    """Validate one public state before it crosses the tool/UI boundary."""

    if state not in RECOVERY_STATES:
        raise ValueError("Unknown Smart Recovery state: {!r}".format(state))
    return state


__all__ = [
    "RECOVERY_STATE_ENHANCED",
    "RECOVERY_STATE_INK",
    "RECOVERY_STATE_INK_FALLBACK",
    "RECOVERY_STATE_RECOVERING",
    "RECOVERY_STATES",
    "require_recovery_state",
]
