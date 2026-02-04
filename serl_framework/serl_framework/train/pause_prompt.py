"""Interactive pause and resume prompts for long-running training loops.

The learner can periodically offer the trainer a short window to pause training.
The offer times out after a few seconds, so unattended runs keep going without any
interaction. When the trainer accepts, the loop idles until Enter is pressed.
"""

import os
import select
import sys
import termios
import tty
from typing import Callable, TextIO

PAUSE_PROMPT_DEFAULT_TIMEOUT_SEC = 5.0


def flush_pending_input(in_stream: TextIO | None = None) -> None:
    """Discard any input already typed on a tty stream.

    Without this, keypresses made before a prompt appears (e.g. an answer that
    missed an earlier prompt's timeout window) would be read as the answer to
    the new prompt. Does nothing for non tty streams.
    """
    stream = in_stream if in_stream is not None else sys.stdin
    if stream.isatty():
        termios.tcflush(stream.fileno(), termios.TCIFLUSH)


def read_key_with_timeout(timeout_sec: float, in_stream: TextIO | None = None) -> str | None:
    """Wait up to ``timeout_sec`` for input on ``in_stream`` and return it.

    On an interactive terminal a single keypress is read immediately, without
    waiting for Enter. On non interactive streams (e.g. pipes in tests) a full
    line is read instead. Returns ``None`` if no input arrives within the
    timeout or the stream is at EOF.
    """
    stream = in_stream if in_stream is not None else sys.stdin
    if stream.isatty():
        return _read_single_key(stream, timeout_sec)
    ready, _, _ = select.select([stream], [], [], timeout_sec)
    if not ready:
        return None
    line = stream.readline()
    return line.strip() if line else None


def _read_single_key(stream: TextIO, timeout_sec: float) -> str | None:
    """Read one raw keypress from a tty stream, or ``None`` on timeout.

    The terminal is switched to cbreak mode only for the duration of the read,
    then restored, so later ``readline()`` calls behave normally again.
    """
    fd = stream.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        # TCSADRAIN keeps pending input: a key pressed right after the prompt
        # appeared must not be discarded by the mode switch (the setcbreak
        # default TCSAFLUSH would drop it). Stale input from before the prompt
        # is already handled by flush_pending_input().
        tty.setcbreak(fd, termios.TCSADRAIN)
        ready, _, _ = select.select([fd], [], [], timeout_sec)
        if not ready:
            return None
        return os.read(fd, 1).decode(errors="replace")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def wait_for_enter(in_stream: TextIO | None = None) -> None:
    """Block until the user presses Enter (or the stream hits EOF)."""
    stream = in_stream if in_stream is not None else sys.stdin
    stream.readline()


def offer_training_pause(
    timeout_sec: float = PAUSE_PROMPT_DEFAULT_TIMEOUT_SEC,
    in_stream: TextIO | None = None,
    write: Callable[[str], None] = print,
) -> bool:
    """Offer to pause training and, if accepted, block until the user resumes.

    Prints a yes/no prompt and waits up to ``timeout_sec`` for an answer.
    Only an answer of ``y`` or ``Y`` pauses. Anything else, including no answer
    within the timeout, resumes immediately. While paused, this call blocks
    until the user presses Enter. Returns ``True`` if a pause happened.
    """
    flush_pending_input(in_stream)
    write(f">>> Do you want to pause the training? Y/n (auto-resume in {timeout_sec:.0f}s) <<<")
    answer = read_key_with_timeout(timeout_sec, in_stream=in_stream)
    if answer is None:
        write(f">>> No answer within {timeout_sec:.0f}s, training continues. <<<")
        return False
    if answer.strip().lower() != "y":
        write(f">>> Answered {answer.strip()!r}, training continues. <<<")
        return False
    # Drop any trailing input (e.g. the Enter of a habitual "y<Enter>" answer),
    # which would otherwise end the pause immediately.
    flush_pending_input(in_stream)
    write(">>> TRAINING PAUSED. Press [enter] to resume training. <<<")
    wait_for_enter(in_stream=in_stream)
    write(">>> Training resumed. <<<")
    return True
