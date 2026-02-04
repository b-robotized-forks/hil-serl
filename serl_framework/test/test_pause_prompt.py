"""Tests for the interactive training pause and resume prompts.

The tests drive the prompt helpers through OS pipes, which exercise the non tty
code path (line based reads with a select timeout).
"""

import os
import pty

from serl_framework.train.pause_prompt import (
    flush_pending_input,
    offer_training_pause,
    read_key_with_timeout,
    wait_for_enter,
)


class _PipeStream:
    """Context manager providing a readable stream fed by a writable pipe end."""

    def __init__(self, data: str = "", close_writer: bool = True):
        self._read_fd, self._write_fd = os.pipe()
        if data:
            os.write(self._write_fd, data.encode())
        self._writer_open = True
        if close_writer:
            self._close_writer()
        self.reader = os.fdopen(self._read_fd, "r")

    def _close_writer(self):
        if self._writer_open:
            os.close(self._write_fd)
            self._writer_open = False

    def __enter__(self):
        return self.reader

    def __exit__(self, exc_type, exc_value, traceback):
        self._close_writer()
        self.reader.close()
        return False


def test_read_key_returns_input_line():
    with _PipeStream("y\n") as stream:
        assert read_key_with_timeout(1.0, in_stream=stream) == "y"


def test_read_key_times_out_without_input():
    # Keep the writer open so the reader neither gets data nor EOF.
    with _PipeStream(close_writer=False) as stream:
        assert read_key_with_timeout(0.05, in_stream=stream) is None


def test_read_key_returns_none_on_eof():
    with _PipeStream() as stream:
        assert read_key_with_timeout(0.05, in_stream=stream) is None


def test_offer_pause_accepts_and_waits_for_enter():
    messages = []
    with _PipeStream("y\n\n") as stream:
        paused = offer_training_pause(timeout_sec=1.0, in_stream=stream, write=messages.append)
    assert paused is True
    assert any("pause the training" in msg for msg in messages)
    assert any("resume training" in msg for msg in messages)


def test_offer_pause_accepts_uppercase_answer():
    with _PipeStream("Y\n\n") as stream:
        assert offer_training_pause(timeout_sec=1.0, in_stream=stream, write=lambda _msg: None)


def test_offer_pause_declines_on_no_and_reports_answer():
    messages = []
    with _PipeStream("n\n") as stream:
        assert not offer_training_pause(timeout_sec=1.0, in_stream=stream, write=messages.append)
    assert any("'n'" in msg and "continues" in msg for msg in messages)


def test_offer_pause_declines_on_timeout_and_reports_it():
    messages = []
    with _PipeStream(close_writer=False) as stream:
        assert not offer_training_pause(timeout_sec=0.05, in_stream=stream, write=messages.append)
    assert any("No answer" in msg and "continues" in msg for msg in messages)


def test_flush_pending_input_is_noop_on_non_tty():
    # Pipes are not ttys, so the flush must leave the buffered data intact.
    with _PipeStream("y\n") as stream:
        flush_pending_input(stream)
        assert stream.readline() == "y\n"


def test_wait_for_enter_consumes_line():
    with _PipeStream("\nleftover\n") as stream:
        wait_for_enter(in_stream=stream)
        assert stream.readline() == "leftover\n"


class _PtyStream:
    """Context manager providing a real tty slave stream fed via the pty master."""

    def __init__(self):
        self.master_fd, self._slave_fd = pty.openpty()
        self.reader = os.fdopen(self._slave_fd, "r")

    def type(self, data: str):
        os.write(self.master_fd, data.encode())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        os.close(self.master_fd)
        self.reader.close()
        return False


def test_offer_pause_on_tty_single_keypress_without_enter():
    # Drive the real tty code path: the y keypress (no Enter) must pause, and a
    # later Enter must resume. Keys are typed in reaction to the prompts.
    with _PtyStream() as tty_stream:
        def _write(msg: str):
            if "pause the training" in msg:
                tty_stream.type("y")
            elif "Press [enter]" in msg:
                tty_stream.type("\n")

        assert offer_training_pause(timeout_sec=2.0, in_stream=tty_stream.reader, write=_write)


def test_offer_pause_on_tty_discards_stale_keypress():
    # Regression test: a key typed before the prompt appears (e.g. an answer
    # that missed an earlier prompt) must not silently answer the new prompt.
    with _PtyStream() as tty_stream:
        tty_stream.type("Y")
        paused = offer_training_pause(
            timeout_sec=0.1, in_stream=tty_stream.reader, write=lambda _msg: None
        )
    assert paused is False


def test_offer_pause_on_tty_ignores_trailing_enter_after_y():
    # A habitual "y<Enter>" answer must not end the pause immediately. The
    # trailing Enter is flushed, so only a later Enter resumes.
    with _PtyStream() as tty_stream:
        def _write(msg: str):
            if "pause the training" in msg:
                tty_stream.type("y\n")
            elif "Press [enter]" in msg:
                tty_stream.type("\n")

        assert offer_training_pause(timeout_sec=2.0, in_stream=tty_stream.reader, write=_write)
