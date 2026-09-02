"""Streaming rendering (video_writer / return_frame) writes zero PNGs and matches the PNG path.

Frames are drawn on real render_pool workers (ProcessPoolExecutor, spawn context) throughout --
the whole point of this path is that _draw_step can hand a raw frame back across that process
boundary instead of writing a PNG, so any test that calls it in-process proves nothing.
"""

import shutil
import subprocess

import numpy as np
import pytest

from scenario_generation.closed_loop_eval import FFmpegVideoWriter
from scenario_generation.render_pool import drain_oldest_frame, render_pool
from scenario_generation.reproducer_rollout import _draw_step
from scenario_generation.tests.test_render_pool import FIXTURE, N_FRAMES

HAS_FFMPEG = shutil.which("ffmpeg") is not None


class _FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _FakeWriter:
    def __init__(self):
        self.written = []

    def write_frame(self, frame_bytes, width, height, pix_fmt="rgba"):
        self.written.append((frame_bytes, width, height))


def test_drain_oldest_frame_is_a_noop_under_the_cap():
    pending = [_FakeFuture((b"f0", 1, 1)), _FakeFuture((b"f1", 1, 1))]
    writer = _FakeWriter()
    drain_oldest_frame(pending, writer, cap=2)
    assert writer.written == []
    assert len(pending) == 2


def test_drain_oldest_frame_pops_the_oldest_in_order():
    pending = [_FakeFuture((f"f{i}".encode(), 1, 1)) for i in range(4)]
    writer = _FakeWriter()
    drain_oldest_frame(pending, writer, cap=2)
    assert [w[0] for w in writer.written] == [b"f0"]
    assert len(pending) == 3
    drain_oldest_frame(pending, writer, cap=2)
    assert [w[0] for w in writer.written] == [b"f0", b"f1"]
    assert len(pending) == 2


def _frame_args(uuids):
    frame = dict(np.load(FIXTURE))
    np_dict = {k: v[None] for k, v in frame.items()}  # _draw_step indexes [0]
    return np_dict, frame["ego_shape"], uuids


def test_streaming_writes_no_png_files(tmp_path):
    np_dict, ego_shape, uuids = _frame_args(
        [f"{0xA3F91C2E + i * 0x1111:08x}" for i in range(6)]
    )
    with render_pool(2) as pool:
        pending = [
            pool.submit(
                _draw_step,
                np_dict,
                np.zeros((8, 4), dtype=np.float32),
                ego_shape,
                None,  # path=None: nothing is written to disk
                neighbor_ids=uuids,
                step=k,
                total=N_FRAMES,
                return_frame=True,
            )
            for k in range(N_FRAMES)
        ]
        results = [f.result() for f in pending]

    assert list(tmp_path.rglob("*.png")) == []
    for rgba, w, h in results:
        assert len(rgba) == w * h * 4


def test_streamed_frame_matches_saved_png_pixels(tmp_path):
    """The only check that would catch a resolution/dpi/channel-order drift between the two paths."""
    np_dict, ego_shape, uuids = _frame_args(
        [f"{0xA3F91C2E + i * 0x1111:08x}" for i in range(6)]
    )
    common = dict(
        pred=np.zeros((8, 4), dtype=np.float32),
        ego_shape=ego_shape,
        neighbor_ids=uuids,
        step=0,
        total=N_FRAMES,
    )
    png_path = tmp_path / "step_00000.png"
    _draw_step(np_dict, common["pred"], common["ego_shape"], png_path, neighbor_ids=uuids, step=0, total=N_FRAMES)
    rgba, w, h = _draw_step(
        np_dict, common["pred"], common["ego_shape"], None,
        neighbor_ids=uuids, step=0, total=N_FRAMES, return_frame=True,
    )

    from matplotlib import image as mpimg

    png_arr = mpimg.imread(png_path)  # float RGBA in [0, 1], shape (h, w, 4)
    assert png_arr.shape == (h, w, 4)
    streamed = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    png_as_uint8 = np.round(png_arr * 255).astype(np.uint8)
    assert np.array_equal(streamed, png_as_uint8)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg binary not available")
def test_ffmpeg_video_writer_produces_playable_mp4(tmp_path):
    w, h = 64, 48
    frame = (np.random.default_rng(0).integers(0, 255, size=(h, w, 4), dtype=np.uint8)).tobytes()
    mp4_path = tmp_path / "out.mp4"
    with FFmpegVideoWriter(mp4_path, fps=10.0) as writer:
        for _ in range(5):
            writer.write_frame(frame, w, h)
    assert writer.frame_count == 5
    assert mp4_path.exists() and mp4_path.stat().st_size > 0

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", str(mp4_path)],
        capture_output=True, text=True, check=True,
    )
    assert int(probe.stdout.strip()) == 5


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg binary not available")
def test_ffmpeg_video_writer_zero_frames_creates_no_file(tmp_path):
    mp4_path = tmp_path / "empty.mp4"
    with FFmpegVideoWriter(mp4_path, fps=10.0) as writer:
        pass
    assert writer.frame_count == 0
    assert not mp4_path.exists()


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg binary not available")
def test_ffmpeg_video_writer_size_change_asserts(tmp_path):
    mp4_path = tmp_path / "out.mp4"
    frame_a = np.zeros((32, 32, 4), dtype=np.uint8).tobytes()
    frame_b = np.zeros((16, 16, 4), dtype=np.uint8).tobytes()
    with pytest.raises(AssertionError):
        with FFmpegVideoWriter(mp4_path, fps=10.0) as writer:
            writer.write_frame(frame_a, 32, 32)
            writer.write_frame(frame_b, 16, 16)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg binary not available")
def test_ffmpeg_video_writer_close_raises_on_ffmpeg_failure(tmp_path):
    mp4_path = tmp_path / "bad.mp4"
    writer = FFmpegVideoWriter(mp4_path, fps=10.0)
    # An unsupported pix_fmt makes ffmpeg reject the arguments and exit non-zero right away.
    # Depending on scheduling, the OS pipe buffer may absorb the write before the kernel notices
    # ffmpeg is gone (surfacing the failure at close()) or raise BrokenPipeError immediately
    # (surfaced by write_frame itself, which then closes and re-raises as RuntimeError) -- either
    # is a correct outcome, so both calls are covered by one raises block.
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        writer.write_frame(b"\x00" * (4 * 4 * 4), 4, 4, pix_fmt="not_a_real_pixfmt")
        writer.close()
