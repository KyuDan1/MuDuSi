import numpy as np
import pytest

from duplex_arena.acoustics import AcousticMixer, Divider, FRAME_SIZE, SAMPLE_RATE, SPEED_OF_SOUND, rms


def mixer(ids=("a", "b"), **kwargs):
    return AcousticMixer(ids, noise_db=-np.inf, reverb=0, **kwargs)


def test_impulse_propagates_causally_across_chunk_boundary():
    acoustic = mixer()
    positions = {"a": [0, 1, 0], "b": [1, 1, 0]}
    delay = int(np.ceil(SAMPLE_RATE / SPEED_OF_SOUND))
    emission = np.zeros(FRAME_SIZE, dtype=np.float32)
    emission[-2] = 1
    first = acoustic.render({"a": emission}, positions)
    assert np.count_nonzero(first["b"]) == 0
    second = acoustic.render({}, positions)
    assert np.count_nonzero(second["b"]) == 1
    assert second["b"][delay - 2] == 1
    assert np.all(second["b"][:delay - 2] == 0)


def test_distance_reduces_pressure_by_inverse_distance():
    acoustic = mixer(("a", "near", "far"))
    positions = {"a": [0, 1, 0], "near": [1, 1, 0], "far": [4, 1, 0]}
    constant = np.full(FRAME_SIZE, 0.2, dtype=np.float32)
    acoustic.render({"a": constant}, positions)
    received = acoustic.render({"a": constant}, positions)
    assert rms(received["near"]) / rms(received["far"]) == pytest.approx(4)


def test_no_self_input_even_with_reflection():
    acoustic = AcousticMixer(("a", "b"), noise_db=-np.inf, reverb=0.8)
    positions = {"a": [-2, 1, 0], "b": [2, 1, 0]}
    received = acoustic.render({"a": np.ones(FRAME_SIZE, dtype=np.float32)}, positions)
    assert np.count_nonzero(received["a"]) == 0
    assert np.count_nonzero(received["b"]) > 0


def test_noise_is_seeded_and_independent_at_each_microphone():
    first = AcousticMixer(("a", "b"), seed=7, noise_db=-35, reverb=0)
    repeat = AcousticMixer(("a", "b"), seed=7, noise_db=-35, reverb=0)
    different = AcousticMixer(("a", "b"), seed=8, noise_db=-35, reverb=0)
    positions = {"a": [0, 1, 0], "b": [1, 1, 0]}
    one = first.render({}, positions)
    two = repeat.render({}, positions)
    three = different.render({}, positions)
    np.testing.assert_array_equal(one["a"], two["a"])
    assert not np.array_equal(one["a"], one["b"])
    assert not np.array_equal(one["a"], three["a"])


def test_divider_attenuates_only_intersecting_paths():
    positions = {"a": [-2, 1, -2], "b": [2, 1, -2]}
    wall = Divider(loss_db=14)
    direct, occluded = mixer(), mixer(dividers=[wall])
    block = np.full(FRAME_SIZE, 0.2, dtype=np.float32)
    clean = direct.render({"a": block}, positions)
    blocked = occluded.render({"a": block}, positions)
    assert rms(blocked["b"]) / rms(clean["b"]) == pytest.approx(10 ** (-14 / 20))
    assert occluded.last_links[0]["occluded"] is True
    assert not wall.blocks(np.array([-2, 1, 1]), np.array([2, 1, 1]))


def test_reflection_is_later_than_direct_arrival():
    acoustic = AcousticMixer(("a", "b"), noise_db=-np.inf, reverb=0.8)
    positions = {"a": [-2, 1, 0], "b": [2, 1, 0]}
    impulse = np.zeros(FRAME_SIZE, dtype=np.float32)
    impulse[0] = 1
    received = acoustic.render({"a": impulse}, positions)["b"]
    arrivals = np.flatnonzero(received)
    assert len(arrivals) == 2
    assert arrivals[0] == int(np.ceil(4 / SPEED_OF_SOUND * SAMPLE_RATE))
    assert arrivals[1] == int(np.ceil(12 / SPEED_OF_SOUND * SAMPLE_RATE))
    assert received[arrivals[1]] < received[arrivals[0]]


@pytest.mark.parametrize("bad_audio", [np.zeros(FRAME_SIZE - 1), np.zeros((FRAME_SIZE, 1)),
                                       np.full(FRAME_SIZE, np.nan), np.full(FRAME_SIZE, np.inf)])
def test_invalid_emissions_do_not_advance_mixer(bad_audio):
    acoustic = mixer()
    positions = {"a": [0, 1, 0], "b": [1, 1, 0]}
    with pytest.raises(ValueError):
        acoustic.render({"a": bad_audio}, positions)
    assert acoustic.last_links == []
    assert acoustic._queues == {}
