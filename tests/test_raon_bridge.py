from types import SimpleNamespace

import pytest

from workers.raon_server import install_speak_first_bridge


@pytest.mark.parametrize("requested", [True, False])
def test_raon_speak_first_reaches_decoding_config(requested):
    state = SimpleNamespace(config=SimpleNamespace(speak_first=not requested), decoding_state=None)
    instance = SimpleNamespace(_inner=SimpleNamespace(state=state), _session_params={})
    module = SimpleNamespace(create_session=lambda **kwargs: instance)
    install_speak_first_bridge(module)
    assert module.create_session(session={"speak_first": requested}) is instance
    assert state.config.speak_first is requested
    assert instance._session_params["speak_first"] is requested


def test_raon_bridge_rejects_mode_change_after_decoding_started():
    state = SimpleNamespace(config=SimpleNamespace(speak_first=False), decoding_state=object())
    instance = SimpleNamespace(_inner=SimpleNamespace(state=state), _session_params={})
    module = SimpleNamespace(create_session=lambda **kwargs: instance)
    install_speak_first_bridge(module)
    with pytest.raises(RuntimeError, match="initialized decoding"):
        module.create_session(session={"speak_first": True})
