import pytest

from workers.edit_film import edit_plan


def test_crossfades_preserve_entire_spoken_intervals():
    plan = edit_plan([{'kind': 'card', 'seconds': 4}, {'kind': 'card', 'seconds': 3},
                      {'kind': 'video', 'seconds': 27}, {'kind': 'card', 'seconds': 3},
                      {'kind': 'video', 'seconds': 30}, {'kind': 'card', 'seconds': 3},
                      {'kind': 'video', 'seconds': 15}], .5)
    assert plan[-1]['end'] == 85
    for i, clip in enumerate(plan):
        if clip['kind'] == 'video':
            assert clip['speech_start'] == clip['start'] + .5
            assert clip['speech_end'] == clip['end'] - .5
            assert clip['speech_end'] - clip['speech_start'] == clip['seconds']
            if i + 1 < len(plan):
                assert plan[i + 1]['start'] == clip['speech_end']


def test_short_cards_are_rejected():
    with pytest.raises(ValueError, match='too short'):
        edit_plan([{'kind': 'card', 'seconds': .4}], .5)
