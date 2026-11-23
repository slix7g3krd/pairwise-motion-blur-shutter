import pytest
import graphics.motion_blur_shutter as mod

CASES = [
    ("shutter_time", (1)),
    ("vector_compose", (1)),
    ("history_split", (1)),
    ("skin_motion", (1)),
    ("result_align", (1)),
    ("trail_cap", (1)),
    ("rotation_path", (1)),
    ("validation", (1)),
    ("audit", (1)),
    ("fallback", (1)),
    ("frame_drop", (1)),
    ("metrics", (1)),
    ("camera_jitter", (1)),
    ("async_post", (1)),
    ("determinism", (1)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
