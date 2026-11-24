"""`--text-only`: a MiniMax-M3 VL checkpoint served without its vision encoder."""

from types import SimpleNamespace

from atom.models.minimax_m3.image_processing import image_input_refusal


def _config(text_only):
    return SimpleNamespace(
        text_only=text_only,
        hf_config=SimpleNamespace(vision_config=object()),
        sequence_parallel_size=1,
        enforce_eager=False,
        compilation_config=SimpleNamespace(cudagraph_mode=None),
    )


def test_text_only_refuses_image_input():
    # The refusal is what skips building and loading the tower (`MiniMaxM3SparseForConditionalGenerationVL`)
    # and what turns an image request away with this reason (`build_minimax_m3_inputs`).
    assert image_input_refusal(_config(True)) == "--text-only"


def test_images_accepted_by_default():
    assert image_input_refusal(_config(False)) is None
