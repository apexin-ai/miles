from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # The four-layer cut of the native model (decoder layers 0-3, complete vision tower) for single-node tests.
    return load_sibling_model_args(__file__, "qwen3.8-flash-next-vision", nlayers=4)
