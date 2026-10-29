from model_args_utils import load_sibling_model_args


def model_args() -> str:
    base = load_sibling_model_args(__file__, "qwen3.8-flash-next").replace(
        "--position-embedding-type rope", "--position-embedding-type mrope"
    )
    return base + " --mrope-section 11 11 10 --no-rope-fusion "
