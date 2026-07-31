class SamplingParams:
    n: int
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    seed: int | None
    stop: str | list[str] | None
    stop_token_ids: list[int] | None
    max_tokens: int | None
    min_tokens: int
    logprobs: int | None
    repetition_penalty: float
    ignore_eos: bool
    detokenize: bool
    # exo2 fork addition: skip the APC lookup for benchmark requests.
    skip_reading_prefix_cache: bool

    def __init__(
        self,
        *,
        n: int = ...,
        temperature: float = ...,
        top_p: float = ...,
        top_k: int = ...,
        min_p: float = ...,
        seed: int | None = ...,
        stop: str | list[str] | None = ...,
        stop_token_ids: list[int] | None = ...,
        max_tokens: int | None = ...,
        min_tokens: int = ...,
        logprobs: int | None = ...,
        repetition_penalty: float = ...,
        ignore_eos: bool = ...,
        detokenize: bool = ...,
        skip_reading_prefix_cache: bool = ...,
    ) -> None: ...
