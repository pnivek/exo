class ModelConfig:
    max_model_len: int

class CompilationConfig:
    mode: object
    cudagraph_mode: object
    def __init__(
        self, *, mode: object = ..., cudagraph_mode: object = ..., **kwargs: object
    ) -> None: ...

class VllmConfig:
    model_config: ModelConfig
    compilation_config: CompilationConfig
    kv_transfer_config: object | None
