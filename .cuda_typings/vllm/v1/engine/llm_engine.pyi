from collections.abc import Iterable, Mapping, Sequence

from vllm.config import ModelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike

# vLLM's `PromptType`: a raw string, a token-ids/embeds mapping such as
# `{"prompt_token_ids": [...]}`, or a bare token sequence.
type PromptType = str | Mapping[str, object] | Sequence[int]

class LLMEngine:
    tokenizer: TokenizerLike | None
    model_config: ModelConfig

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> LLMEngine: ...
    def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: SamplingParams,
        arrival_time: float | None = ...,
    ) -> None: ...
    def abort_request(self, request_ids: str | Iterable[str]) -> None: ...
    def step(self) -> list[RequestOutput | PoolingRequestOutput]: ...
    def has_unfinished_requests(self) -> bool: ...
    def get_tokenizer(self) -> TokenizerLike: ...
