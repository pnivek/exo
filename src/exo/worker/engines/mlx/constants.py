# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

KV_GROUP_SIZE: int | None = 32
KV_BITS: int | None = None
ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = 3200
KEEP_KV_SIZE: int | None = 1600
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
KV_CACHE_BITS: int | None = None

DEFAULT_TOP_LOGPROBS: int = 5

# Number of tail prompt tokens the decode node re-prefills in disaggregated
# inference.  The prefill node sends KV for all positions *except* the last
# DISAGG_REPREFILL_TOKENS, and the decode node recomputes those using its own
# model.  This "grounds" the hidden states so MoE expert routing on the
# decode device is consistent with its own computation, avoiding divergence
# caused by cross-device numerical differences in the transferred KV cache.
DISAGG_REPREFILL_TOKENS: int = 64

# gpt-oss Harmony protocol: the <|channel|> control token ID.
# The prefill node appends this to the prompt so the decode node's first
# generated token is the channel name (not a protocol marker that MoE
# routing divergence might corrupt).
HARMONY_CHANNEL_TOKEN_ID: int = 200005

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
