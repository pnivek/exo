from collections.abc import Generator

from exo.shared.types.api import FinishReason
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from exo.worker.runner.llm_inference.model_output_parsers import (
    parse_gpt_oss,
    strip_harmony_tokens,
)

# Token IDs from mlx-community/gpt-oss-20b-MXFP4-Q8 tokenizer.
# These are stable since they come from the model's vocabulary.
_CHANNEL = 200005  # <|channel|>
_START = 200006  # <|start|>
_MESSAGE = 200008  # <|message|>
_CALL = 200012  # <|call|>
_END = 200007  # <|end|>
_ASSISTANT = 173781  # "assistant"

# fmt: off
# " to=functions.get_current_weather<|channel|>commentary json<|message|>{\"location\": \"Tokyo\"}<|call|>"
FORMAT_A_TOKENS: list[tuple[int, str]] = [
    (316,    " to"),
    (28,     "="),
    (44580,  "functions"),
    (775,    ".get"),
    (23981,  "_current"),
    (170154, "_weather"),
    (_CHANNEL, "<|channel|>"),
    (12606,  "comment"),
    (815,    "ary"),
    (5701,   " json"),
    (_MESSAGE, "<|message|>"),
    (10848,  '{"'),
    (7693,   "location"),
    (1243,   '":'),
    (392,    ' "'),
    (173844, "Tokyo"),
    (18583,  '"}'),
    (_CALL,  "<|call|>"),
]

# "<|channel|>commentary to=functions.get_current_weather json<|message|>{\"location\": \"Tokyo\"}<|call|>"
FORMAT_B_TOKENS: list[tuple[int, str]] = [
    (_CHANNEL, "<|channel|>"),
    (12606,  "comment"),
    (815,    "ary"),
    (316,    " to"),
    (28,     "="),
    (44580,  "functions"),
    (775,    ".get"),
    (23981,  "_current"),
    (170154, "_weather"),
    (5701,   " json"),
    (_MESSAGE, "<|message|>"),
    (10848,  '{"'),
    (7693,   "location"),
    (1243,   '":'),
    (392,    ' "'),
    (173844, "Tokyo"),
    (18583,  '"}'),
    (_CALL,  "<|call|>"),
]

# "<|channel|>analysis<|message|>Let me think...<|end|><|start|>assistant<|channel|>commentary to=functions.X ..."
# Full analysis-then-tool-call as the model actually generates it.
THINKING_THEN_TOOL_TOKENS: list[tuple[int, str]] = [
    (_CHANNEL, "<|channel|>"),
    (35644,  "analysis"),
    (_MESSAGE, "<|message|>"),
    (12845,  "Let"),
    (668,    " me"),
    (2411,   " think"),
    (1078,   " about"),
    (495,    " this"),
    (13,     "."),
    (_END,   "<|end|>"),
    # Model generates a new message header for the tool call:
    (_START, "<|start|>"),
    (_ASSISTANT, "assistant"),
    *FORMAT_B_TOKENS,
]
# fmt: on


def _make_gen_responses(
    tokens: list[tuple[int, str]],
    last_finish_reason: FinishReason = "stop",
) -> list[GenerationResponse]:
    """Build GenerationResponse list from (token_id, text) pairs."""
    responses: list[GenerationResponse] = []
    for i, (tid, text) in enumerate(tokens):
        is_last = i == len(tokens) - 1
        responses.append(
            GenerationResponse(
                text=text,
                token=tid,
                finish_reason=last_finish_reason if is_last else None,
                usage=None,
            )
        )
    return responses


def _collect(
    tokens: list[tuple[int, str]],
    last_finish_reason: FinishReason = "stop",
) -> list[GenerationResponse | ToolCallResponse]:
    """Feed tokens through parse_gpt_oss and collect all yielded responses."""

    def _gen() -> Generator[GenerationResponse, None, None]:
        yield from _make_gen_responses(tokens, last_finish_reason)

    return list(x for x in parse_gpt_oss(_gen()) if x is not None)


def _get_tool_call(
    results: list[GenerationResponse | ToolCallResponse],
) -> ToolCallResponse:
    """Extract the single ToolCallResponse from results."""
    tool_calls = [r for r in results if isinstance(r, ToolCallResponse)]
    assert len(tool_calls) == 1, f"Expected 1 ToolCallResponse, got {len(tool_calls)}"
    return tool_calls[0]


class TestParseGptOssRecipientPlacement:
    """Both Harmony recipient placements must produce identical tool calls."""

    def test_format_a_yields_tool_call(self):
        results = _collect(FORMAT_A_TOKENS)
        tc = _get_tool_call(results)
        assert tc.tool_calls[0].name == "get_current_weather"
        assert '"location"' in tc.tool_calls[0].arguments
        assert "Tokyo" in tc.tool_calls[0].arguments

    def test_format_b_yields_tool_call(self):
        results = _collect(FORMAT_B_TOKENS)
        tc = _get_tool_call(results)
        assert tc.tool_calls[0].name == "get_current_weather"
        assert '"location"' in tc.tool_calls[0].arguments
        assert "Tokyo" in tc.tool_calls[0].arguments

    def test_both_formats_produce_identical_tool_calls(self):
        tc_a = _get_tool_call(_collect(FORMAT_A_TOKENS))
        tc_b = _get_tool_call(_collect(FORMAT_B_TOKENS))
        assert tc_a.tool_calls[0].name == tc_b.tool_calls[0].name
        assert tc_a.tool_calls[0].arguments == tc_b.tool_calls[0].arguments


class TestParseGptOssThinkingThenToolCall:
    """Analysis (thinking) followed by a tool call must yield both."""

    def test_thinking_then_tool_call(self):
        results = _collect(THINKING_THEN_TOOL_TOKENS)

        # Thinking tokens should have is_thinking=True and no <think> tags
        thinking_responses = [
            r for r in results if isinstance(r, GenerationResponse) and r.is_thinking
        ]
        thinking_text = "".join(r.text for r in thinking_responses)
        assert "Let me think about this." in thinking_text
        assert "<think>" not in thinking_text
        assert "</think>" not in thinking_text

        # Non-thinking tokens should have is_thinking=False
        non_thinking = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and not r.is_thinking
        ]
        non_thinking_text = "".join(r.text for r in non_thinking)
        assert "<think>" not in non_thinking_text

        # And the tool call
        tc = _get_tool_call(results)
        assert tc.tool_calls[0].name == "get_current_weather"
        assert "Tokyo" in tc.tool_calls[0].arguments


# fmt: off
# Simulates what gpt-oss generates in disagg mode: <|message|> directly
# (skipping <|channel|>), junk content, then a self-corrected full block.
DISAGG_FALLBACK_TOKENS: list[tuple[int, str]] = [
    (_MESSAGE, "<|message|>"),         # triggers HarmonyError (no channel)
    (1620,     "We"),
    (1182,     " just"),
    (3553,     " give"),
    (261,      " a"),
    (142958,   " haiku"),
    (13,       "."),
    (_END,     "<|end|>"),
    (_START,   "<|start|>"),
    (_ASSISTANT, "assistant"),
    (_CHANNEL, "<|channel|>"),
    (15824,    "final"),
    (_MESSAGE, "<|message|>"),
    (68127,    "Wind"),
    (118411,   " whispers"),
    (1752,     " through"),
    (26343,    " waves"),
]

# Analysis channel content in the stripped fallback.
DISAGG_THINKING_TOKENS: list[tuple[int, str]] = [
    (_MESSAGE, "<|message|>"),         # triggers HarmonyError
    (12845,    "Let"),
    (668,      " me"),
    (_END,     "<|end|>"),
    (_START,   "<|start|>"),
    (_ASSISTANT, "assistant"),
    (_CHANNEL, "<|channel|>"),
    (35644,    "analysis"),
    (_MESSAGE, "<|message|>"),
    (35676,    "Thinking"),
    (1078,     " about"),
    (495,      " this"),
    (_END,     "<|end|>"),
    (_START,   "<|start|>"),
    (_ASSISTANT, "assistant"),
    (_CHANNEL, "<|channel|>"),
    (15824,    "final"),
    (_MESSAGE, "<|message|>"),
    (35676,    "Answer"),
    (668,      " here"),
]

# Truncated tool call: recipient + channel + message + partial args, no <|call|>
TRUNCATED_TOOL_CALL_TOKENS: list[tuple[int, str]] = [
    (316,    " to"),
    (28,     "="),
    (44580,  "functions"),
    (775,    ".get"),
    (23981,  "_current"),
    (170154, "_weather"),
    (_CHANNEL, "<|channel|>"),
    (12606,  "comment"),
    (815,    "ary"),
    (5701,   " json"),
    (_MESSAGE, "<|message|>"),
    (10848,  '{"'),
    (7693,   "location"),
    (1243,   '":'),
    (392,    ' "'),
    (173844, "Tokyo"),
    # No <|call|> — generation truncated here
]

# Plain text tokens (no tool call)
PLAIN_TEXT_TOKENS: list[tuple[int, str]] = [
    (_CHANNEL, "<|channel|>"),
    (35644,  "analysis"),
    (_MESSAGE, "<|message|>"),
    (12845,  "Let"),
    (668,    " me"),
    (2411,   " think"),
    (1078,   " about"),
    (495,    " this"),
    (13,     "."),
    (_END,   "<|end|>"),
    (_START, "<|start|>"),
    (_ASSISTANT, "assistant"),
    (_CHANNEL, "<|channel|>"),
    (12606,  "comment"),
    (815,    "ary"),
    (_MESSAGE, "<|message|>"),
    (9906,   "Hello"),
    (14,     ","),
    (2989,   " world"),
]
# fmt: on


def _collect_strip(
    tokens: list[tuple[int, str]],
) -> list[GenerationResponse]:
    """Feed tokens through _strip_harmony_tokens and collect results."""
    responses = _make_gen_responses(tokens)
    first = responses[0]

    def _gen() -> Generator[GenerationResponse, None, None]:
        yield from responses[1:]

    return list(strip_harmony_tokens(first, _gen()))


class TestStripHarmonyTokens:
    """Harmony stripping fallback for disaggregated mode."""

    def test_strips_control_tokens(self):
        results = _collect_strip(DISAGG_FALLBACK_TOKENS)
        text = "".join(r.text for r in results)
        # Control tokens must not appear in output
        assert "<|message|>" not in text
        assert "<|end|>" not in text
        assert "<|start|>" not in text
        assert "<|channel|>" not in text
        # Role and channel name metadata must be stripped
        assert "assistant" not in text
        assert "final" not in text.split("Wind")[0] if "Wind" in text else True

    def test_yields_content_from_both_blocks(self):
        results = _collect_strip(DISAGG_FALLBACK_TOKENS)
        text = "".join(r.text for r in results)
        # Content from the first (channel-less) block
        assert "We just give a haiku." in text
        # Content from the self-corrected block
        assert "Wind whispers through waves" in text

    def test_all_tokens_not_thinking(self):
        results = _collect_strip(DISAGG_FALLBACK_TOKENS)
        for r in results:
            assert not r.is_thinking

    def test_thinking_channel_sets_is_thinking(self):
        results = _collect_strip(DISAGG_THINKING_TOKENS)
        thinking = [r for r in results if r.is_thinking]
        non_thinking = [r for r in results if not r.is_thinking]
        thinking_text = "".join(r.text for r in thinking)
        non_thinking_text = "".join(r.text for r in non_thinking)
        assert "Thinking about this" in thinking_text
        assert "Answer here" in non_thinking_text


# fmt: off
# Template-aware disagg: <|channel|> is injected synthetically by the decode
# node (matching the <|channel|> the prefill node appended to the prompt).
# The model then generates the channel name, <|message|>, and content.
TEMPLATE_AWARE_DISAGG_TOKENS: list[tuple[int, str]] = [
    (_CHANNEL, "<|channel|>"),   # Synthetic injection
    (15824,    "final"),         # Model generates channel name
    (_MESSAGE, "<|message|>"),
    (68127,    "Hello"),
    (7029,     " world"),
]

TEMPLATE_AWARE_DISAGG_THINKING_TOKENS: list[tuple[int, str]] = [
    (_CHANNEL, "<|channel|>"),   # Synthetic injection
    (35644,    "analysis"),      # Model chooses thinking
    (_MESSAGE, "<|message|>"),
    # Use tokens known to decode identically in the Harmony encoding.
    (12845,    "Let"),
    (668,      " me"),
    (2411,     " think"),
    (_END,     "<|end|>"),
    (_START,   "<|start|>"),
    (_ASSISTANT, "assistant"),
    (_CHANNEL, "<|channel|>"),
    (15824,    "final"),
    (_MESSAGE, "<|message|>"),
    (68127,    "Answer"),
    (668,      " here"),
]
# fmt: on


class TestTemplateAwareDisagg:
    """Template-aware disagg: synthetic <|channel|> enables parse_gpt_oss."""

    def test_content_yields_without_fallback(self):
        """parse_gpt_oss processes synthetic <|channel|> → content correctly."""
        results = _collect(TEMPLATE_AWARE_DISAGG_TOKENS)
        content = [r for r in results if isinstance(r, GenerationResponse)]
        text = "".join(r.text for r in content)
        assert "Hello world" in text
        # No ToolCallResponse should appear
        assert not any(isinstance(r, ToolCallResponse) for r in results)

    def test_thinking_then_content(self):
        """Synthetic <|channel|> + analysis → thinking, then final → content.

        The analysis channel is parsed correctly by parse_gpt_oss.  The
        second message block (<|end|><|start|>assistant<|channel|>final)
        triggers a HarmonyError and falls back to strip_harmony_tokens,
        which still yields the final-channel content correctly.
        """
        results = _collect(TEMPLATE_AWARE_DISAGG_THINKING_TOKENS)
        thinking = [
            r for r in results if isinstance(r, GenerationResponse) and r.is_thinking
        ]
        non_thinking = [
            r
            for r in results
            if isinstance(r, GenerationResponse) and not r.is_thinking
        ]
        thinking_text = "".join(r.text for r in thinking)
        non_thinking_text = "".join(r.text for r in non_thinking)
        # Analysis channel detected as thinking by parse_gpt_oss.
        assert "Let me think" in thinking_text
        # Final channel content recovered via strip_harmony_tokens fallback.
        assert "Answer here" in non_thinking_text


class TestParseGptOssMaxTokensTruncation:
    """Truncated tool calls must still yield finish_reason."""

    def test_truncated_tool_call_yields_finish_reason(self):
        results = _collect(TRUNCATED_TOOL_CALL_TOKENS, last_finish_reason="length")
        gen_responses = [r for r in results if isinstance(r, GenerationResponse)]
        finish_reasons = [
            r.finish_reason for r in gen_responses if r.finish_reason is not None
        ]
        assert "length" in finish_reasons

    def test_truncated_tool_call_emits_partial_args(self):
        results = _collect(TRUNCATED_TOOL_CALL_TOKENS, last_finish_reason="length")
        gen_responses = [r for r in results if isinstance(r, GenerationResponse)]
        last = [r for r in gen_responses if r.finish_reason is not None][-1]
        assert len(last.text) > 0

    def test_truncated_plain_text_still_works(self):
        results = _collect(PLAIN_TEXT_TOKENS, last_finish_reason="length")
        gen_responses = [r for r in results if isinstance(r, GenerationResponse)]
        finish_reasons = [
            r.finish_reason for r in gen_responses if r.finish_reason is not None
        ]
        assert "length" in finish_reasons
        # Verify non-empty text was yielded (delta text differs from raw token text
        # due to Harmony encoding, so we just check something was emitted)
        all_text = "".join(r.text for r in gen_responses)
        assert len(all_text) > 0
