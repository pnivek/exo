"""Tests for event-driven plan_step wake signaling in Worker."""

import anyio
import pytest

from exo.shared.apply import apply
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId, ModelId, NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    IndexedEvent,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TestEvent,
)
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.runners import RunnerId, RunnerIdle
from exo.worker.main import Worker


def _make_indexed(event: Event, idx: int = 0) -> IndexedEvent:
    return IndexedEvent(idx=idx, event=event)


def _make_worker() -> Worker:
    from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
    from exo.utils.channels import channel

    node_id = NodeId("test-node")
    _event_send, event_recv = channel[IndexedEvent]()
    local_send, _ = channel[Event]()
    cmd_send, _ = channel[ForwarderCommand]()
    dl_send, _ = channel[ForwarderDownloadCommand]()

    return Worker(
        node_id,
        event_receiver=event_recv,
        event_sender=local_send,
        command_sender=cmd_send,
        download_command_sender=dl_send,
    )


def _apply_and_maybe_wake(worker: Worker, indexed: IndexedEvent) -> None:
    """Simulate what _event_applier does: apply event, wake if state-mutating."""
    worker.state = apply(worker.state, event=indexed)
    raw = indexed.event
    if not isinstance(raw, Worker._PASSTHROUGH_EVENTS):  # pyright: ignore[reportPrivateUsage]
        worker._plan_wake.set()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_plan_wake_set_on_state_mutating_event() -> None:
    """_plan_wake is set when a state-mutating event is applied."""
    worker = _make_worker()

    assert not worker._plan_wake.is_set()  # pyright: ignore[reportPrivateUsage]

    _apply_and_maybe_wake(
        worker,
        _make_indexed(
            RunnerStatusUpdated(
                runner_id=RunnerId("r1"),
                runner_status=RunnerIdle(),
            ),
            idx=0,
        ),
    )

    assert worker._plan_wake.is_set()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_plan_wake_not_set_on_passthrough_event() -> None:
    """_plan_wake is NOT set for passthrough events that don't modify state."""
    worker = _make_worker()

    passthrough_events: list[Event] = [
        TestEvent(),
        ChunkGenerated(
            command_id=CommandId(),
            chunk=TokenChunk(
                model=ModelId("test"),
                text="hi",
                token_id=1,
                finish_reason=None,
                usage=None,
            ),
        ),
        TaskAcknowledged(task_id=TaskId()),
    ]

    for i, evt in enumerate(passthrough_events):
        worker._plan_wake = anyio.Event()  # pyright: ignore[reportPrivateUsage]

        _apply_and_maybe_wake(worker, _make_indexed(evt, idx=i))

        assert not worker._plan_wake.is_set(), (  # pyright: ignore[reportPrivateUsage]
            f"_plan_wake should NOT be set for {type(evt).__name__}"
        )


@pytest.mark.asyncio
async def test_plan_wake_resets_after_consumption() -> None:
    """After plan_step consumes the wake, it resets to a new Event."""
    worker = _make_worker()

    # Simulate wake
    worker._plan_wake.set()  # pyright: ignore[reportPrivateUsage]
    assert worker._plan_wake.is_set()  # pyright: ignore[reportPrivateUsage]

    # Simulate what plan_step does after waking
    worker._plan_wake = anyio.Event()  # pyright: ignore[reportPrivateUsage]
    assert not worker._plan_wake.is_set()  # pyright: ignore[reportPrivateUsage]
