"""Tests for disaggregated task pairing via paired_task_id."""

from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.tasks import (
    DisaggDecode,
    DisaggPrefill,
    TaskId,
    TensorParallelDisaggPrefill,
)
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.instances import InstanceId

INSTANCE_ID = InstanceId("test-instance")
CMD_ID = CommandId()
TASK_PARAMS = TextGenerationTaskParams(
    model=ModelId("test-model"),
    input=[InputMessage(role="user", content="Hello")],
)


def test_disagg_prefill_accepts_paired_task_id() -> None:
    """DisaggPrefill can be created with paired_task_id."""
    decode_id = TaskId()
    task = DisaggPrefill(
        instance_id=INSTANCE_ID,
        command_id=CMD_ID,
        task_params=TASK_PARAMS,
        decode_node_host="192.168.1.100",
        paired_task_id=decode_id,
    )
    assert task.paired_task_id == decode_id


def test_disagg_decode_accepts_paired_task_id() -> None:
    """DisaggDecode can be created with paired_task_id."""
    prefill_id = TaskId()
    task = DisaggDecode(
        instance_id=INSTANCE_ID,
        command_id=CMD_ID,
        task_params=TASK_PARAMS,
        paired_task_id=prefill_id,
    )
    assert task.paired_task_id == prefill_id


def test_tp_disagg_prefill_accepts_paired_task_id() -> None:
    """TensorParallelDisaggPrefill can be created with paired_task_id."""
    decode_id = TaskId()
    task = TensorParallelDisaggPrefill(
        instance_id=INSTANCE_ID,
        command_id=CMD_ID,
        task_params=TASK_PARAMS,
        decode_node_host="192.168.1.100",
        paired_task_id=decode_id,
    )
    assert task.paired_task_id == decode_id


def test_paired_task_id_defaults_to_none() -> None:
    """paired_task_id defaults to None for backwards compatibility."""
    task = DisaggPrefill(
        instance_id=INSTANCE_ID,
        command_id=CMD_ID,
        task_params=TASK_PARAMS,
        decode_node_host="192.168.1.100",
    )
    assert task.paired_task_id is None


def test_paired_tasks_cross_linked() -> None:
    """Prefill and decode tasks can be cross-linked via paired_task_id."""
    prefill_id = TaskId()
    decode_id = TaskId()

    prefill = DisaggPrefill(
        task_id=prefill_id,
        instance_id=INSTANCE_ID,
        command_id=CMD_ID,
        task_params=TASK_PARAMS,
        decode_node_host="192.168.1.100",
        paired_task_id=decode_id,
    )
    decode = DisaggDecode(
        task_id=decode_id,
        instance_id=INSTANCE_ID,
        command_id=CMD_ID,
        task_params=TASK_PARAMS,
        paired_task_id=prefill_id,
    )

    assert prefill.paired_task_id == decode.task_id
    assert decode.paired_task_id == prefill.task_id


def test_paired_task_id_survives_serialization() -> None:
    """paired_task_id round-trips through TaggedModel serialization."""
    decode_id = TaskId()
    original = DisaggPrefill(
        instance_id=INSTANCE_ID,
        command_id=CMD_ID,
        task_params=TASK_PARAMS,
        decode_node_host="192.168.1.100",
        paired_task_id=decode_id,
    )

    serialized = original.model_dump()
    deserialized = DisaggPrefill.model_validate(serialized)

    assert deserialized.paired_task_id == decode_id


def test_paired_task_id_none_survives_serialization() -> None:
    """paired_task_id=None round-trips correctly (backwards compat)."""
    original = DisaggDecode(
        instance_id=INSTANCE_ID,
        command_id=CMD_ID,
        task_params=TASK_PARAMS,
    )

    serialized = original.model_dump()
    deserialized = DisaggDecode.model_validate(serialized)

    assert deserialized.paired_task_id is None
