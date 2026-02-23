from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.tasks import DisaggDecode, DisaggPrefill, TaskId, TaskStatus
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.instances import (
    DisaggregatedInstance,
    InstanceId,
)
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata


def _make_task_params() -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("test-model"),
        input=[InputMessage(role="user", content="hello")],
    )


def _make_model_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_mb(100),
        n_layers=32,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )


def _make_disaggregated_instance() -> DisaggregatedInstance:
    model_card = _make_model_card()
    prefill_runner = RunnerId()
    decode_runner = RunnerId()
    prefill_node = NodeId("prefill-node")
    decode_node = NodeId("decode-node")

    full_shard = PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=32,
        n_layers=32,
    )

    return DisaggregatedInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=model_card.model_id,
            runner_to_shard={prefill_runner: full_shard, decode_runner: full_shard},
            node_to_runner={prefill_node: prefill_runner, decode_node: decode_runner},
        ),
        prefill_node_id=prefill_node,
        decode_node_id=decode_node,
        decode_node_host="10.0.0.2",
    )


def test_disagg_prefill_task_roundtrip():
    """DisaggPrefill serializes/deserializes via model_dump_json/model_validate_json."""
    task = DisaggPrefill(
        task_id=TaskId(),
        instance_id=InstanceId(),
        task_status=TaskStatus.Pending,
        command_id=CommandId(),
        task_params=_make_task_params(),
        decode_node_host="10.0.0.2",
        decode_node_port=52416,
    )

    json_repr = task.model_dump_json()
    restored = DisaggPrefill.model_validate_json(json_repr)
    assert restored == task
    assert restored.decode_node_host == "10.0.0.2"
    assert restored.decode_node_port == 52416


def test_disagg_decode_task_roundtrip():
    """DisaggDecode serializes/deserializes via model_dump_json/model_validate_json."""
    task = DisaggDecode(
        task_id=TaskId(),
        instance_id=InstanceId(),
        task_status=TaskStatus.Pending,
        command_id=CommandId(),
        task_params=_make_task_params(),
        kv_transfer_port=52416,
    )

    json_repr = task.model_dump_json()
    restored = DisaggDecode.model_validate_json(json_repr)
    assert restored == task
    assert restored.kv_transfer_port == 52416


def test_disaggregated_instance_roundtrip():
    """DisaggregatedInstance serializes/deserializes via model_dump_json/model_validate_json."""
    instance = _make_disaggregated_instance()

    json_repr = instance.model_dump_json()
    restored = DisaggregatedInstance.model_validate_json(json_repr)
    assert restored == instance
    assert restored.prefill_node_id == instance.prefill_node_id
    assert restored.decode_node_id == instance.decode_node_id
    assert restored.decode_node_host == "10.0.0.2"


def test_state_with_disagg_instance_roundtrip():
    """State containing a DisaggregatedInstance round-trips through JSON."""
    instance = _make_disaggregated_instance()

    state = State(instances={instance.instance_id: instance})

    json_repr = state.model_dump_json()
    restored_state = State.model_validate_json(json_repr)

    assert instance.instance_id in restored_state.instances
    restored_instance = restored_state.instances[instance.instance_id]
    assert isinstance(restored_instance, DisaggregatedInstance)
    assert restored_instance.prefill_node_id == instance.prefill_node_id
    assert restored_instance.decode_node_id == instance.decode_node_id
    assert restored_instance.decode_node_host == instance.decode_node_host
    assert restored_state.model_dump_json() == json_repr
