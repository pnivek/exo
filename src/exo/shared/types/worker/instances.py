from enum import Enum

from pydantic import model_validator

from exo.shared.models.model_cards import ModelTask
from exo.shared.types.common import Host, Id, NodeId
from exo.shared.types.worker.runners import RunnerId, ShardAssignments, ShardMetadata
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


class InstanceId(Id):
    pass


class InstanceMeta(str, Enum):
    MlxRing = "MlxRing"
    MlxJaccl = "MlxJaccl"
    Disaggregated = "Disaggregated"
    TensorPrefillDisagg = "TensorPrefillDisagg"


class BaseInstance(TaggedModel):
    instance_id: InstanceId
    shard_assignments: ShardAssignments

    def shard(self, runner_id: RunnerId) -> ShardMetadata | None:
        return self.shard_assignments.runner_to_shard.get(runner_id, None)


class MlxRingInstance(BaseInstance):
    hosts_by_node: dict[NodeId, list[Host]]
    ephemeral_port: int


class MlxJacclInstance(BaseInstance):
    jaccl_devices: list[list[str | None]]
    jaccl_coordinators: dict[NodeId, str]


class DisaggregatedInstance(BaseInstance):
    """Instance for disaggregated prefill/decode across two nodes."""

    prefill_node_id: NodeId
    decode_node_id: NodeId
    decode_node_host: str
    kv_transfer_port: int = 52416


class TensorPrefillDisaggInstance(BaseInstance):
    """Instance for tensor-parallel prefill across multiple nodes + disaggregated decode.

    Multiple prefill nodes form a tensor-parallel group via NCCL.
    After prefill, rank 0 gathers the full KV cache and streams it to the
    decode node using the existing KVPS pipelined protocol.
    """

    prefill_node_ids: list[NodeId]
    nccl_host_ip: str
    nccl_port: int
    decode_node_id: NodeId
    decode_node_host: str
    kv_transfer_port: int = 52416
    kv_sender_node_id: NodeId
    kv_relay_host: str
    kv_relay_port: int


Instance = (
    MlxRingInstance
    | MlxJacclInstance
    | DisaggregatedInstance
    | TensorPrefillDisaggInstance
)


class BoundInstance(CamelCaseModel):
    instance: Instance
    bound_runner_id: RunnerId
    bound_node_id: NodeId

    @property
    def bound_shard(self) -> ShardMetadata:
        shard = self.instance.shard(self.bound_runner_id)
        assert shard is not None
        return shard

    @property
    def is_image_model(self) -> bool:
        return (
            ModelTask.TextToImage in self.bound_shard.model_card.tasks
            or ModelTask.ImageToImage in self.bound_shard.model_card.tasks
        )

    @model_validator(mode="after")
    def validate_shard_exists(self) -> "BoundInstance":
        assert (
            self.bound_runner_id in self.instance.shard_assignments.runner_to_shard
        ), (
            "Bound Instance must be constructed with a runner_id that is in the instances assigned shards"
        )
        return self
