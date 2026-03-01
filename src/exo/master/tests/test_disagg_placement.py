import pytest

from exo.master.placement import place_disaggregated_instance
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    NetworkInterfaceInfo,
    NodeIdentity,
    NodeNetworkInfo,
)
from exo.shared.types.worker.instances import DisaggregatedInstance


@pytest.fixture
def model_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_mb(1000),
        n_layers=32,
        hidden_size=2048,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )


@pytest.fixture
def prefill_node_id() -> NodeId:
    return NodeId("prefill-node")


@pytest.fixture
def decode_node_id() -> NodeId:
    return NodeId("decode-node")


@pytest.fixture
def node_identities(
    prefill_node_id: NodeId, decode_node_id: NodeId
) -> dict[NodeId, NodeIdentity]:
    return {
        prefill_node_id: NodeIdentity(chip_id="dgx-spark"),
        decode_node_id: NodeIdentity(chip_id="Apple M4 Max"),
    }


@pytest.fixture
def node_network(
    prefill_node_id: NodeId, decode_node_id: NodeId
) -> dict[NodeId, NodeNetworkInfo]:
    return {
        prefill_node_id: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="eth0", ip_address="10.0.0.1", interface_type="ethernet"
                )
            ]
        ),
        decode_node_id: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0", ip_address="10.0.0.2", interface_type="ethernet"
                ),
                NetworkInterfaceInfo(
                    name="en1", ip_address="192.168.1.5", interface_type="wifi"
                ),
            ]
        ),
    }


def test_place_disaggregated_identifies_nodes(
    model_card: ModelCard,
    prefill_node_id: NodeId,
    decode_node_id: NodeId,
    node_identities: dict[NodeId, NodeIdentity],
    node_network: dict[NodeId, NodeNetworkInfo],
):
    """Correctly identifies prefill (chip 'dgx-spark') and decode (chip 'Apple M4 Max') nodes."""
    placements = place_disaggregated_instance(
        model_card=model_card,
        current_instances={},
        node_identities=node_identities,
        node_network=node_network,
    )

    assert len(placements) == 1
    instance = list(placements.values())[0]
    assert isinstance(instance, DisaggregatedInstance)
    assert instance.prefill_node_id == prefill_node_id
    assert instance.decode_node_id == decode_node_id


def test_place_disaggregated_full_model_shards(
    model_card: ModelCard,
    node_identities: dict[NodeId, NodeIdentity],
    node_network: dict[NodeId, NodeNetworkInfo],
):
    """Both runners get start_layer=0, end_layer=n_layers, world_size=1."""
    placements = place_disaggregated_instance(
        model_card=model_card,
        current_instances={},
        node_identities=node_identities,
        node_network=node_network,
    )

    instance = list(placements.values())[0]
    assert isinstance(instance, DisaggregatedInstance)

    for shard in instance.shard_assignments.runner_to_shard.values():
        assert shard.start_layer == 0
        assert shard.end_layer == model_card.n_layers
        assert shard.world_size == 1
        assert shard.device_rank == 0


def test_place_disaggregated_finds_decode_ip(
    model_card: ModelCard,
    node_identities: dict[NodeId, NodeIdentity],
    node_network: dict[NodeId, NodeNetworkInfo],
):
    """Extracts decode node IP from network interfaces, preferring ethernet."""
    placements = place_disaggregated_instance(
        model_card=model_card,
        current_instances={},
        node_identities=node_identities,
        node_network=node_network,
    )

    instance = list(placements.values())[0]
    assert isinstance(instance, DisaggregatedInstance)
    # Should pick ethernet (10.0.0.2) over wifi (192.168.1.5)
    assert instance.decode_node_host == "10.0.0.2"


def test_place_disaggregated_raises_if_no_prefill_node(
    model_card: ModelCard,
    decode_node_id: NodeId,
    node_network: dict[NodeId, NodeNetworkInfo],
):
    """ValueError when no CUDA/DGX node found."""
    node_identities = {
        decode_node_id: NodeIdentity(chip_id="Apple M4 Max"),
    }

    with pytest.raises(ValueError, match="No prefill node"):
        place_disaggregated_instance(
            model_card=model_card,
            current_instances={},
            node_identities=node_identities,
            node_network=node_network,
        )


def test_place_disaggregated_raises_if_no_decode_node(
    model_card: ModelCard,
    prefill_node_id: NodeId,
    node_network: dict[NodeId, NodeNetworkInfo],
):
    """ValueError when no Apple Silicon node found."""
    node_identities = {
        prefill_node_id: NodeIdentity(chip_id="dgx-spark"),
    }

    with pytest.raises(ValueError, match="No decode node"):
        place_disaggregated_instance(
            model_card=model_card,
            current_instances={},
            node_identities=node_identities,
            node_network=node_network,
        )


def test_place_disaggregated_prefers_subnet_connected_prefill(
    model_card: ModelCard,
):
    """When 2 NVIDIA nodes exist, picks the one sharing a subnet with the decode node."""
    sparkly = NodeId("sparkly")
    spark1 = NodeId("spark1")
    mac = NodeId("mac")

    node_identities = {
        sparkly: NodeIdentity(chip_id="dgx-spark"),
        spark1: NodeIdentity(chip_id="dgx-spark"),
        mac: NodeIdentity(chip_id="Apple M2 Ultra"),
    }
    node_network = {
        sparkly: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="eth0",
                    ip_address="192.168.0.101",
                    interface_type="ethernet",
                    netmask="255.255.255.0",
                ),
            ]
        ),
        spark1: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="enp1s0f0",
                    ip_address="10.0.0.2",
                    interface_type="ethernet",
                    netmask="255.255.255.0",
                ),
            ]
        ),
        mac: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0",
                    ip_address="192.168.0.156",
                    interface_type="ethernet",
                    netmask="255.255.255.0",
                ),
            ]
        ),
    }

    placements = place_disaggregated_instance(
        model_card=model_card,
        current_instances={},
        node_identities=node_identities,
        node_network=node_network,
    )

    instance = list(placements.values())[0]
    assert isinstance(instance, DisaggregatedInstance)
    # Sparkly shares 192.168.0.x/24 with Mac; Spark1 is on 10.0.0.x/24
    assert instance.prefill_node_id == sparkly
    assert instance.decode_node_id == mac


def test_place_disaggregated_fallback_no_netmask(
    model_card: ModelCard,
):
    """Without netmask data, falls back to first candidate (backward compat)."""
    sparkly = NodeId("sparkly")
    spark1 = NodeId("spark1")
    mac = NodeId("mac")

    node_identities = {
        sparkly: NodeIdentity(chip_id="dgx-spark"),
        spark1: NodeIdentity(chip_id="dgx-spark"),
        mac: NodeIdentity(chip_id="Apple M2 Ultra"),
    }
    # No netmask on any interface
    node_network = {
        sparkly: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="eth0", ip_address="192.168.0.101", interface_type="ethernet"
                ),
            ]
        ),
        spark1: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="enp1s0f0", ip_address="10.0.0.2", interface_type="ethernet"
                ),
            ]
        ),
        mac: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0", ip_address="192.168.0.156", interface_type="ethernet"
                ),
            ]
        ),
    }

    placements = place_disaggregated_instance(
        model_card=model_card,
        current_instances={},
        node_identities=node_identities,
        node_network=node_network,
    )

    instance = list(placements.values())[0]
    assert isinstance(instance, DisaggregatedInstance)
    # Fallback: picks first candidate (no subnet data to differentiate)
    assert instance.prefill_node_id in (sparkly, spark1)
