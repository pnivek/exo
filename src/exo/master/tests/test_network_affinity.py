from exo.master.network_affinity import (
    find_nccl_coordinator_ip,
    nodes_share_subnet,
    pick_best_connected_node,
)
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import (
    InterfaceType,
    NetworkInterfaceInfo,
    NodeNetworkInfo,
)


def _iface(
    name: str,
    ip: str,
    netmask: str | None = None,
    rdma: str | None = None,
    iface_type: InterfaceType = "ethernet",
) -> NetworkInterfaceInfo:
    return NetworkInterfaceInfo(
        name=name,
        ip_address=ip,
        interface_type=iface_type,
        netmask=netmask,
        rdma_device_name=rdma,
    )


# --- nodes_share_subnet ---


def test_nodes_share_subnet_same_subnet() -> None:
    a = NodeId("a")
    b = NodeId("b")
    network = {
        a: NodeNetworkInfo(interfaces=[_iface("eth0", "192.168.0.1", "255.255.255.0")]),
        b: NodeNetworkInfo(interfaces=[_iface("eth0", "192.168.0.2", "255.255.255.0")]),
    }
    assert nodes_share_subnet(a, b, network) is True


def test_nodes_share_subnet_different_subnet() -> None:
    a = NodeId("a")
    b = NodeId("b")
    network = {
        a: NodeNetworkInfo(interfaces=[_iface("eth0", "10.0.0.1", "255.255.255.0")]),
        b: NodeNetworkInfo(interfaces=[_iface("eth0", "192.168.0.1", "255.255.255.0")]),
    }
    assert nodes_share_subnet(a, b, network) is False


def test_nodes_share_subnet_no_netmask() -> None:
    a = NodeId("a")
    b = NodeId("b")
    network = {
        a: NodeNetworkInfo(interfaces=[_iface("eth0", "192.168.0.1")]),
        b: NodeNetworkInfo(interfaces=[_iface("eth0", "192.168.0.2")]),
    }
    assert nodes_share_subnet(a, b, network) is False


def test_nodes_share_subnet_missing_node() -> None:
    a = NodeId("a")
    b = NodeId("b")
    network: dict[NodeId, NodeNetworkInfo] = {
        a: NodeNetworkInfo(interfaces=[_iface("eth0", "192.168.0.1", "255.255.255.0")]),
    }
    assert nodes_share_subnet(a, b, network) is False


# --- pick_best_connected_node ---


def test_pick_best_connected_node_prefers_shared_subnet() -> None:
    sparkly = NodeId("sparkly")
    spark1 = NodeId("spark1")
    mac = NodeId("mac")
    network = {
        sparkly: NodeNetworkInfo(
            interfaces=[_iface("eth0", "192.168.0.101", "255.255.255.0")]
        ),
        spark1: NodeNetworkInfo(
            interfaces=[_iface("rdma0", "10.0.0.2", "255.255.255.0")]
        ),
        mac: NodeNetworkInfo(
            interfaces=[_iface("en0", "192.168.0.156", "255.255.255.0")]
        ),
    }
    result = pick_best_connected_node([spark1, sparkly], mac, network)
    assert result == sparkly


def test_pick_best_connected_node_fallback_no_match() -> None:
    a = NodeId("a")
    b = NodeId("b")
    target = NodeId("target")
    network = {
        a: NodeNetworkInfo(interfaces=[_iface("eth0", "10.0.0.1", "255.255.255.0")]),
        b: NodeNetworkInfo(interfaces=[_iface("eth0", "10.0.1.1", "255.255.255.0")]),
        target: NodeNetworkInfo(
            interfaces=[_iface("eth0", "172.16.0.1", "255.255.255.0")]
        ),
    }
    result = pick_best_connected_node([a, b], target, network)
    assert result == a  # fallback to first candidate


def test_pick_best_connected_node_single_candidate() -> None:
    only = NodeId("only")
    target = NodeId("target")
    network: dict[NodeId, NodeNetworkInfo] = {}
    result = pick_best_connected_node([only], target, network)
    assert result == only


# --- find_nccl_coordinator_ip ---


def test_find_nccl_coordinator_ip_shared_subnet() -> None:
    """Strategy 1: IP sharing subnet with all other prefill nodes."""
    coord = NodeId("coord")
    other = NodeId("other")
    network = {
        coord: NodeNetworkInfo(
            interfaces=[
                _iface("eth0", "192.168.0.101", "255.255.255.0"),
                _iface("rdma0", "10.0.0.1", "255.255.255.0"),
            ]
        ),
        other: NodeNetworkInfo(
            interfaces=[_iface("rdma0", "10.0.0.2", "255.255.255.0")]
        ),
    }
    result = find_nccl_coordinator_ip(coord, [coord, other], network)
    # 192.168.0.101 doesn't share subnet with other (10.0.0.x)
    # 10.0.0.1 shares subnet with other (10.0.0.2) → picked
    assert result == "10.0.0.1"


def test_find_nccl_coordinator_ip_rdma_fallback() -> None:
    """Strategy 2: RDMA-associated IP when no subnet match."""
    coord = NodeId("coord")
    other = NodeId("other")
    network = {
        coord: NodeNetworkInfo(
            interfaces=[
                _iface("eth0", "192.168.0.101"),  # no netmask
                _iface("rdma0", "10.0.0.1", rdma="mlx5_0"),  # RDMA, no netmask
            ]
        ),
        other: NodeNetworkInfo(interfaces=[_iface("rdma0", "10.0.0.2")]),
    }
    result = find_nccl_coordinator_ip(coord, [coord, other], network)
    assert result == "10.0.0.1"


def test_find_nccl_coordinator_ip_generic_fallback() -> None:
    """Strategy 3: first non-loopback IP."""
    coord = NodeId("coord")
    network = {
        coord: NodeNetworkInfo(interfaces=[_iface("eth0", "192.168.0.101")]),
    }
    result = find_nccl_coordinator_ip(coord, [coord], network)
    assert result == "192.168.0.101"


def test_pick_best_connected_node_prefers_ethernet_over_wifi() -> None:
    """Ethernet↔ethernet should beat wifi↔wifi when both share a subnet with target."""
    sparkly = NodeId("sparkly")
    sparky = NodeId("sparky")
    mac = NodeId("mac")
    network = {
        sparkly: NodeNetworkInfo(
            interfaces=[
                _iface(
                    "enP7s7", "192.168.0.101", "255.255.255.0", iface_type="ethernet"
                ),
                _iface(
                    "enp1s0f0np0",
                    "169.254.144.33",
                    "255.255.0.0",
                    iface_type="ethernet",
                ),
                _iface("wlP9s9", "192.168.0.172", "255.255.255.0", iface_type="wifi"),
            ]
        ),
        sparky: NodeNetworkInfo(
            interfaces=[
                _iface("wlP9s9", "192.168.0.112", "255.255.255.0", iface_type="wifi"),
                _iface(
                    "enp1s0f1np1",
                    "169.254.249.25",
                    "255.255.0.0",
                    iface_type="ethernet",
                ),
            ]
        ),
        mac: NodeNetworkInfo(
            interfaces=[
                _iface("en0", "192.168.0.156", "255.255.255.0", iface_type="ethernet"),
                _iface("en1", "192.168.0.100", "255.255.255.0", iface_type="wifi"),
            ]
        ),
    }
    # Sparkly has ethernet↔ethernet with Mac (score 3), sparky only has wifi↔wifi (score 1)
    result = pick_best_connected_node([sparky, sparkly], mac, network)
    assert result == sparkly
    # Order shouldn't matter
    result2 = pick_best_connected_node([sparkly, sparky], mac, network)
    assert result2 == sparkly


def test_find_nccl_coordinator_ip_skips_loopback() -> None:
    coord = NodeId("coord")
    network = {
        coord: NodeNetworkInfo(
            interfaces=[
                _iface("lo", "127.0.0.1"),
                _iface("eth0", "192.168.0.5"),
            ]
        ),
    }
    result = find_nccl_coordinator_ip(coord, [coord], network)
    assert result == "192.168.0.5"
