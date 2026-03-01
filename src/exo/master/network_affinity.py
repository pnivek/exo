"""Network affinity helpers for placement decisions.

Pure functions that determine network proximity between nodes
based on shared subnets. Used by placement logic to prefer nodes
with direct IP connectivity for KV transfer and NCCL coordination.
"""

import ipaddress
from collections.abc import Mapping, Sequence

from exo.shared.types.common import NodeId
from exo.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo


def _interface_network(
    iface: NetworkInterfaceInfo,
) -> ipaddress.IPv4Network | None:
    """Compute the subnet for an interface, or None if netmask is missing."""
    if iface.netmask is None:
        return None
    try:
        return ipaddress.IPv4Network(
            f"{iface.ip_address}/{iface.netmask}", strict=False
        )
    except (ValueError, ipaddress.AddressValueError):
        return None


def nodes_share_subnet(
    node_a: NodeId,
    node_b: NodeId,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> bool:
    """Return True if any interface on node_a shares a subnet with any on node_b."""
    net_a = node_network.get(node_a, NodeNetworkInfo())
    net_b = node_network.get(node_b, NodeNetworkInfo())

    for iface_a in net_a.interfaces:
        network_a = _interface_network(iface_a)
        if network_a is None:
            continue
        for iface_b in net_b.interfaces:
            network_b = _interface_network(iface_b)
            if network_b is None:
                continue
            if network_a.overlaps(network_b):
                return True
    return False


def pick_best_connected_node(
    candidates: Sequence[NodeId],
    target_node: NodeId,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> NodeId:
    """From candidates, pick the one with best network path to target_node.

    Priority:
    1. Shares a subnet with target_node (direct IP connectivity)
    2. Falls back to first candidate (preserves current behavior)
    """
    for candidate in candidates:
        if nodes_share_subnet(candidate, target_node, node_network):
            return candidate
    return candidates[0]


def find_nccl_coordinator_ip(
    coordinator_node: NodeId,
    all_prefill_nodes: Sequence[NodeId],
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> str | None:
    """Find coordinator IP reachable by all prefill nodes.

    Priority:
    1. IP on coordinator that shares subnet with ALL other prefill nodes
    2. RDMA-associated IP on coordinator (prefill nodes connect via RDMA)
    3. First non-loopback IP (current fallback behavior)
    """
    coordinator_network = node_network.get(coordinator_node, NodeNetworkInfo())
    other_prefill = [n for n in all_prefill_nodes if n != coordinator_node]

    # Strategy 1: IP sharing subnet with all other prefill nodes
    for iface in coordinator_network.interfaces:
        if iface.ip_address in ("127.0.0.1", "::1") or iface.ip_address.startswith(
            "fe80:"
        ):
            continue
        coord_net = _interface_network(iface)
        if coord_net is None:
            continue

        reachable_by_all = True
        for other_node in other_prefill:
            other_network = node_network.get(other_node, NodeNetworkInfo())
            reachable = any(
                (other_net := _interface_network(other_iface)) is not None
                and coord_net.overlaps(other_net)
                for other_iface in other_network.interfaces
            )
            if not reachable:
                reachable_by_all = False
                break

        if reachable_by_all:
            return iface.ip_address

    # Strategy 2: prefer RDMA-associated IP
    for iface in coordinator_network.interfaces:
        if iface.rdma_device_name is not None and iface.ip_address not in (
            "127.0.0.1",
            "::1",
        ):
            return iface.ip_address

    # Strategy 3: first non-loopback IP (current behavior)
    for iface in coordinator_network.interfaces:
        if (
            iface.ip_address
            and iface.ip_address not in ("127.0.0.1", "::1")
            and not iface.ip_address.startswith("172.")
            and not iface.ip_address.startswith("fe80:")
        ):
            return iface.ip_address

    return None
