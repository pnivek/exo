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


def _best_shared_link_score(
    node_a: NodeId,
    node_b: NodeId,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> int:
    """Score the best shared link between two nodes.

    Returns:
      3 = ethernet↔ethernet or thunderbolt (both wired)
      2 = one side wired, neither WiFi
      1 = any subnet match (wifi, unknown, etc.)
      0 = no shared subnet
    """
    net_a = node_network.get(node_a, NodeNetworkInfo())
    net_b = node_network.get(node_b, NodeNetworkInfo())
    best = 0

    for iface_a in net_a.interfaces:
        network_a = _interface_network(iface_a)
        if network_a is None:
            continue
        for iface_b in net_b.interfaces:
            network_b = _interface_network(iface_b)
            if network_b is None:
                continue
            if not network_a.overlaps(network_b):
                continue

            types = {iface_a.interface_type, iface_b.interface_type}
            if types <= {"ethernet", "thunderbolt", "maybe_ethernet"}:
                score = 3
            elif "wifi" not in types and (
                "ethernet" in types or "thunderbolt" in types
            ):
                score = 2
            else:
                score = 1
            best = max(best, score)

    return best


def pick_best_connected_node(
    candidates: Sequence[NodeId],
    target_node: NodeId,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> NodeId:
    """From candidates, pick the one with best network path to target_node.

    Priority:
    1. Wired (ethernet/thunderbolt) shared subnet with target (10GbE, direct cable)
    2. Any shared subnet with target (WiFi, etc.)
    3. Falls back to first candidate
    """
    best_candidate = candidates[0]
    best_score = 0

    for candidate in candidates:
        score = _best_shared_link_score(candidate, target_node, node_network)
        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


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


def _is_reachable_by_all(
    iface: NetworkInterfaceInfo,
    other_nodes: Sequence[NodeId],
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> bool:
    """Check if an interface shares a subnet with at least one interface on every other node."""
    coord_net = _interface_network(iface)
    if coord_net is None:
        return False
    for other_node in other_nodes:
        other_network = node_network.get(other_node, NodeNetworkInfo())
        reachable = any(
            (other_net := _interface_network(other_iface)) is not None
            and coord_net.overlaps(other_net)
            for other_iface in other_network.interfaces
        )
        if not reachable:
            return False
    return True


def find_relay_ip(
    sender_node: NodeId,
    all_prefill_nodes: Sequence[NodeId],
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> str | None:
    """Find sender's IP on the fastest link reachable by other prefill nodes.

    Unlike find_nccl_coordinator_ip which returns the first subnet match,
    this function prioritizes high-bandwidth interfaces (RDMA, link-local)
    over regular ethernet/WiFi.

    Priority:
    1. RDMA interface IP sharing subnet with all other prefill nodes
    2. Link-local (169.254.x.x) IP sharing subnet with all other prefill nodes
    3. Any non-loopback IP sharing subnet with all other prefill nodes
    """
    sender_network = node_network.get(sender_node, NodeNetworkInfo())
    others = [n for n in all_prefill_nodes if n != sender_node]

    # Strategy 1: RDMA interface reachable by all
    for iface in sender_network.interfaces:
        if iface.rdma_device_name is None:
            continue
        if iface.ip_address in ("127.0.0.1", "::1"):
            continue
        if _is_reachable_by_all(iface, others, node_network):
            return iface.ip_address

    # Strategy 2: link-local (169.254.x.x) reachable by all
    for iface in sender_network.interfaces:
        if not iface.ip_address.startswith("169.254."):
            continue
        if _is_reachable_by_all(iface, others, node_network):
            return iface.ip_address

    # Strategy 3: any non-loopback IP reachable by all
    for iface in sender_network.interfaces:
        if iface.ip_address in ("127.0.0.1", "::1") or iface.ip_address.startswith(
            "fe80:"
        ):
            continue
        if _is_reachable_by_all(iface, others, node_network):
            return iface.ip_address

    return None
