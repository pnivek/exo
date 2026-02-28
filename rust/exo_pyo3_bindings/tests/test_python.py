import asyncio

import pytest
from exo_pyo3_bindings import Keypair, NetworkingHandle, NoPeersSubscribedToTopicError


@pytest.mark.asyncio
async def test_sleep_on_multiple_items() -> None:
    print("PYTHON: starting handle")
    h = NetworkingHandle(Keypair.generate())

    rt = asyncio.create_task(_await_recv(h))

    # sleep for 4 ticks
    for i in range(4):
        await asyncio.sleep(1)

        try:
            await h.gossipsub_publish("topic", b"somehting or other")
        except NoPeersSubscribedToTopicError as e:
            print("caught it", e)


async def _await_recv(h: NetworkingHandle):
    while True:
        msg = await h.recv()
        print(f"PYTHON: received: {msg}")
