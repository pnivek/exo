import exo.worker.plan as plan_mod
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import LoadModel, StartWarmup
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerIdle,
    RunnerLoaded,
)
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
)
from exo.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_disaggregated_instance,
    get_pipeline_shard_metadata,
)


def test_disagg_skips_distributed_backend():
    """plan() does not emit ConnectToGroup for DisaggregatedInstance even when both runners are idle."""
    instance = get_disaggregated_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        prefill_node_id=NODE_A,
        decode_node_id=NODE_B,
        prefill_runner_id=RUNNER_1_ID,
        decode_runner_id=RUNNER_2_ID,
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerIdle()
    )

    # Both runners idle, no downloads yet — non-disagg would trigger ConnectToGroup
    result = plan_mod.plan(
        node_id=NODE_A,
        runners={RUNNER_1_ID: local_runner},  # type: ignore
        global_download_status={NODE_A: [], NODE_B: []},
        instances={INSTANCE_1_ID: instance},
        all_runners={RUNNER_1_ID: RunnerIdle(), RUNNER_2_ID: RunnerIdle()},
        tasks={},
    )

    # Should emit DownloadModel (not ConnectToGroup) since distributed init is skipped
    assert not isinstance(result, plan_mod.ConnectToGroup)


def test_disagg_loads_model_when_idle():
    """plan() triggers LoadModel for DisaggregatedInstance when runner is RunnerIdle and downloads are complete."""
    instance = get_disaggregated_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        prefill_node_id=NODE_A,
        decode_node_id=NODE_B,
        prefill_runner_id=RUNNER_1_ID,
        decode_runner_id=RUNNER_2_ID,
    )
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerIdle()
    )

    # Downloads complete for both nodes
    global_download_status = {
        NODE_A: [
            DownloadCompleted(shard_metadata=shard, node_id=NODE_A, total=Memory())
        ],
        NODE_B: [
            DownloadCompleted(shard_metadata=shard, node_id=NODE_B, total=Memory())
        ],
    }

    result = plan_mod.plan(
        node_id=NODE_A,
        runners={RUNNER_1_ID: local_runner},  # type: ignore
        global_download_status=global_download_status,
        instances={INSTANCE_1_ID: instance},
        all_runners={RUNNER_1_ID: RunnerIdle(), RUNNER_2_ID: RunnerIdle()},
        tasks={},
    )

    assert isinstance(result, LoadModel)
    assert result.instance_id == INSTANCE_1_ID


def test_disagg_warmup_independently():
    """plan() triggers StartWarmup when runner is RunnerLoaded, without waiting for peer."""
    instance = get_disaggregated_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        prefill_node_id=NODE_A,
        decode_node_id=NODE_B,
        prefill_runner_id=RUNNER_1_ID,
        decode_runner_id=RUNNER_2_ID,
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerLoaded()
    )

    # Peer is still idle — warmup should still trigger for disaggregated
    result = plan_mod.plan(
        node_id=NODE_A,
        runners={RUNNER_1_ID: local_runner},  # type: ignore
        global_download_status={NODE_A: []},
        instances={INSTANCE_1_ID: instance},
        all_runners={RUNNER_1_ID: RunnerLoaded(), RUNNER_2_ID: RunnerIdle()},
        tasks={},
    )

    assert isinstance(result, StartWarmup)
    assert result.instance_id == INSTANCE_1_ID
