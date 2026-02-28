import exo.worker.plan as plan_mod
from exo.shared.types.tasks import (
    DisaggDecode,
    DisaggPrefill,
    TaskStatus,
)
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import (
    RunnerIdle,
    RunnerReady,
)
from exo.worker.tests.constants import (
    COMMAND_1_ID,
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
    TASK_1_ID,
    TASK_2_ID,
)
from exo.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_disaggregated_instance,
)

TASK_PARAMS = TextGenerationTaskParams(
    model=MODEL_A_ID, input=[InputMessage(role="user", content="hello")]
)


def test_plan_forwards_disagg_prefill_to_prefill_runner():
    """DisaggPrefill task forwarded when prefill runner is ready and this node is the prefill node."""
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
        bound_instance=bound_instance, status=RunnerReady()
    )

    task = DisaggPrefill(
        task_id=TASK_1_ID,
        instance_id=INSTANCE_1_ID,
        task_status=TaskStatus.Pending,
        command_id=COMMAND_1_ID,
        task_params=TASK_PARAMS,
        decode_node_host="192.168.1.100",
    )

    result = plan_mod.plan(
        node_id=NODE_A,
        runners={RUNNER_1_ID: local_runner},  # type: ignore
        global_download_status={NODE_A: []},
        instances={INSTANCE_1_ID: instance},
        all_runners={RUNNER_1_ID: RunnerReady(), RUNNER_2_ID: RunnerReady()},
        tasks={TASK_1_ID: task},
    )

    assert result is task


def test_plan_forwards_disagg_decode_to_decode_runner():
    """DisaggDecode task forwarded when decode runner is ready and this node is the decode node."""
    instance = get_disaggregated_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        prefill_node_id=NODE_A,
        decode_node_id=NODE_B,
        prefill_runner_id=RUNNER_1_ID,
        decode_runner_id=RUNNER_2_ID,
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_2_ID, bound_node_id=NODE_B
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerReady()
    )

    task = DisaggDecode(
        task_id=TASK_1_ID,
        instance_id=INSTANCE_1_ID,
        task_status=TaskStatus.Pending,
        command_id=COMMAND_1_ID,
        task_params=TASK_PARAMS,
    )

    result = plan_mod.plan(
        node_id=NODE_B,
        runners={RUNNER_2_ID: local_runner},  # type: ignore
        global_download_status={NODE_B: []},
        instances={INSTANCE_1_ID: instance},
        all_runners={RUNNER_1_ID: RunnerReady(), RUNNER_2_ID: RunnerReady()},
        tasks={TASK_1_ID: task},
    )

    assert result is task


def test_plan_does_not_forward_disagg_prefill_to_decode_runner():
    """Prefill task NOT forwarded when local runner is the decode runner."""
    instance = get_disaggregated_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        prefill_node_id=NODE_A,
        decode_node_id=NODE_B,
        prefill_runner_id=RUNNER_1_ID,
        decode_runner_id=RUNNER_2_ID,
    )
    # We are NODE_B (decode node), but the task is DisaggPrefill
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_2_ID, bound_node_id=NODE_B
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerReady()
    )

    task = DisaggPrefill(
        task_id=TASK_1_ID,
        instance_id=INSTANCE_1_ID,
        task_status=TaskStatus.Pending,
        command_id=COMMAND_1_ID,
        task_params=TASK_PARAMS,
        decode_node_host="192.168.1.100",
    )

    result = plan_mod.plan(
        node_id=NODE_B,
        runners={RUNNER_2_ID: local_runner},  # type: ignore
        global_download_status={NODE_B: []},
        instances={INSTANCE_1_ID: instance},
        all_runners={RUNNER_1_ID: RunnerReady(), RUNNER_2_ID: RunnerReady()},
        tasks={TASK_1_ID: task},
    )

    assert result is None


def test_plan_does_not_forward_disagg_decode_to_prefill_runner():
    """Decode task NOT forwarded when local runner is the prefill runner."""
    instance = get_disaggregated_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        prefill_node_id=NODE_A,
        decode_node_id=NODE_B,
        prefill_runner_id=RUNNER_1_ID,
        decode_runner_id=RUNNER_2_ID,
    )
    # We are NODE_A (prefill node), but the task is DisaggDecode
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    local_runner = FakeRunnerSupervisor(
        bound_instance=bound_instance, status=RunnerReady()
    )

    task = DisaggDecode(
        task_id=TASK_1_ID,
        instance_id=INSTANCE_1_ID,
        task_status=TaskStatus.Pending,
        command_id=COMMAND_1_ID,
        task_params=TASK_PARAMS,
    )

    result = plan_mod.plan(
        node_id=NODE_A,
        runners={RUNNER_1_ID: local_runner},  # type: ignore
        global_download_status={NODE_A: []},
        instances={INSTANCE_1_ID: instance},
        all_runners={RUNNER_1_ID: RunnerReady(), RUNNER_2_ID: RunnerReady()},
        tasks={TASK_1_ID: task},
    )

    assert result is None


def test_disagg_requires_both_runners_ready():
    """Neither task forwarded until both runners are RunnerReady."""
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
        bound_instance=bound_instance, status=RunnerReady()
    )

    prefill_task = DisaggPrefill(
        task_id=TASK_1_ID,
        instance_id=INSTANCE_1_ID,
        task_status=TaskStatus.Pending,
        command_id=COMMAND_1_ID,
        task_params=TASK_PARAMS,
        decode_node_host="192.168.1.100",
    )
    decode_task = DisaggDecode(
        task_id=TASK_2_ID,
        instance_id=INSTANCE_1_ID,
        task_status=TaskStatus.Pending,
        command_id=COMMAND_1_ID,
        task_params=TASK_PARAMS,
    )

    # RUNNER_2 (decode) is still Idle
    result = plan_mod.plan(
        node_id=NODE_A,
        runners={RUNNER_1_ID: local_runner},  # type: ignore
        global_download_status={NODE_A: []},
        instances={INSTANCE_1_ID: instance},
        all_runners={RUNNER_1_ID: RunnerReady(), RUNNER_2_ID: RunnerIdle()},
        tasks={TASK_1_ID: prefill_task, TASK_2_ID: decode_task},
    )

    assert result is None
