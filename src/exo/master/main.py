import time
from datetime import datetime, timedelta, timezone

import anyio
from anyio import BrokenResourceError
from loguru import logger

from exo.master.event_log import DiskEventLog
from exo.master.placement import (
    add_instance_to_placements,
    cancel_unnecessary_downloads,
    delete_instance,
    get_transition_events,
    place_disaggregated_instance,
    place_instance,
    place_tensor_prefill_disagg_instance,
)
from exo.shared.apply import apply
from exo.shared.constants import EXO_EVENT_LOG_DIR, EXO_TRACING_ENABLED
from exo.shared.types.commands import (
    CreateInstance,
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageEdits,
    ImageGeneration,
    PlaceInstance,
    RequestEventLog,
    SendInputChunk,
    TaskCancelled,
    TaskFinished,
    TestCommand,
    TextGeneration,
)
from exo.shared.types.common import CommandId, NodeId, SessionId, SystemId
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    InputChunkReceived,
    InstanceDeleted,
    LocalForwarderEvent,
    NodeGatheredInfo,
    NodeTimedOut,
    TaskCreated,
    TaskDeleted,
    TaskFailed,
    TaskStatusUpdated,
    TraceEventData,
    TracesCollected,
    TracesMerged,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    DisaggDecode as DisaggDecodeTask,
)
from exo.shared.types.tasks import (
    DisaggPrefill as DisaggPrefillTask,
)
from exo.shared.types.tasks import (
    ImageEdits as ImageEditsTask,
)
from exo.shared.types.tasks import (
    ImageGeneration as ImageGenerationTask,
)
from exo.shared.types.tasks import (
    TaskId,
    TaskStatus,
)
from exo.shared.types.tasks import (
    TensorParallelDisaggPrefill as TensorParallelDisaggPrefillTask,
)
from exo.shared.types.tasks import (
    TextGeneration as TextGenerationTask,
)
from exo.shared.types.worker.instances import (
    DisaggregatedInstance,
    InstanceId,
    InstanceMeta,
    TensorPrefillDisaggInstance,
)
from exo.utils.channels import Receiver, Sender
from exo.utils.event_buffer import MultiSourceBuffer
from exo.utils.task_group import TaskGroup

# Task types that carry a paired_task_id for disaggregated inference
_PAIRED_TASK_TYPES = (
    DisaggPrefillTask,
    DisaggDecodeTask,
    TensorParallelDisaggPrefillTask,
)


class Master:
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        *,
        command_receiver: Receiver[ForwarderCommand],
        event_sender: Sender[Event],
        local_event_receiver: Receiver[LocalForwarderEvent],
        global_event_sender: Sender[GlobalForwarderEvent],
        download_command_sender: Sender[ForwarderDownloadCommand],
    ):
        self.node_id = node_id
        self.session_id = session_id
        self.state = State()
        self._tg: TaskGroup = TaskGroup()
        self.command_task_mapping: dict[CommandId, TaskId] = {}
        self.command_receiver = command_receiver
        self.local_event_receiver = local_event_receiver
        self.global_event_sender = global_event_sender
        self.download_command_sender = download_command_sender
        self.event_sender = event_sender
        self._system_id = SystemId()
        self._multi_buffer = MultiSourceBuffer[SystemId, Event]()
        self._event_log = DiskEventLog(EXO_EVENT_LOG_DIR / "master")
        self._pending_traces: dict[TaskId, dict[int, list[TraceEventData]]] = {}
        self._expected_ranks: dict[TaskId, set[int]] = {}
        # Deduplication: track last processed time per (node_id, info_type) to throttle NodeGatheredInfo
        self._last_gathered_info: dict[tuple[str, str], float] = {}
        self._gathered_info_cooldown: float = 5.0  # seconds

    async def run(self):
        logger.info("Starting Master")

        try:
            async with self._tg as tg:
                tg.start_soon(self._event_processor)
                tg.start_soon(self._command_processor)
                tg.start_soon(self._plan)
        finally:
            self._event_log.close()
            self.global_event_sender.close()
            self.local_event_receiver.close()
            self.command_receiver.close()

    async def shutdown(self):
        logger.info("Stopping Master")
        self._tg.cancel_tasks()

    async def _command_processor(self) -> None:
        with self.command_receiver as commands:
            async for forwarder_command in commands:
                try:
                    logger.info(f"Executing command: {forwarder_command.command}")

                    generated_events: list[Event] = []
                    command = forwarder_command.command
                    instance_task_counts: dict[InstanceId, int] = {}
                    match command:
                        case TestCommand():
                            pass
                        case TextGeneration():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            selected_instance = self.state.instances[
                                available_instance_ids[0]
                            ]

                            if isinstance(
                                selected_instance, TensorPrefillDisaggInstance
                            ):
                                # TP Disagg: create TP prefill task (dispatched to all prefill runners) + decode task
                                prefill_task_id = TaskId()
                                decode_task_id = TaskId()

                                generated_events.append(
                                    TaskCreated(
                                        task_id=prefill_task_id,
                                        task=TensorParallelDisaggPrefillTask(
                                            task_id=prefill_task_id,
                                            command_id=command.command_id,
                                            instance_id=selected_instance.instance_id,
                                            task_status=TaskStatus.Pending,
                                            task_params=command.task_params,
                                            decode_node_host=selected_instance.decode_node_host,
                                            decode_node_port=selected_instance.kv_transfer_port,
                                            paired_task_id=decode_task_id,
                                        ),
                                    )
                                )
                                generated_events.append(
                                    TaskCreated(
                                        task_id=decode_task_id,
                                        task=DisaggDecodeTask(
                                            task_id=decode_task_id,
                                            command_id=command.command_id,
                                            instance_id=selected_instance.instance_id,
                                            task_status=TaskStatus.Pending,
                                            task_params=command.task_params,
                                            kv_transfer_port=selected_instance.kv_transfer_port,
                                            paired_task_id=prefill_task_id,
                                        ),
                                    )
                                )

                                self.command_task_mapping[command.command_id] = (
                                    decode_task_id
                                )
                            elif isinstance(selected_instance, DisaggregatedInstance):
                                # Disaggregated: create prefill + decode tasks
                                prefill_task_id = TaskId()
                                decode_task_id = TaskId()

                                generated_events.append(
                                    TaskCreated(
                                        task_id=prefill_task_id,
                                        task=DisaggPrefillTask(
                                            task_id=prefill_task_id,
                                            command_id=command.command_id,
                                            instance_id=selected_instance.instance_id,
                                            task_status=TaskStatus.Pending,
                                            task_params=command.task_params,
                                            decode_node_host=selected_instance.decode_node_host,
                                            decode_node_port=selected_instance.kv_transfer_port,
                                            paired_task_id=decode_task_id,
                                        ),
                                    )
                                )
                                generated_events.append(
                                    TaskCreated(
                                        task_id=decode_task_id,
                                        task=DisaggDecodeTask(
                                            task_id=decode_task_id,
                                            command_id=command.command_id,
                                            instance_id=selected_instance.instance_id,
                                            task_status=TaskStatus.Pending,
                                            task_params=command.task_params,
                                            kv_transfer_port=selected_instance.kv_transfer_port,
                                            paired_task_id=prefill_task_id,
                                        ),
                                    )
                                )

                                self.command_task_mapping[command.command_id] = (
                                    decode_task_id
                                )
                            else:
                                task_id = TaskId()
                                generated_events.append(
                                    TaskCreated(
                                        task_id=task_id,
                                        task=TextGenerationTask(
                                            task_id=task_id,
                                            command_id=command.command_id,
                                            instance_id=available_instance_ids[0],
                                            task_status=TaskStatus.Pending,
                                            task_params=command.task_params,
                                        ),
                                    )
                                )

                                self.command_task_mapping[command.command_id] = task_id
                        case ImageGeneration():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=ImageGenerationTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id

                            if EXO_TRACING_ENABLED:
                                selected_instance = self.state.instances.get(
                                    selected_instance_id
                                )
                                if selected_instance:
                                    ranks = set(
                                        shard.device_rank
                                        for shard in selected_instance.shard_assignments.runner_to_shard.values()
                                    )
                                    self._expected_ranks[task_id] = ranks
                        case ImageEdits():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=ImageEditsTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id

                            if EXO_TRACING_ENABLED:
                                selected_instance = self.state.instances.get(
                                    selected_instance_id
                                )
                                if selected_instance:
                                    ranks = set(
                                        shard.device_rank
                                        for shard in selected_instance.shard_assignments.runner_to_shard.values()
                                    )
                                    self._expected_ranks[task_id] = ranks
                        case DeleteInstance():
                            placement = delete_instance(command, self.state.instances)
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            for cmd in cancel_unnecessary_downloads(
                                placement, self.state.downloads
                            ):
                                await self.download_command_sender.send(
                                    ForwarderDownloadCommand(
                                        origin=self._system_id, command=cmd
                                    )
                                )
                            generated_events.extend(transition_events)
                        case PlaceInstance():
                            if (
                                command.instance_meta
                                == InstanceMeta.TensorPrefillDisagg
                            ):
                                placement = place_tensor_prefill_disagg_instance(
                                    command.model_card,
                                    self.state.topology,
                                    self.state.instances,
                                    self.state.node_identities,
                                    self.state.node_network,
                                )
                            elif command.instance_meta == InstanceMeta.Disaggregated:
                                placement = place_disaggregated_instance(
                                    command.model_card,
                                    self.state.instances,
                                    self.state.node_identities,
                                    self.state.node_network,
                                )
                            else:
                                placement = place_instance(
                                    command,
                                    self.state.topology,
                                    self.state.instances,
                                    self.state.node_memory,
                                    self.state.node_network,
                                )
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case CreateInstance():
                            placement = add_instance_to_placements(
                                command,
                                self.state.topology,
                                self.state.instances,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case SendInputChunk(chunk=chunk):
                            generated_events.append(
                                InputChunkReceived(
                                    command_id=chunk.command_id,
                                    chunk=chunk,
                                )
                            )
                        case TaskCancelled():
                            if (
                                task_id := self.command_task_mapping.get(
                                    command.cancelled_command_id
                                )
                            ) is not None:
                                generated_events.append(
                                    TaskStatusUpdated(
                                        task_status=TaskStatus.Cancelled,
                                        task_id=task_id,
                                    )
                                )
                                # Cascade cancellation to paired disagg task
                                task = self.state.tasks.get(task_id)
                                if (
                                    task is not None
                                    and isinstance(task, _PAIRED_TASK_TYPES)
                                    and task.paired_task_id is not None
                                ):
                                    generated_events.append(
                                        TaskStatusUpdated(
                                            task_status=TaskStatus.Cancelled,
                                            task_id=task.paired_task_id,
                                        )
                                    )
                        case TaskFinished():
                            if (
                                task_id := self.command_task_mapping.pop(
                                    command.finished_command_id, None
                                )
                            ) is not None:
                                generated_events.append(TaskDeleted(task_id=task_id))
                                # Clean up paired task via paired_task_id
                                finished_task = self.state.tasks.get(task_id)
                                if (
                                    finished_task is not None
                                    and isinstance(finished_task, _PAIRED_TASK_TYPES)
                                    and finished_task.paired_task_id is not None
                                ):
                                    generated_events.append(
                                        TaskDeleted(
                                            task_id=finished_task.paired_task_id
                                        )
                                    )
                        case RequestEventLog():
                            # We should just be able to send everything, since other buffers will ignore old messages
                            # rate limit to 1000 at a time
                            end = min(command.since_idx + 1000, len(self._event_log))
                            for i, event in enumerate(
                                self._event_log.read_range(command.since_idx, end),
                                start=command.since_idx,
                            ):
                                await self._send_event(IndexedEvent(idx=i, event=event))
                    for event in generated_events:
                        await self.event_sender.send(event)
                except BrokenResourceError:
                    logger.warning(
                        "Event sender closed (likely master shutdown during election)"
                    )
                    return
                except ValueError as e:
                    logger.opt(exception=e).warning("Error in command processor")

    # These plan loops are the cracks showing in our event sourcing architecture - more things could be commands
    async def _plan(self) -> None:
        while True:
            try:
                # kill broken instances
                connected_node_ids = set(self.state.topology.list_nodes())
                for instance_id, instance in self.state.instances.items():
                    for node_id in instance.shard_assignments.node_to_runner:
                        if node_id not in connected_node_ids:
                            await self.event_sender.send(
                                InstanceDeleted(instance_id=instance_id)
                            )
                            break

                # time out dead nodes
                for node_id, time in self.state.last_seen.items():
                    now = datetime.now(tz=timezone.utc)
                    if now - time > timedelta(seconds=30):
                        logger.info(
                            f"Manually removing node {node_id} due to inactivity"
                        )
                        await self.event_sender.send(NodeTimedOut(node_id=node_id))
            except BrokenResourceError:
                logger.warning("Event sender closed in plan loop")
                return

            await anyio.sleep(10)

    async def _event_processor(self) -> None:
        with self.local_event_receiver as local_events:
            async for local_event in local_events:
                # Discard all events not from our session
                if local_event.session != self.session_id:
                    continue
                self._multi_buffer.ingest(
                    local_event.origin_idx,
                    local_event.event,
                    local_event.origin,
                )
                for event in self._multi_buffer.drain():
                    if isinstance(event, TracesCollected):
                        await self._handle_traces_collected(event)
                        continue

                    # Deduplicate NodeGatheredInfo: skip if same (node, info_type) was processed recently
                    if isinstance(event, NodeGatheredInfo):
                        info_type_name = type(event.info).__name__
                        dedup_key = (event.node_id, info_type_name)
                        now_mono = time.monotonic()
                        last_time = self._last_gathered_info.get(dedup_key, 0.0)
                        if now_mono - last_time < self._gathered_info_cooldown:
                            continue
                        self._last_gathered_info[dedup_key] = now_mono

                    logger.debug(f"Master indexing event: {str(event)[:100]}")
                    indexed = IndexedEvent(event=event, idx=len(self._event_log))
                    self.state = apply(self.state, indexed)

                    event._master_time_stamp = datetime.now(tz=timezone.utc)  # pyright: ignore[reportPrivateUsage]
                    if isinstance(event, NodeGatheredInfo):
                        event.when = str(datetime.now(tz=timezone.utc))

                    self._event_log.append(event)
                    await self._send_event(indexed)

                    # Cascade failure to paired disagg task
                    if isinstance(event, TaskFailed):
                        failed_task = self.state.tasks.get(event.task_id)
                        if (
                            failed_task is not None
                            and isinstance(failed_task, _PAIRED_TASK_TYPES)
                            and failed_task.paired_task_id is not None
                        ):
                            paired = self.state.tasks.get(failed_task.paired_task_id)
                            if paired is not None and paired.task_status not in (
                                TaskStatus.Complete,
                                TaskStatus.Failed,
                                TaskStatus.Cancelled,
                            ):
                                cancel_event = TaskStatusUpdated(
                                    task_id=failed_task.paired_task_id,
                                    task_status=TaskStatus.Cancelled,
                                )
                                await self.event_sender.send(cancel_event)

    # This function is re-entrant, take care!
    async def _send_event(self, event: IndexedEvent):
        # Convenience method since this line is ugly
        await self.global_event_sender.send(
            GlobalForwarderEvent(
                origin=self.node_id,
                origin_idx=event.idx,
                session=self.session_id,
                event=event.event,
            )
        )

    async def _handle_traces_collected(self, event: TracesCollected) -> None:
        task_id = event.task_id
        if task_id not in self._pending_traces:
            self._pending_traces[task_id] = {}
        self._pending_traces[task_id][event.rank] = event.traces

        if (
            task_id in self._expected_ranks
            and set(self._pending_traces[task_id].keys())
            >= self._expected_ranks[task_id]
        ):
            await self._merge_and_save_traces(task_id)

    async def _merge_and_save_traces(self, task_id: TaskId) -> None:
        all_trace_data: list[TraceEventData] = []
        for trace_data in self._pending_traces[task_id].values():
            all_trace_data.extend(trace_data)

        await self.event_sender.send(
            TracesMerged(task_id=task_id, traces=all_trace_data)
        )

        del self._pending_traces[task_id]
        if task_id in self._expected_ranks:
            del self._expected_ranks[task_id]
