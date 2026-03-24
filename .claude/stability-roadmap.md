# Stability & Integration Roadmap

Research conducted 2026-03-02 on branch `feat/event-driven-plan-and-task-pairing`.

## Tier 1: Active Failure Modes

### 1. Instance deletion doesn't cascade task cleanup
- **Location**: `src/exo/shared/apply.py:186-190` (`apply_instance_deleted`)
- **Problem**: Only removes instance from state. Tasks for that instance are orphaned — they linger in `state.tasks` forever, runners keep running, KV transfer servers keep listening.
- **Root cause of**: The "stale FAILED instance" seen in dashboard after benchmarks.
- **Fix**: Master's `_event_processor` should emit `TaskDeleted` for all tasks with matching `instance_id` when `InstanceDeleted` arrives. Worker should shut down runners whose instance is gone.

### 2. KV transfer has no socket timeout on receive side
- **Location**: `src/exo/worker/engines/mlx/kv_transfer.py:972-979` (`KVTransferServer.receive()`)
- **Problem**: `_receive_one_connection()` blocks on socket operations with no timeout. If prefill sender dies mid-transfer, decode runner hangs forever.
- **Impact**: Deadlock — decode runner stuck on `recv()`, never reports failure, never accepts new tasks.
- **Fix**: Add `SO_RCVTIMEO` to KV transfer socket, or use `select()` with timeout. On timeout, raise and let runner report TaskFailed.

### 3. RunnerSupervisor silently drops tasks when runner dies
- **Location**: `src/exo/worker/runner/runner_supervisor.py:153-155`
- **Problem**: If task sender is closed (runner died), task is dropped with a warning but **no TaskFailed event is sent**. Master thinks task is still running.
- **Also**: `_watch_runner` polls every 5s (`line 207-212`), so there's up to 5s delay before death detection.
- **Also**: `_forward_events` at lines 198-199 unblocks pending tasks with `set()` but never marks them failed.
- **Fix**: In `ClosedResourceError` handler, emit `TaskFailed` event for the dropped task. In `_check_runner`, iterate `self.pending` and emit `TaskFailed` for all pending tasks.

### 4. No task timeout mechanism
- **Location**: `src/exo/master/main.py:520-535` (`_plan` loop)
- **Problem**: Master has 30s node timeout, but **no timeout for tasks**. Tasks stuck in Pending/Running stay forever.
- **Common scenario**: Decode task waiting for KV transfer that never arrives (prefill node crashed).
- **Fix**: Add task age check in master's `_plan` loop. Tasks in Pending/Running beyond threshold → emit `TaskFailed(error_type="timeout")`.

### 5. ValueError in command processor never generates TaskFailed
- **Location**: `src/exo/master/main.py:510-511`
- **Problem**: `ValueError("No instance found for model...")` is caught and logged as warning, but the API client gets no error — it waits forever.
- **Fix**: Catch ValueError per-command, emit error event back to API.

### 6. No error handling in `_event_processor`
- **Location**: `src/exo/master/main.py:541-596`
- **Problem**: If `apply(self.state, indexed)` raises (corrupted event, state violation), the entire event pipeline crashes. No try-except.
- **Fix**: Wrap in try-except, log error, skip corrupted event.

## Tier 2: Observability Gaps

### 7. Silent paired task cascades (no logging)
- **Location**: `src/exo/master/main.py:578-596` (TaskFailed cascade), lines 458-474 (TaskCancelled cascade), lines 478-493 (TaskFinished cascade)
- **Problem**: All three cascade paths operate silently — no `logger.warning` showing which task triggered cascade, with what command_id, to what paired_task_id.
- **Fix**: Add `logger.info` after each cascade event emission showing the chain.

### 8. Worker plan_step doesn't log wake vs timeout
- **Location**: `src/exo/worker/main.py:164-170`
- **Problem**: After event-driven change, no way to tell from logs whether plan_step woke on event or hit 2s fallback.
- **Fix**: Add `logger.debug` for wake source (event vs timeout).

### 9. No task state in dashboard
- **Location**: `dashboard/src/lib/stores/app.svelte.ts`
- **Problem**: Dashboard has zero task status display. No visibility into: task count per instance, failed tasks, paired task relationships, task-to-command mapping.
- **Fix**: Add tasks to State TypeScript interface, render in instance detail view.

### 10. No task query API
- **Location**: `src/exo/master/api.py`
- **Problem**: No `/v1/tasks` endpoint. Only way to see tasks is parsing full `/state` response.
- **Fix**: Add REST endpoints for task listing/filtering.

### 11. Health endpoint is minimal
- **Location**: `src/exo/master/api.py:357-368`
- **Problem**: Only checks `self.paused` and `node_count`. Doesn't report: pending task count, failed task count, stuck tasks, instance health.
- **Fix**: Expand health endpoint with task/instance summary.

## Tier 3: Code Unification

### 12. Master command processor duplication (~89 lines)
- **Location**: `src/exo/master/main.py:186-263`
- **Problem**: TensorPrefillDisagg and Disaggregated branches are near-identical. Both create prefill+decode tasks with paired_task_id.
- **Fix**: Extract `_create_disagg_tasks(instance, command, prefill_task_class)` helper.

### 13. Runner task dispatch duplication (~362 lines)
- **Location**: `src/exo/worker/runner/llm_inference/runner.py:464-826`
- **Problem**: DisaggPrefill, TensorParallelDisaggPrefill, and DisaggDecode have 80% shared code: task ack, model assertions, stream gen, KV cleanup, error chunk creation.
- **Fix**: Extract shared setup/teardown, KV cleanup into helpers. Consider PrefillExecutor abstraction.

### 14. Placement logic duplication (~31 lines)
- **Location**: `src/exo/master/placement.py:236-243` vs `320-341` (node detection), `264-277` vs `395-408` (IP selection)
- **Fix**: Extract `detect_prefill_decode_nodes()` and `find_decode_host()` helpers.

### 15. No base class for disagg instances
- **Location**: `src/exo/shared/types/worker/instances.py`
- **Problem**: `DisaggregatedInstance` and `TensorPrefillDisaggInstance` share `decode_node_id`, `decode_node_host`, `kv_transfer_port` but have no common base.
- **Fix**: Create `BaseDisaggInstance` with shared decode config.

### 16. Paired task cascade pattern repeated 3x
- **Location**: `src/exo/master/main.py` — TaskCancelled (466-474), TaskFinished (486-493), TaskFailed (583-596)
- **Fix**: Extract `_cascade_to_paired_task(task_id, target_event_factory)` helper.

### 17. Worker plan.py has 19 isinstance branches
- **Location**: `src/exo/worker/plan.py:174-451`
- **Problem**: `_init_distributed_backend`, `_load_model`, `_ready_to_warmup`, `_pending_tasks` all have cascading isinstance checks for each instance type.
- **Fix**: Add query methods to instance types (`is_independent()`, `is_prefill_node()`, `get_peer_runners()`) to move dispatch logic into the types themselves.

## Testing Gaps

### No integration test for disagg lifecycle through Master
- **Location**: `src/exo/master/tests/test_master.py` — only tests MlxRing flow
- **Missing**: async test that places disagg instance, creates paired tasks, sends TaskFailed for prefill, asserts decode gets cancelled.
- **Challenge**: `place_disaggregated_instance` requires specific `node_identities` with chip_id values ("dgx-spark", "Apple M4 Max"), making integration tests more complex to set up.

### No test for task timeout
- **Missing**: Test that tasks stuck beyond threshold are cleaned up.

### No test for stale paired task detection
- **Missing**: Test for scenario where prefill completes but decode hangs — needs reconciliation.

## Implementation Priority

**Phase 1** (stability): Items 1-4 — fix the bugs that cause real failures
**Phase 2** (observability): Items 7-8 — add logging to the code we just wrote
**Phase 3** (unification): Items 12, 14, 16 — low-risk deduplication
**Phase 4** (deeper refactors): Items 13, 15, 17 — higher-risk structural changes
