"""
Design recap (see prior conversation for full rationale):
- 1 coordinator thread per stage; func is ALWAYS batch_in -> batch_out,
  de-batched uniformly by the coordinator (so stage2 can fan out 1->N
  with no special casing anywhere in the harness).
- Coordinator loop = 3 independent checks per iteration: pull-if-not-
  draining, submit-if-full-or-final-partial, stop-or-opportunistic-drain.
- Pool room checked in exactly one place (_submit_when_room) via
  concurrent.futures.wait() over a capped in_flight set.
- 4 mutually exclusive states (working / waiting_for_input /
  waiting_for_pool / waiting_for_downstream); "stuck" is DERIVED
  (still "working" past stuck_threshold), not a 5th state.
- items_in / items_out / errors tracked separately, in input-item units.
- kill_on_stuck is a GLOBAL policy; stuck_threshold is a PER-STAGE
  constructor parameter (no StageRunner/StageSpec split -- deliberately
  rejected as overengineering).
- Exactly ONE sentinel needed at pipeline entry now (one coordinator
  per stage, no peer-thread race to resolve).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, Future, wait, FIRST_COMPLETED
from dataclasses import dataclass
from typing import Any, Callable, Optional




POLL_INTERVAL = 0.2


# --------------------------------------------------------------------------
# Coordinator activity record
# --------------------------------------------------------------------------

class CoordinatorState:
    WORKING = "working"
    WAITING_FOR_INPUT = "waiting_for_input"
    WAITING_FOR_POOL = "waiting_for_pool"
    WAITING_FOR_DOWNSTREAM = "waiting_for_downstream"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = self.WORKING
        self._since = time.monotonic()

    def set(self, state: str) -> None:
        with self._lock:
            self._state = state
            self._since = time.monotonic()

    def snapshot(self) -> tuple[str, float]:
        with self._lock:
            return self._state, time.monotonic() - self._since

    def is_stuck(self, threshold: float) -> bool:
        state, age = self.snapshot()
        return state == self.WORKING and age > threshold


@dataclass
class InFlightBatch:
    future: Future
    dispatched_at: float
    items_in: int


# --------------------------------------------------------------------------
# CoordinatedStage
# --------------------------------------------------------------------------

class CoordinatedStage:
    """
    One stage: a single coordinator thread that pulls from in_q, batches,
    dispatches batches to an internal ThreadPoolExecutor (func is always
    batch_in -> batch_out), de-batches results onto out_q, and tracks its
    own health/stats. No per-worker threads of its own otherwise.
    """

    def __init__(
        self,
        name: str,
        in_q: "queue.Queue",
        out_q: Optional["queue.Queue"],
        func: Callable[[list], list],
        max_workers: int = 4,
        batch_size: int = 1,
        stuck_threshold: float = 60.0,
    ):
        self.name = name
        self.in_q = in_q
        self.out_q = out_q
        self.func = func
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.stuck_threshold = stuck_threshold

        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.state = CoordinatorState()

        self.items_in = 0
        self.items_out = 0
        self.errors = 0

        self._thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._coordinator_loop, name=f"{self.name}-coord", daemon=True
        )
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive() if self._thread else False

    # ----------------------------------------------------------------
    # The coordinator loop -- three independent checks per iteration
    # ----------------------------------------------------------------

    def _coordinator_loop(self) -> None:
        batch: list[Any] = []
        in_flight: dict[Future, InFlightBatch] = {}
        sentinel_seen = False

        while True:
            # ---- 1. Pull one token, unless already draining ----
            if not sentinel_seen:
                self.state.set(CoordinatorState.WAITING_FOR_INPUT)
                try:
                    token = self.in_q.get(timeout=POLL_INTERVAL)
                    self.in_q.task_done()
                    if token is None:
                        sentinel_seen = True
                    else:
                        batch.append(token)
                except queue.Empty:
                    pass  # nothing yet -- try later

            # ---- 2. Submit if batch is full, or it's the final partial ----
            if len(batch) >= self.batch_size or (sentinel_seen and batch):
                self._submit_when_room(batch, in_flight)
                batch = []

            # ---- 3. Fully stopped? else opportunistically drain ----
            if sentinel_seen and not batch and not in_flight:
                self.state.set(CoordinatorState.WORKING)
                if self.out_q is not None:
                    self._put_downstream(None)
                self._stopped.set()
                logging.info(f"{self.name}: drained, forwarded STOP")
                return

            if in_flight:
                done = {f for f in in_flight if f.done()}
                if done:
                    self.state.set(CoordinatorState.WORKING)
                    self._drain_completed(done, in_flight)

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _submit_when_room(
        self, batch: list[Any], in_flight: dict[Future, InFlightBatch]
    ) -> None:
        while len(in_flight) >= self.max_workers:
            self.state.set(CoordinatorState.WAITING_FOR_POOL)
            done, _ = wait(
                in_flight.keys(), timeout=POLL_INTERVAL,
                return_when=FIRST_COMPLETED,
            )
            if done:
                self._drain_completed(done, in_flight)

        self.state.set(CoordinatorState.WORKING)
        future = self.executor.submit(self.func, list(batch))
        in_flight[future] = InFlightBatch(
            future=future, dispatched_at=time.monotonic(), items_in=len(batch)
        )

    def _drain_completed(
        self, done: set[Future], in_flight: dict[Future, InFlightBatch]
    ) -> None:
        for future in done:
            record = in_flight.pop(future)
            self.items_in += record.items_in
            try:
                batch_out = future.result()
            except Exception:
                logging.exception(f"error in stage {self.name}")
                self.errors += record.items_in
                continue

            for out_item in batch_out:
                if self.out_q is not None:
                    self._put_downstream(out_item)
            self.items_out += len(batch_out)

    def _put_downstream(self, item: Any) -> None:
        while True:
            try:
                self.out_q.put(item, timeout=POLL_INTERVAL)  # type: ignore[union-attr] # noqa: F821
                return
            except queue.Full:
                self.state.set(CoordinatorState.WAITING_FOR_DOWNSTREAM)
                continue

    # ----------------------------------------------------------------
    # Monitoring surface
    # ----------------------------------------------------------------

    def health_snapshot(self) -> dict:
        state, age = self.state.snapshot()
        return {
            "state": state,
            "state_age_s": round(age, 2),
            "stuck": self.state.is_stuck(self.stuck_threshold),
            "items_in": self.items_in,
            "items_out": self.items_out,
            "errors": self.errors,
            "pool_max_workers": self.max_workers,
        }

    def emergency_stop(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


# --------------------------------------------------------------------------
# Pipeline construction / teardown
# --------------------------------------------------------------------------

@dataclass
class StageSpec:
    func: Callable[[list], list]
    max_workers: int = 4
    batch_size: int = 1
    stuck_threshold: float = 60.0


class Pipeline():     
    def __init__(self,
        specs: list[StageSpec],
        queue_maxsize: int = 100,
    ) -> None:
        queues = [queue.Queue(maxsize=queue_maxsize) for _ in range(len(specs) + 1)]
        stages = []
        for i, spec in enumerate(specs):
            stage = CoordinatedStage(
                name=f"stage{i + 1}",
                in_q=queues[i],
                out_q=queues[i + 1],
                func=spec.func,
                max_workers=spec.max_workers,
                batch_size=spec.batch_size,
                stuck_threshold=spec.stuck_threshold,
            )
            stages.append(stage)
        self.stages = stages
        self.queues = queues
        self.monitor_logger = logging.getLogger("pipeline.monitor")
        self.monitor_logger.setLevel(logging.INFO)
        self.pid: Optional[int] = None
        self.monitor_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        for s in self.stages:
            s.start()
        self.pid = self.write_pid_file()
        
    def monitor_start(self, interval: float = 2.0, kill_on_stuck: str = "yes") -> None:
        self.monitor_thread = threading.Thread(
            target=self.monitor,
            args=(),
            kwargs={"interval": interval, "kill_on_stuck": kill_on_stuck},
            daemon=True,
        )
        self.monitor_thread.start()
        
    def run(self, input: list[Any], monitoring_interval: float = 1.0, kill_on_stuck: str = "yes") -> None:
        self.start()
        self.monitor_start(interval=monitoring_interval, kill_on_stuck=kill_on_stuck)
        for item in input:
            self.queues[0].put(item)
        self.queues[0].put(None)
        final_q = self.queues[-1]
        for item in tqdm(iter(final_q.get, None)):
            final_q.task_done()
        final_q.task_done()  # account for the sentinel None
        self.stages[-1]._stopped.wait()
        
        
    
    def shutdown_gracefully(self, stages: list[CoordinatedStage], queues: list[queue.Queue]) -> None:
        queues[0].put(None)
        for s in stages:
            s.join()
        if self.monitor_thread is not None:
            self.monitor_thread.join()

# --------------------------------------------------------------------------
# Monitor (direct port of old monitor(), reading health_snapshot() instead
# of stuck_workers()/alive_count())
# --------------------------------------------------------------------------
    def configure_monitor_log_file(self,path: str = "pipeline_monitor.jsonl") -> None:
        
        handler = logging.FileHandler(path, mode="a")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.monitor_logger.addHandler(handler)
        self.monitor_logger.propagate = False
        

    def monitor(
        self,
        
        interval: float = 2.0,
        kill_on_stuck: str = "yes",  # "yes" | "no" | "ask"  -- GLOBAL policy
        kill_flag_path: str = "pipeline_kill.flag",
    ) -> None:
        assert kill_on_stuck in ("yes", "no", "ask")

        prev_in = {s.name: 0 for s in self.stages}
        prev_t = time.monotonic()

        while True:
            time.sleep(interval)
            now = time.monotonic()
            dt = max(now - prev_t, 1e-9)
            prev_t = now

            snapshot: dict[str, Any] = {"ts": time.time(), "pid": self.pid, "stages": {}, "queues": {}}
            any_stuck = False
            stuck_detail: dict[str, dict] = {}

            for s in self.stages:
                health = s.health_snapshot()
                rate = (health["items_in"] - prev_in[s.name]) / dt
                prev_in[s.name] = health["items_in"]
                health["items_in_rate"] = round(rate, 2)

                if health["stuck"]:
                    any_stuck = True
                    stuck_detail[s.name] = health

                snapshot["stages"][s.name] = health

            for i, q in enumerate(self.queues):
                snapshot["queues"][f"q{i}"] = {"size": q.qsize()}

            self.monitor_logger.info(json.dumps(snapshot))

            if os.path.exists(kill_flag_path):
                logging.critical("External kill requested via flag file — exiting")
                os._exit(1)

            if any_stuck:
                if kill_on_stuck == "yes":
                    logging.critical(f"Stuck stage(s) detected — killing process: {stuck_detail}")
                    os._exit(1)
                elif kill_on_stuck == "no":
                    logging.critical(f"Stuck stage(s) detected — continuing (kill_on_stuck=no): {stuck_detail}")
                elif kill_on_stuck == "ask":
                    logging.critical(
                        f"Stuck stage(s) detected — waiting for external kill via "
                        f"{kill_flag_path} (kill_on_stuck=ask): {stuck_detail}"
                    )

            if all(not s.is_alive() for s in self.stages):
                logging.info("All stages finished — monitor exiting")
                return


# --------------------------------------------------------------------------
# Startup helper (PID file) 
# --------------------------------------------------------------------------

    def write_pid_file(self, path: str = "pipeline.pid") -> int:
        pid = os.getpid()
        with open(path, "w") as f:
            f.write(str(pid))
        print(f"=== Pipeline started, PID={pid} === kill with: kill -9 {pid}")
        logging.info(f"Pipeline running as PID {pid} (written to {path})")
        return pid
