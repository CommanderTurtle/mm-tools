import os
import signal
import subprocess
import threading

from paths import DB, NATIVE, ROOT, WORK, WORKER_ID, offline_env, overrides


class Processes:
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        self.stop_lock = threading.Lock()
        self.children = set()
        self.worker = None
        self.adapter = "base"
        self.closed = False
        self.stopping = False
        self.epoch = 0
        WORK.mkdir(parents=True, exist_ok=True)
        self.log = (WORK / "runtime.log").open("ab", buffering=0)

    def spawn(self, args, *, epoch=None, **kwargs):
        with self.lock:
            if (
                self.closed
                or self.stopping
                or (epoch is not None and epoch != self.epoch)
            ):
                raise RuntimeError("Studio is shutting down")
            child = subprocess.Popen(
                [str(ROOT / ".venv/bin/python"), str(ROOT / "supervisor.py"), *args],
                cwd=NATIVE,
                env=kwargs.pop("env", offline_env()),
                start_new_session=True,
                stderr=self.log,
                **kwargs,
            )
            self.children.add(child)
            return child

    def reap(self, child):
        with self.lock:
            self.children.discard(child)

    @staticmethod
    def terminate(child):
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=10)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            pass
        finally:
            # A group can outlive its supervisor; clean descendants as well.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()

    def start(self, adapter):
        with self.lock:
            if self.worker and self.worker.poll() is None:
                if adapter != self.adapter:
                    raise ValueError("Stop the model before changing adapters")
                return
            command = [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "worker.py"),
                "--worker-only",
                "--job-db",
                str(DB),
                "--output-dir",
                str(WORK / "outputs"),
                "--rollout-type",
                "seg_ar",
                *overrides(adapter),
            ]
            self.worker = self.spawn(
                command, stdout=self.log, env=offline_env() | {"NOMAS_ADAPTER": adapter}
            )
            self.adapter = adapter
            threading.Thread(
                target=self._watch_worker, args=(self.worker,), daemon=True
            ).start()

    def _watch_worker(self, child):
        child.wait()
        with self.stop_lock:
            with self.lock:
                if self.worker is not child or self.closed or self.stopping:
                    return
            self._stop("Model exited; see runtime log, then create or restore a world")

    def stop(self):
        with self.stop_lock:
            self._stop()

    def _stop(self, reason="Stopped by user; create or restore a world"):
        with self.lock:
            self.stopping = True
            self.epoch += 1
            children = list(self.children)
        try:
            for child in children:
                self.terminate(child)
                self.reap(child)
            for job in self.store.list_jobs(limit=100000):
                if job["status"] in {"enhancing", "queued", "running"}:
                    self.store.patch(job["id"], status="failed", error=reason)
            self.store.reset_worker(WORKER_ID)
            self.store.clear_worker(WORKER_ID)
        finally:
            with self.lock:
                self.worker = None
                self.stopping = False

    def close(self):
        with self.lock:
            self.closed = True
        self.stop()
        self.log.close()
