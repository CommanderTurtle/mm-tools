import json
import os
import sys

from paths import NATIVE, SNAPSHOTS, model_identity, parent_guard

sys.path.insert(0, str(NATIVE))


def main():
    parent_guard()
    from apps import seg_ar_web_server as native
    from queue_store import StudioStore

    class StudioGeneration(native.GenerationService):
        def execute_queue_job(self, store, job):
            kind = job["kind"]
            if kind not in {"snapshot", "restore"}:
                return super().execute_queue_job(store, job)
            payload = job["payload"]
            name, session_id = payload["snapshot_id"], payload["session_id"]
            if len(name) != 32 or any(c not in "0123456789abcdef" for c in name):
                raise ValueError("Invalid snapshot identifier")
            SNAPSHOTS.mkdir(parents=True, exist_ok=True)
            path = SNAPSHOTS / f"{name}.pt"
            record_path = SNAPSHOTS / f"{name}.json"
            try:
                if kind == "snapshot":
                    session = store.get_session(session_id)
                    if not session:
                        raise ValueError("World has no completed segment")
                    result = self.snapshot_session(session_id, str(path))
                    record = {
                        "id": name,
                        "session_id": session_id,
                        "session": session,
                        "identity": model_identity(os.environ["NOMAS_ADAPTER"]),
                        "studio": payload.get("studio", {}),
                        "title": payload.get("title", "World"),
                    }
                    temporary = record_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(record), encoding="utf-8")
                    temporary.replace(record_path)
                else:
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                    if record["identity"] != model_identity(
                        os.environ["NOMAS_ADAPTER"]
                    ):
                        raise ValueError(
                            "Load the same model/adapter used for this snapshot"
                        )
                    result = self.restore_session(session_id, str(path))
                    store.put_session(session_id, record["session"])
                    result["session"] = record["session"]
                    result["studio"] = record.get("studio", {})
                store.patch(job["id"], status="succeeded", result=result)
            except Exception as exc:  # noqa: BLE001 — native queue jobs report failure rather than killing the worker
                store.patch(job["id"], status="failed", error=str(exc))

    native.GenerationService = StudioGeneration
    native.JobStore = StudioStore
    native.main()


if __name__ == "__main__":
    main()
