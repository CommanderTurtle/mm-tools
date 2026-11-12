from apps.seg_ar_job_store import JobStore


class StudioStore(JobStore):
    @staticmethod
    def _job_session_id(job):
        if job.get("kind") in {"snapshot", "restore"}:
            return str(job["payload"]["session_id"])
        return JobStore._job_session_id(job)
