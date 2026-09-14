"""Private process fixture for the shared lease tests; never installed."""
import json
import os
import select
import subprocess
import sys
import time

from kilix_device_lease import LeaseError, acquire


def emit(value):
    print(json.dumps(value), flush=True)


def main():
    if sys.argv[1] == "supervisor":
        child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, close_fds=True)
        emit({"engine": child.pid})
        child.wait()
        return
    namespace, workload, job_id, seconds = sys.argv[1:5]
    flags = {"cancelled": False, "disconnected": False}

    def incoming():
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if not line:
                flags["disconnected"] = True
            elif line.strip() == "cancel":
                flags["cancelled"] = True
        return flags["cancelled"]

    try:
        lease = acquire(job_id=job_id, workload=workload, device="test-accelerator",
                        deadline=time.monotonic() + float(seconds), namespace=namespace,
                        cancelled=incoming, disconnected=lambda: flags["disconnected"],
                        progress=lambda row: emit({"state": row.state, "ticket": row.ticket,
                                                   "position": row.position, "job": job_id}))
        emit({"state": "held", "ticket": lease.ticket, "guard_fd": lease.guard_fd,
              "pid": os.getpid(), "job": job_id})
        while True:
            command = sys.stdin.readline().strip()
            if command == "release":
                lease.release(cleanup_complete=True)
                emit({"state": "released"})
                return
            if command == "supervisor":
                process = subprocess.Popen([sys.executable, __file__, "supervisor"],
                                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, pass_fds=(lease.guard_fd,))
                engine = json.loads(process.stdout.readline())
                emit({"supervisor": process.pid, **engine})
                continue
            lease.release()
            emit({"state": "quarantined"})
            return
    except LeaseError as error:
        emit({"state": "refused", "code": error.code})


if __name__ == "__main__":
    main()
