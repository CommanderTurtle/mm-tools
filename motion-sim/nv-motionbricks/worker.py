import os
import signal
import traceback


def serve(pipe):
    # Linux parent-death handling also covers a forcibly closed terminal/server.
    if os.name == "posix":
        import ctypes

        parent = os.getppid()
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        if os.getppid() != parent:
            return
    runtime = None
    try:
        while True:
            message = pipe.recv()
            try:
                if message["op"] == "load":
                    from native import Runtime

                    runtime = Runtime(message)
                    result = runtime.describe()
                elif runtime is None:
                    raise ValueError("Load a rig first.")
                else:
                    result = runtime.dispatch(message)
                pipe.send(result)
            except Exception as error:
                traceback.print_exc()
                pipe.send({"error": str(error)})
                if message.get("op") == "load":
                    return
    except (EOFError, BrokenPipeError):
        pass
    finally:
        if runtime is not None:
            runtime.engine.close()
        pipe.close()
