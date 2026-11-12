import runpy
import sys
from lifecycle import own, release
from native import install, local_dino

if __name__ == "__main__":
    name, parent = sys.argv[1:3]
    own(int(parent))
    try:
        install()
        local_dino()
        sys.argv = [name, *sys.argv[3:]]
        runpy.run_module(name, run_name="__main__")
    finally:
        release()
