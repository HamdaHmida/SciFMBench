"""Top-level launcher.

Equivalent to `python -m engine.run`. Lets you do:

    python run.py --model MPP --mode train --config configs/mpp_default.json
"""
from engine.run import main

if __name__ == "__main__":
    main()
