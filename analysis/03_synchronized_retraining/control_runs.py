"""Synchronized-retraining Methods: ACDC inputs to same-label control outputs.

The five-epoch control uses the original paired training implementation.
This entry point does not start a convention-training campaign.
"""

import sys

from train_synchronized_pair import main


if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv[1:1] = ["--mode", "sham-5epoch"]
    if "--mode" in sys.argv:
        index = sys.argv.index("--mode")
        if len(sys.argv) > index + 1 and sys.argv[index + 1] not in (
            "sham-batch", "sham-5epoch"
        ):
            raise SystemExit("Controls allow only sham-batch or sham-5epoch")
    raise SystemExit(main())
