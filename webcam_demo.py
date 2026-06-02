from __future__ import annotations

import sys

from main import main


if __name__ == "__main__":
    sys.argv = [
        "webcam_demo.py",
        "--source",
        "0",
        "--source-type",
        "webcam",
        "--display",
    ]
    main()
