"""
Helper function to translate from gripper ratio into the four finger joint commands.
"""

from __future__ import annotations

import numpy as np

# order: a_RIGHT_BOTTOM, a_RIGHT_TIP, a_LEFT_BOTTOM, a_LEFT_TIP
_GRIP_MIN = np.array([-0.09, -0.5, -0.96, -0.5])
_GRIP_MAX = np.array([0.96, 0.21, 0.09, 0.21])


def gripper_ctrl_from_ratio(ratio: float):
    r = float(np.clip(ratio, 0.0, 1.0))
    raw = np.array(
        [
            r * 0.96,
            r * -1.03,
            r * -0.96,
            r * -1.03,
        ]
    )
    return np.clip(raw, _GRIP_MIN, _GRIP_MAX)
