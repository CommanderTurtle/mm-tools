# Copyright 2026 Adventists AI. Apache-2.0.
"""Default engine settings of the released Sophon-Realtime v1 (Expressive). Any of them can be overridden by the environment."""
import os

DEFAULTS = dict(
    REGEN="1", FM_K="1024", LORA_R="256", LORA_A="512", EV_NC="9", E2_DIN="5120",
    E2V2_HOLD_SEC="0", ST_TAIL_TOK="3", T32_AHEAD="6", T32_EV="1", EV_GAIN="1", EV_LGAIN="1", EVFIX="1", EV="1", FR65="1",
    SPK_MODE="ctx", SPK_CTX="1", HID="1", T32_SERIAL_BUDGET="6", T32_SCHED_BUDGET="4", SPEED65="1",
    E2V2="1", E2V2_NT="4", E2V2_CFG="0", E2V2_NT_FIRST="0", E2V2_KVC="1", E2V2_LEFT="12", E2V2_LA_FIRST="1", E2V2_LA_FIRST_N="3",
    E2V2_OUT_GAIN="0.5", E2V2_FADEIN="1", E2_PROMPT="1", E2V2_HIFT_ABS="1", E2V2_HIFT_CTX="40", E2V2_HIFT_ANCHOR="0",
    NVIDIA_TF32_OVERRIDE="1", T32_FP8="1", T32_M2="1", T32_SCHED="1", T32_SLOTS="8", T32_KEEP_LMH="0", T32_CTXGRAPH="1",
    T32_SPLIT="0", T32_CAPC="1280", T32_CAPT="4096", T32_DBG="0", HF_HUB_OFFLINE="1",
)


def apply_defaults():
    for k, v in DEFAULTS.items():
        os.environ.setdefault(k, v)
