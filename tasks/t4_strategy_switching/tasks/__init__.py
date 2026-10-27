"""Task modules for T4 Strategy Switching.

Each module exports:
    run(model, task)  -> dict with at least {solved: bool, messages: list}
    TASKS_JSON_PATH   -> str, relative path from TASKS_DIR to tasks.json
    JUDGE_CONTEXT     -> str, description for the LLM judge (optional)
    SWITCH_REGEX      -> str, regex pattern for fast switch detection (optional)
    detect_switch(result) -> bool or None, programmatic switch detection (optional)
"""
