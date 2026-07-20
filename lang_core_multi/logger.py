"""
logger.py — Centralized logging configuration for the Gen 3 multi-agent system.

Usage
-----
    from lang_core_multi.logger import get_agent_logger
    logger = get_agent_logger(__name__)

    logger.info("Something happened")
    logger.debug("Detailed trace: %s", some_value)
    logger.error("Something failed", exc_info=True)

Output
------
  Terminal : coloured, human-readable via colorlog.ColoredFormatter
  File     : clean plain-text to agent_system.log in the project root
"""

import logging
# import colorlog

# ── Log format strings ───────────────────────────────────────────────────────
_COLOR_FORMAT = "%(log_color)s%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_FILE_FORMAT  = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DATE_FORMAT  = "%Y-%m-%d %H:%M:%S"

# Map log levels to terminal colours
_LOG_COLORS = {
    "DEBUG":    "cyan",
    "INFO":     "green",
    "WARNING":  "yellow",
    "ERROR":    "red",
    "CRITICAL": "bold_red",
}

# Path for the persistent log file (project root, next to main.py)
_LOG_FILE = "agent_system.log"

# Internal flag — setup runs exactly once per process
_configured = False


def _configure_root_logger() -> None:
    """
    Configures the root logger once.

    Called automatically the first time get_agent_logger() is used.
    Subsequent calls are no-ops.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)          # capture everything; handlers filter

    # # ── Terminal handler (coloured) ──────────────────────────────────────────
    # stream_handler = logging.StreamHandler()
    # stream_handler.setLevel(logging.DEBUG)
    # stream_handler.setFormatter(
    #     colorlog.ColoredFormatter(
    #         fmt=_COLOR_FORMAT,
    #         datefmt=_DATE_FORMAT,
    #         log_colors=_LOG_COLORS,
    #     )
    # )

    # ── File handler (plain text) ────────────────────────────────────────────
    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(fmt=_FILE_FORMAT, datefmt=_DATE_FORMAT)
    )

    # root.addHandler(stream_handler)
    root.addHandler(file_handler)

    _configured = True


def get_agent_logger(name: str) -> logging.Logger:
    """
    Returns a named logger under the configured root logger.

    Parameters
    ----------
    name : str
        Typically __name__ from the calling module.

    Returns
    -------
    logging.Logger
        A logger that writes to both the coloured terminal stream
        and the plain-text agent_system.log file.
    """
    _configure_root_logger()
    return logging.getLogger(name)
