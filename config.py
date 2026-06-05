import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from rich.logging import RichHandler

# Load .env variables
load_dotenv(override=True)

# Root directory
ROOT_DIR = Path(__file__).parent.resolve()

# App Configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "20"))
FILE_SIZE_WARNING_MB = int(os.getenv("FILE_SIZE_WARNING_MB", "100"))

# Directories
REPORTS_DIR = ROOT_DIR / "reports"
LOGS_DIR = ROOT_DIR / "logs"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure dual logging: console via RichHandler and file logging via FileHandler
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Setup root logger
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)

# Clean up existing handlers to avoid duplicates
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)

# Create console handler (Rich) - default to WARNING to keep CLI interface clean
console_log_level = os.getenv("CONSOLE_LOG_LEVEL", "WARNING").upper()
console_handler = RichHandler(rich_tracebacks=True, markup=True)
console_handler.setLevel(getattr(logging, console_log_level, logging.WARNING))
root_logger.addHandler(console_handler)

# Create file handler - keep it at LOG_LEVEL (e.g. INFO) for detailed background logging
file_handler = logging.FileHandler(LOGS_DIR / "app.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter(log_format))
file_handler.setLevel(LOG_LEVEL)
root_logger.addHandler(file_handler)

logger = logging.getLogger("reconciliation_agent")
logger.info("Logging configured. Log level: %s, Console level: %s", LOG_LEVEL, console_log_level)

