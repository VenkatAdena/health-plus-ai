"""Day 1 smoke test: prove that configuration and logging work end to end.

Run from the project root:

    python scripts/verify_setup.py
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.table import Table

from healthplus.config import get_settings
from healthplus.core import configure_logging

logger = logging.getLogger("healthplus.verify")


def main() -> None:
    settings = get_settings()
    settings.ensure_directories()
    configure_logging(level=settings.log_level, log_dir=settings.log_dir)

    logger.info("Logging configured (level=%s)", settings.log_level)
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    table = Table(title="HealthPlus AI — Active Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    key = settings.anthropic_api_key
    table.add_row("environment", settings.environment)
    table.add_row("debug", str(settings.debug))
    table.add_row("log_level", settings.log_level)
    table.add_row("claude_model", settings.claude_model)
    table.add_row("data_dir", str(settings.data_dir.resolve()))
    table.add_row("log_dir", str(settings.log_dir.resolve()))
    table.add_row("chroma_dir", str(settings.chroma_dir.resolve()))
    table.add_row("anthropic_api_key", str(key) if key else "[red]NOT SET[/red]")

    Console().print(table)

    if key is None:
        logger.warning("ANTHROPIC_API_KEY is not set — required before Claude integration")

    logger.info("Foundation verified ✔")


if __name__ == "__main__":
    main()
