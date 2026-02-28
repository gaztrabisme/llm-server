"""CLI entry point — Click group with subcommand registration."""

import click

from . import __version__


@click.group()
@click.version_option(version=__version__, prog_name="llm-bench")
def cli():
    """Reusable benchmark framework for llama.cpp LLM servers."""


# Import and register subcommands
from .commands.setup import setup  # noqa: E402
from .commands.speed import speed  # noqa: E402
from .commands.quality import quality  # noqa: E402
from .commands.eval import eval_cmd  # noqa: E402
from .commands.compare import compare  # noqa: E402
from .commands.report import report  # noqa: E402

cli.add_command(setup)
cli.add_command(speed)
cli.add_command(quality)
cli.add_command(eval_cmd, name="eval")
cli.add_command(compare)
cli.add_command(report)
