from typing import Any

import typer


def run_shell(component: Any, name: str | None = None) -> None:
    shell = typer.Typer(
        name=name,
        no_args_is_help=True,
    )

    shell.command()(component)

    while True:
        try:
            args = typer.prompt("> ", prompt_suffix="").split()
        except typer.Abort:
            break

        try:
            shell(args)
        except SystemExit:
            pass
