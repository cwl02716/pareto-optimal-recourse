import traceback
from typing import Any

import fire
from fire.core import FireExit


def fire_cmd(component: Any, name: str | None = None) -> None:
    prompt = "> "
    while True:
        try:
            fire.Fire(component, input(prompt), name)
        except (FireExit, EOFError):
            break
        except Exception:
            traceback.print_exc()
        finally:
            print()
