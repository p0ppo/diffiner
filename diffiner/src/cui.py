import os
import sys
import dotenv
import click

from .ddrm.run import run


def main():
    # Temporary, you need to implement interactive interface
    run()


@click.command(name="cui")
def cui():
    main()
