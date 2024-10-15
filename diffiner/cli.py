import click
from .src import cui

@click.group()
def main():
    pass

main.add_command(cui)
