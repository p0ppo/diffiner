import click
from .src import cui

@click.group()
@click.option("--noisy")
@click.option("--proc")
@click.option("--output")
@click.option("--sound-class")
def main(noisy, proc, output, sound_class):
    pass

main.add_command(cui)
