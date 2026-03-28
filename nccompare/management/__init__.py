"""Command-line entrypoint helpers."""

import argparse

from nccompare import core
from nccompare.management.cli import get_args
from nccompare.printlib import formatter

def start_from_command_line_interface():
    args: argparse.Namespace = get_args()
    formatter.print_report(core.execute(**vars(args)))
