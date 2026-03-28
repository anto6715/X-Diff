import nccompare
from nccompare.core import main


def execute(**kwargs):
    nccompare.setup()
    # kwargs.pop("verbose")
    get_version = kwargs.pop("get_version", False)

    if get_version:
        exit(0)

    main.execute(**kwargs)
