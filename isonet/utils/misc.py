def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)


def pprint_without_newline(*args):
    print('\r', end='')
    print(*args, end=' ')
