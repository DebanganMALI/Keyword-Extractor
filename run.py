from importlib import import_module


def run():
    module = import_module('src.main')
    module.main()


if __name__ == '__main__':
    run()
