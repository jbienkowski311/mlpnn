from os import getcwd


class File(object):
    def __init__(self, relative_path):
        self.path = getcwd() + relative_path
