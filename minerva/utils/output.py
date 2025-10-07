from minerva.utils.typing import PathLike
import sys


class Tee:
    def __init__(self, filename: PathLike, mode: str = "w", stream=None):
        self.file = open(filename, mode)
        self.stream = stream or sys.stdout

    def write(self, message: str):
        self.file.write(message)
        self.stream.write(message)

    def flush(self):
        self.file.flush()
        self.stream.flush()

    def close(self):
        self.file.close()
