import numpy as np
import io


class HexIO:
    def __init__(self, data = b""):
        self.data = bytearray(data)
        self.ptr = 0

    def write_uint(self, x, size):
        self.data.extend(x.to_bytes(size, byteorder = "little"))

    def write_uint8(self, x):
        self.write_uint(x, 1)

    def write_uint16(self, x):
        self.write_uint(x, 2)

    def write_uint32(self, x):
        self.write_uint(x, 4)

    def write_bin8(self, string):
        self.write_uint8(int(string, 2))

    def write_numpy(self, array):
        buffer = io.BytesIO()
        np.save(buffer, array)
        data = buffer.getvalue()

        self.write_uint32(len(data))
        self.data.extend(data)

    def ptr_valid(self):
        return self.ptr < len(self.data)

    def read(self, size):
        segment = self.data[self.ptr:self.ptr + size]
        self.ptr += size
        return segment

    def read_uint(self, size):
        return int.from_bytes(self.read(size), byteorder = "little")

    def read_uint8(self):
        return self.read_uint(1)

    def read_uint16(self):
        return self.read_uint(2)

    def read_uint32(self):
        return self.read_uint(4)

    def read_bin8(self):
        return format(self.read_uint8(), "08b")

    def read_numpy(self):
        data = self.read(self.read_uint32())
        buffer = io.BytesIO(data)
        return np.load(buffer)