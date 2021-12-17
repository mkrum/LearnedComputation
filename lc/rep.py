import itertools
from typing import List, Dict
from dataclasses import dataclass, field
from bitstring import BitArray

import torch
import numpy as np


def split_expression_into_tokens(exp):
    unpacked = []
    for e in exp:
        unpacked.extend(list(str(e)))
    return unpacked


class EnumRep:
    """
    Simple enumeration based representation.

    Manages a representation that maps a list of strings to unique integer
    values.

    >>> class MyEnumRep(EnumRep): _str_to_int = {"a": 0}; _int_to_str = {0: "a"}
    >>> x = MyEnumRep.from_str('a')
    >>> x.to_str()
    "a"
    >>> x.to_int()
    0
    >>> MyEnumRep.from_int(x.to_int())
    "a"
    """

    _str_to_int: Dict[str, int] = None
    _int_to_str: Dict[int, str] = None

    def __init__(self, str_value: str):
        """
        Initializer, use from_str or from_int instead.
        """
        assert isinstance(str_value, str)
        self._str_rep = str_value

    def __eq__(self, other):
        return self._str_rep == other._str_rep

    @classmethod
    def valid_str(cls) -> List[str]:
        """List of valid strings"""
        return list(cls._str_to_int.keys())

    @classmethod
    def valid_int(cls) -> List[int]:
        """List of valid integers"""
        return list(cls._str_to_int.values())

    @classmethod
    def sample(cls):
        """Randomly sample from the valid values"""
        str_value = np.random.choice(cls.valid_str())
        return cls(str_value)

    @classmethod
    def from_str(cls, str_value: str):
        """Create representation from a string value"""

        if str_value not in cls._str_to_int.keys():
            raise ValueError(f"{str_value} is not a valid piece string.")

        return cls(str_value)

    def to_str(self) -> str:
        """Convert representation into a string"""
        return self._str_rep

    def __str__(self) -> str:
        return self.to_str()

    @classmethod
    def from_int(cls, int_value: int):
        """Create representation from an integer value"""

        if int_value not in cls._int_to_str.keys():
            raise ValueError(f"{int_value} is not a valid piece int.")

        piece_str = cls._int_to_str[int_value]
        return cls(piece_str)

    def to_int(self) -> int:
        """Convert representation into a integer"""
        return self._str_to_int[self._str_rep]

    def __int__(self) -> int:
        return self.to_int()

    @classmethod
    def size(cls):
        return len(cls._int_to_str.keys())


class ListEnum:
    """
    Representation for a list of EnumReps

    >>> class MyEnum(EnumRep):
        _str_to_int = {"a": 0, "b": 1}
        _int_to_str = {0: "a", 1: "b"}
    >>> class MyListEnum(ListEnum):
        token_type = MyEnum
    >>> x = MyListEnum.from_str_list(["a", "b"])
    >>> x.to_int_list()
    [0, 1]
    >>> x.to_str_list()
    ["a", "b"]
    >>> x.to_tensor()
    tensor([0, 1])
    >>> x.from_tensor(x.to_tensor()).to_str_list()
    ["a", "b"]
    """

    token_type: EnumRep = None

    def __init__(self, list_of_values: List[EnumRep]):
        self._values = list_of_values

    def __eq__(self, other):
        return self._values == other._values

    def __len__(self):
        return len(self._values)

    @classmethod
    def sample(cls, n):
        return cls([cls.token_type.sample() for _ in range(n)])

    @classmethod
    def from_str_list(cls, list_vals: List[str]):
        return cls(list(map(lambda x: cls.token_type.from_str(x), list_vals)))

    def to_str_list(self) -> List[str]:
        return list(map(lambda x: x.to_str(), self._values))

    @classmethod
    def from_int_list(cls, list_vals):
        return cls(list(map(lambda x: cls.token_type.from_int(x), list_vals)))

    def to_int_list(self) -> List:
        return list(map(lambda x: x.to_int(), self._values))

    @classmethod
    def from_numpy(cls, arr):
        return cls(list(map(lambda x: cls.token_type.from_int(x), arr)))

    def to_numpy(self) -> np.ndarray:
        return np.array(self.to_int_list())

    @classmethod
    def from_tensor(cls, tensor_vals):
        list_vals = list(map(lambda x: x.item(), tensor_vals))
        return cls.from_int_list(list_vals)

    def to_tensor(self) -> torch.LongTensor:
        return torch.LongTensor(self.to_int_list())


class TupleEnum(ListEnum):
    """
    A tuple of EnumRep

    Implements the same behavior of a ListEnum, but requires the list to always
    be a specific size. Only supports single type tuples.
    """

    length = None
    token_type = None

    def __init__(self, list_of_values: List[EnumRep]):
        assert len(list_of_values) == self.length
        super().__init__(list_of_values)

    @classmethod
    def sample(cls):
        return cls([cls.token_type.sample() for _ in range(cls.length)])

    @classmethod
    def shape(cls):
        return (cls.length, cls.token_type.size())

    @classmethod
    def width(cls):
        return cls.token_type.size()


class MathToken(EnumRep):
    _tokens = list(map(str, range(10))) + ["+", "-", "<start>", "<stop>"]

    _str_to_int = {t: i for (i, t) in enumerate(_tokens)}
    _int_to_str = {i: t for (t, i) in _str_to_int.items()}


class ExpressionRep(ListEnum):
    token_type = MathToken

    @classmethod
    def from_str_list(cls, str_val):
        str_val = split_expression_into_tokens(str_val)
        return super().from_str_list(str_val)

    @classmethod
    def from_str(cls, str_val):
        str_val = ["<start>"] + list(str_val) + ["<stop>"]
        return super().from_str_list(str_val)

    def to_str(cls):
        return "".join(map(lambda x: x.to_str(), cls._values))

    @classmethod
    def parse(cls, tokens):
        stop_ind = -1
        if tokens[0] != "<start>":
            return None

        tokens = tokens[1:]

        if "<stop>" not in tokens:
            return None

        stop_ind = tokens.index("<stop>")
        tokens = tokens[:stop_ind]

        str_value = "".join(tokens)
        try:
            return int(str_value)
        except ValueError:
            return None

class ExpressionRepOneHot(ExpressionRep):

    @classmethod
    def size(self):
        return 14

    def to_tensor(self):
        tensor = super().to_tensor()
        out = torch.nn.functional.one_hot(tensor, num_classes=14)
        return out.float()
    
    @classmethod
    def from_expression_tensor(cls, output):
        out = torch.nn.functional.one_hot(output, num_classes=14)
        return out.float()

class BinaryOutputToken(EnumRep):
    _tokens = ["0", "1", "<start>", "<stop>"]

    _str_to_int = {t: i for (i, t) in enumerate(_tokens)}
    _int_to_str = {i: t for (t, i) in _str_to_int.items()}


class BinaryOutputRep(ListEnum):
    token_type = BinaryOutputToken

    @classmethod
    def from_str(cls, str_val):
        y_vals = list(BitArray(int=int(str_val), length=9).bin)
        raw_y = ["<start>"] + y_vals + ["<stop>"]
        return cls.from_str_list(raw_y)


class BinaryVectorRep8bit:

    symbols = ["+", "-", "<start>", "<stop>"]
    n_bits = 8

    def __init__(self, vec):
        self.vec = vec

    @classmethod
    def size(cls):
        return cls.n_bits + len(cls.symbols)

    @classmethod
    def from_str(cls, val):
        x = torch.zeros(cls.n_bits + len(cls.symbols))

        if val in cls.symbols:
            val_idx = cls.symbols.index(val)
            x[cls.n_bits + val_idx] = 1.0
            return x

        try:
            val = int(val)
        except ValueError:
            print(f"Invalid value? {val}")

        vals = BitArray(int=val, length=cls.n_bits)

        for (i, v) in enumerate(vals.bin):
            x[i] = float(v)

        return x

    @classmethod
    def from_str_list(cls, str_list):
        vecs = []
        for x in str_list:
            vecs.append(cls.from_str(x))
        return cls(torch.stack(vecs))

    def to_tensor(self):
        return self.vec


class FloatRep:

    symbols = ["+", "-"]
    n_bits = 1

    def __init__(self, vec):
        self.vec = vec

    @classmethod
    def size(cls):
        return cls.n_bits + len(cls.symbols)

    @classmethod
    def from_str(cls, val):
        x = torch.zeros(cls.n_bits + len(cls.symbols))

        if val in cls.symbols:
            val_idx = cls.symbols.index(val)
            x[cls.n_bits + val_idx] = 1.0
            return x

        try:
            val = float(val)
        except ValueError:
            print(f"Invalid value? {val}")

        x[0] = float(val)
        return x

    @classmethod
    def from_str_list(cls, str_list):
        vecs = []
        for x in split_expression_into_tokens(str_list):
            vecs.append(cls.from_str(x))
        return cls(torch.stack(vecs))

    def to_tensor(self):
        return self.vec

    @classmethod
    def from_expression_tensor(cls, output):
        out = torch.nn.functional.one_hot(output, num_classes=14)
        return out.float()
