from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_array_equal


@dataclass
class DataFrame:
    columns: dict[str, np.ndarray]

    @property
    def rows(self) -> int:
        """Number of rows"""
        return len(next(iter(self.columns.values())))

    @property
    def cols(self) -> int:
        """Number of columns"""
        return len(self.columns)

    def col(self, index: int | str) -> np.ndarray:
        if isinstance(index, int):
            return list(self.columns.values())[index]
        return self.columns[index]


class TestDataFrame:
    @pytest.fixture
    def cats(self):
        return DataFrame(
            {
                "name": ["Alex", "Banjo"],
                "age": [4, 7],
                "tail": [15, 16],
            }
        )

    def test_rows(self, cats):
        assert cats.rows == 2

    def test_cols(self, cats):
        assert cats.cols == 3

    def test_index_column_by_number(self, cats):
        assert_array_equal(cats.col(1), [4, 7])

    def test_index_column_by_name(self, cats):
        assert_array_equal(cats.col("age"), [4, 7])


def _detect_type(s: str):
    try:
        int(s)
        return int
    except ValueError:
        try:
            float(s)
            return float
        except ValueError:
            return str


def load_file(file: str | list[str]) -> DataFrame:
    """Loads file into a DataFrame.
    `file` is a filename or a list of lines.
    """

    if isinstance(file, str):
        with open(file) as fp:
            file = fp.readlines()

    # Infer delimiter
    line0 = file[0]
    delims = [" ", "\t", ",", ";"]
    delim_counts = [line0.count(x) for x in delims]
    delim = sorted(zip(delim_counts, delims, strict=True))[-1][1]

    # Infer column names
    fields = file[0].split(delim)
    if all(_detect_type(f) is str for f in fields):
        names = fields
        file = file[1:]
        line0 = file[0]
    else:
        names = [f"column{i + 1}" for i in range(len(fields))]

    # Infer types
    fields = file[0].split(delim)
    types = [_detect_type(f) for f in fields]

    # Read data
    coldata = [[] for _ in types]
    for line in file:
        if line.strip() == "":
            continue
        fields = line.split(delim)
        for i, (f, t) in enumerate(zip(fields, types, strict=True)):
            try:
                coldata[i].append(t(f))
            except ValueError:
                # Invalid value, re-detect column type.
                types[i] = _detect_type(f)
                coldata[i].append(types[i](f))

    # Add row numbers
    types = [int] + types
    names = ["rowno"] + names
    coldata = [np.arange(len(coldata[0]))] + coldata

    # Convert to numpy and DataFrame
    columns = [np.array(d, dtype=t) for d, t in zip(coldata, types, strict=True)]
    return DataFrame(dict(zip(names, columns, strict=True)))


def test_automatic_column_names():
    df = load_file(["11 12 13", "21 22 23"])
    assert list(df.columns.keys()) == ["rowno", "column1", "column2", "column3"]


def test_automatic_row_numbers():
    df = load_file(["11 12 13", "21 22 23"])
    assert_array_equal(df.col(0), [0, 1])
    assert_array_equal(df.col("rowno"), [0, 1])


def test_numeric():
    df = load_file(["11 12 13", "21 22 23"])
    assert df.cols == 4
    assert df.rows == 2
    assert_array_equal(df.col(1), [11, 21])
    assert_array_equal(df.col(2), [12, 22])
    assert_array_equal(df.col(3), [13, 23])


def test_string_on_first_row():
    df = load_file(["11 foo xyz", "21 bar 23"])
    assert df.cols == 4
    assert df.rows == 2
    assert_array_equal(df.col(1), [11, 21])
    assert_array_equal(df.col(2), ["foo", "bar"])
    assert_array_equal(df.col(3), ["xyz", "23"])


def test_string_on_second_row():
    df = load_file(["11 12 13", "21 foo 23"])
    assert df.cols == 4
    assert df.rows == 2
    assert_array_equal(df.col(1), [11, 21])
    assert_array_equal(df.col(2), ["12", "foo"])
    assert_array_equal(df.col(3), [13, 23])


def test_delimiter_detection():
    # more spaces
    df = load_file(["1 2,3 4,5 6"])
    assert_array_equal(df.col(1), [1])
    assert_array_equal(df.col(2), ["2,3"])
    assert_array_equal(df.col(3), ["4,5"])
    assert_array_equal(df.col(4), 6)

    # more commas
    df = load_file(["1 2,3,4,5 6"])
    assert_array_equal(df.col(1), ["1 2"])
    assert_array_equal(df.col(2), [3])
    assert_array_equal(df.col(3), [4])
    assert_array_equal(df.col(4), ["5 6"])

    # commas win on even match
    df = load_file(["1,2 3"])
    assert_array_equal(df.col(1), [1])
    assert_array_equal(df.col(2), ["2 3"])


def test_header_detection():
    df = load_file(["first second third", "11 12 13", "21 22 23"])
    assert list(df.columns.keys()) == ["rowno", "first", "second", "third"]
    assert df.cols == 4
    assert df.rows == 2
    assert_array_equal(df.col(1), [11, 21])
    assert_array_equal(df.col(2), [12, 22])
    assert_array_equal(df.col(3), [13, 23])


def test_line_endings_and_empty_lines():
    df = load_file(["11 12 13\n", "   \n", "21 22 23\n", "\n"])
    assert df.cols == 4
    assert df.rows == 2
    assert_array_equal(df.col(1), [11, 21])
    assert_array_equal(df.col(2), [12, 22])
    assert_array_equal(df.col(3), [13, 23])
