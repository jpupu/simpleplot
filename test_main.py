from main import parse_args, tokenize


def test_tokenizer_basic():
    assert ["foo", "bar", "path/file.txt"] == list(
        tokenize("foo bar path/file.txt".split())
    )


def test_tokenizer_comma():
    # Word-ending comma is a separator
    assert ["foo", ",", "bar"] == list(tokenize(["foo,", "bar"]))
    # Word-starting comma is a separator
    assert ["foo", ",", "bar"] == list(tokenize(["foo", ",bar"]))
    # Both work together
    assert ["foo", ",", "bar", ","] == list(tokenize(["foo", ",bar,"]))
    # Midword comma is NOT a separator
    assert ["foo,bar"] == list(tokenize(["foo,bar"]))
    # A lonely comma is just a separator
    assert ["foo", ",", "bar"] == list(tokenize(["foo", ",", "bar"]))


def test_file_path():
    plots = parse_args("file testdata/tens.txt")
    assert ["testdata/tens.txt"] == [p.path for p in plots]


def test_multiple_files():
    plots = parse_args("file testdata/tens.txt, file testdata/hundreds.txt")
    assert ["testdata/tens.txt", "testdata/hundreds.txt"] == [p.path for p in plots]


def test_data():
    plots = parse_args("file testdata/tens.txt")
    assert [0, 1, 2, 3] == list(plots[0].x)
    assert [0, 10, 20, 30] == list(plots[0].y)


def test_data_y_expressions_column_references():
    plots = parse_args("file testdata/tens.txt y data[:,1], y col(1), y $1")
    assert [0, 1, 2, 3] == list(plots[0].x)
    assert [0, 10, 20, 30] == list(plots[0].y)

    assert [0, 1, 2, 3] == list(plots[1].x)
    assert [0, 10, 20, 30] == list(plots[1].y)

    assert [0, 1, 2, 3] == list(plots[2].x)
    assert [0, 10, 20, 30] == list(plots[2].y)


def test_data_y_expressions_math():
    plots = parse_args("file testdata/tens.txt y $0*2+1")
    assert [0, 1, 2, 3] == list(plots[0].x)
    assert [1, 3, 5, 7] == list(plots[0].y)


def test_multifile_data():
    plots = parse_args("file testdata/tens.txt, file testdata/hundreds.txt")
    assert [0, 1, 2, 3] == list(plots[0].x)
    assert [0, 10, 20, 30] == list(plots[0].y)

    assert [0, 1, 2, 3] == list(plots[1].x)
    assert [0, 100, 200, 300] == list(plots[1].y)


def test_multicolumn_data():
    plots = parse_args("file testdata/multi.txt y $1, y $2, y $3")
    assert [0, 1, 2, 3] == list(plots[0].x)
    assert [0, 10, 20, 30] == list(plots[0].y)

    assert [0, 1, 2, 3] == list(plots[1].x)
    assert [30, 20, 10, 0] == list(plots[1].y)

    assert [0, 1, 2, 3] == list(plots[2].x)
    assert [10, 15, 20, 25] == list(plots[2].y)
