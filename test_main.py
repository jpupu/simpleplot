from main import eval_expr, parse_specs, tokenize, load_datas
import numpy as np
import pytest


def test_tokenizer_is_passthru_when_no_commas():
    assert ["foo", "bar", "path/file.txt", "with space"] == list(
        tokenize(["foo", "bar", "path/file.txt", "with space"])
    )


def test_tokenizer_splits_comma_at_end_of_word():
    assert ["foo", ","] == list(tokenize(["foo,"]))


def test_tokenizer_splits_comma_at_end_of_word():
    assert ["foo", ","] == list(tokenize(["foo,"]))


def test_tokenizer_splits_comma_at_start_of_word():
    assert [",", "foo"] == list(tokenize([",foo"]))


def test_tokenizer_keeps_comma_inside_word():
    assert ["foo, bar"] == list(tokenize(["foo, bar"]))


def test_tokenizer_passes_lonely_comma():
    assert [","] == list(tokenize([","]))


def test_parsespecs_gets_specified_filenames():
    _, specs = parse_specs("file foo, file bar, file baz")
    assert ["foo", "bar", "baz"] == [s.path for s in specs]


def test_parsespecs_uses_previous_filename_when_not_specified():
    _, specs = parse_specs("file foo, , , file baz")
    assert ["foo", "foo", "foo", "baz"] == [s.path for s in specs]


def test_parsespecs_get_specified_yexprs():
    _, specs = parse_specs("y sin(c1*0.5)+2")
    assert ["sin(c1*0.5)+2"] == [s.yexpr for s in specs]


def test_parsespecs_uses_column_one_when_yexpr_not_specified():
    _, specs = parse_specs("y some,")
    assert ["some", "c1"] == [s.yexpr for s in specs]


def test_parsespecs_get_specified_xexprs():
    _, specs = parse_specs("x sin(c1*0.5)+2")
    assert ["sin(c1*0.5)+2"] == [s.xexpr for s in specs]


def test_parsespecs_uses_column_zero_when_xexpr_not_specified():
    _, specs = parse_specs("x some,")
    assert ["some", "c0"] == [s.xexpr for s in specs]


@pytest.fixture
def data():
    return np.array([[0, 0, 0], [1, 10, 100], [2, 20, 200]])


def test_evalexpr_exposes_dataframe_as_data(data):
    assert [0, 10, 20] == list(eval_expr("data[:,1]", data))


def test_evalexpr_exposes_column_zero_as_i(data):
    assert [0, 1, 2] == list(eval_expr("i", data))


def test_evalexpr_exposes_columns_as_cN(data):
    assert [0, 1, 2] == list(eval_expr("c0", data))
    assert [0, 10, 20] == list(eval_expr("c1", data))
    assert [0, 100, 200] == list(eval_expr("c2", data))


def test_evalexpr_exposes_columns_as_function_col_N(data):
    assert [0, 1, 2] == list(eval_expr("col(0)", data))
    assert [0, 10, 20] == list(eval_expr("col(1)", data))
    assert [0, 100, 200] == list(eval_expr("col(1+1)", data))


def test_evalexpr_exposes_numpy_math_functions(data):
    assert [np.sin(0), np.sin(1), np.sin(2)] == list(eval_expr("sin(c0)", data))


def test_evalexpr_broadcasts_scalar_value(data):
    assert [42, 42, 42] == list(eval_expr("42", data))


def test_loaddatas_loads_data_from_files():
    _, specs = parse_specs("file testdata/tens.txt, file testdata/hundreds.txt")
    datas = load_datas(specs)
    assert [0, 1, 2, 3] == list(datas[0].x)
    assert [0, 10, 20, 30] == list(datas[0].y)

    assert [0, 1, 2, 3] == list(datas[1].x)
    assert [0, 100, 200, 300] == list(datas[1].y)


def test_loaddatas_applies_xexpr():
    _, specs = parse_specs("file testdata/tens.txt x c0*2")
    datas = load_datas(specs)
    assert [0, 2, 4, 6] == list(datas[0].x)
    assert [0, 10, 20, 30] == list(datas[0].y)


def test_loaddatas_applies_yexpr():
    _, specs = parse_specs("file testdata/tens.txt y c1*2")
    datas = load_datas(specs)
    assert [0, 1, 2, 3] == list(datas[0].x)
    assert [0, 20, 40, 60] == list(datas[0].y)
