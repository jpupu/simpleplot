import numpy as np
import pytest

from main import (
    CommandLineError,
    PlotSpec,
    build_plots,
    eval_expr,
    load_datas,
    parse_cmdline,
    parse_fmt,
    tokenize,
)


def test_tokenizer_is_passthru_when_no_commas():
    assert list(tokenize(["foo", "bar", "path/file.txt", "with space"])) == [
        "foo",
        "bar",
        "path/file.txt",
        "with space",
    ]


def test_tokenizer_splits_comma_at_end_of_word():
    assert list(tokenize(["foo,"])) == ["foo", ","]


def test_tokenizer_splits_comma_at_start_of_word():
    assert list(tokenize([",foo"])) == [",", "foo"]


def test_tokenizer_keeps_comma_inside_word():
    assert list(tokenize(["foo, bar"])) == ["foo, bar"]


def test_tokenizer_passes_lonely_comma():
    assert list(tokenize([","])) == [","]


def test_parse_fmt():
    assert parse_fmt("") == ("", "", None)
    assert parse_fmt("o") == ("o", "", None)
    assert parse_fmt("--") == ("", "--", None)
    assert parse_fmt("c") == ("", "", "c")
    assert parse_fmt("C2") == ("", "", "C2")
    assert parse_fmt("o-.") == ("o", "-.", None)
    assert parse_fmt("-.C2") == ("", "-.", "C2")
    assert parse_fmt("o-.C2") == ("o", "-.", "C2")
    assert parse_fmt("greenstuff") is None


def test_parse_cmdline_path():
    plots, graph = parse_cmdline("path foo")
    assert plots[0].path == "foo"


def test_parse_cmdline_xexpr():
    plots, graph = parse_cmdline("xexpr c1*2")
    assert plots[0].xexpr == "c1*2"


def test_parse_cmdline_yexpr():
    plots, graph = parse_cmdline("yexpr c1*2")
    assert plots[0].yexpr == "c1*2"


def test_parse_cmdline_linestyle():
    plots, graph = parse_cmdline("linestyle --")
    assert plots[0].linestyle == "--"


def test_parse_cmdline_marker():
    plots, graph = parse_cmdline("marker +")
    assert plots[0].marker == "+"


def test_parse_cmdline_color():
    plots, graph = parse_cmdline("color orange")
    assert plots[0].color == "orange"


def test_parse_cmdline_label():
    plots, graph = parse_cmdline("label Some_curve")
    assert plots[0].label == "Some_curve"


def test_parse_cmdline_fmt():
    plots, _ = parse_cmdline("fmt o--")
    assert plots[0].marker == "o"
    assert plots[0].linestyle == "--"

    plots, _ = parse_cmdline("linestyle -- fmt o")
    assert plots[0].marker == "o"
    assert plots[0].linestyle == ""

    with pytest.raises(CommandLineError):
        parse_cmdline("fmt badly")


def test_parse_cmdline_xticks():
    _, graph = parse_cmdline("--xticks 5")
    assert graph.xticks == 5
    _, graph = parse_cmdline("--xticks 5.5,11")
    assert graph.xticks == [5.5, 11]


def test_parse_cmdline_xlim():
    _, graph = parse_cmdline("--xlim 10 100")
    assert graph.xlim == (10, 100)
    _, graph = parse_cmdline("--xlim - 100")
    assert graph.xlim == (None, 100)
    _, graph = parse_cmdline("--xlim 10 -")
    assert graph.xlim == (10, None)
    _, graph = parse_cmdline("--xlim - -")
    assert graph.xlim == (None, None)


def test_parse_cmdline_ylim():
    _, graph = parse_cmdline("--ylim 10 100")
    assert graph.ylim == (10, 100)
    _, graph = parse_cmdline("--ylim - 100")
    assert graph.ylim == (None, 100)
    _, graph = parse_cmdline("--ylim 10 -")
    assert graph.ylim == (10, None)
    _, graph = parse_cmdline("--ylim - -")
    assert graph.ylim == (None, None)


def test_parse_cmdline_naked_path():
    plots, _ = parse_cmdline("nosuchthing")
    assert plots[0].path == "nosuchthing"


def test_parse_cmdline_multiple_plots():
    plots, _ = parse_cmdline("xexpr c1, xexpr c3")
    assert len(plots) == 2
    assert plots[0].xexpr == "c1"
    assert plots[1].xexpr == "c3"


def test_parse_cmdline_multiple_plots_empty_plots_are_ignored():
    plots, _ = parse_cmdline("xexpr c1, , xexpr c3,")
    assert len(plots) == 2
    assert plots[0].xexpr == "c1"
    assert plots[1].xexpr == "c3"


def test_build_plots_first_plot_inherits_defaults():
    oplots = build_plots([PlotSpec()])
    assert oplots == [PlotSpec.default()]


def test_build_plots_unset_values_are_inherited():
    oplots = build_plots([PlotSpec(xexpr="c2")])
    assert oplots == [PlotSpec.default().replace(xexpr="c2")]


def test_build_plots_inherits_from_previous_plot():
    oplots = build_plots([PlotSpec(xexpr="c2"), PlotSpec(yexpr="c3")])
    assert oplots == [
        PlotSpec.default().replace(xexpr="c2"),
        PlotSpec.default().replace(xexpr="c2", yexpr="c3"),
    ]


@pytest.fixture
def data():
    return np.array([[0, 0, 0], [1, 10, 100], [2, 20, 200]])


def test_evalexpr_exposes_dataframe_as_data(data):
    assert list(eval_expr("data[:,1]", data)) == [0, 10, 20]


def test_evalexpr_exposes_column_zero_as_i(data):
    assert list(eval_expr("i", data)) == [0, 1, 2]


def test_evalexpr_exposes_columns_as_cN(data):
    assert list(eval_expr("c0", data)) == [0, 1, 2]
    assert list(eval_expr("c1", data)) == [0, 10, 20]
    assert list(eval_expr("c2", data)) == [0, 100, 200]


def test_evalexpr_exposes_columns_as_function_col_N(data):
    assert list(eval_expr("col(0)", data)) == [0, 1, 2]
    assert list(eval_expr("col(1)", data)) == [0, 10, 20]
    assert list(eval_expr("col(1+1)", data)) == [0, 100, 200]


def test_evalexpr_exposes_numpy_math_functions(data):
    assert [np.sin(0), np.sin(1), np.sin(2)] == list(eval_expr("sin(c0)", data))


def test_evalexpr_broadcasts_scalar_value(data):
    assert list(eval_expr("42", data)) == [42, 42, 42]


def test_loaddatas_loads_data_from_files():
    datas = load_datas(
        [
            PlotSpec(path="testdata/tens.txt", xexpr="c0", yexpr="c1"),
            PlotSpec(path="testdata/hundreds.txt", xexpr="c0", yexpr="c1"),
        ]
    )
    assert list(datas[0].x) == [0, 1, 2, 3]
    assert list(datas[0].y) == [0, 10, 20, 30]

    assert list(datas[1].x) == [0, 1, 2, 3]
    assert list(datas[1].y) == [0, 100, 200, 300]


def test_loaddatas_applies_xexpr():
    datas = load_datas(
        [
            PlotSpec(path="testdata/tens.txt", xexpr="c0*2", yexpr="c1"),
        ]
    )
    assert list(datas[0].x) == [0, 2, 4, 6]
    assert list(datas[0].y) == [0, 10, 20, 30]


def test_loaddatas_applies_yexpr():
    datas = load_datas(
        [
            PlotSpec(path="testdata/tens.txt", xexpr="c0", yexpr="c1*2"),
        ]
    )
    assert list(datas[0].x) == [0, 1, 2, 3]
    assert list(datas[0].y) == [0, 20, 40, 60]
