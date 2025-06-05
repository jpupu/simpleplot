from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
import sys


@dataclass
class PlotData:
    path: str = ""
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))


def parse_args(cmdline: list[str]) -> list[PlotData]:
    plots: list[PlotData] = []

    dataset: dict[str, np.ndarray] = {}

    path = ""
    cmdline = cmdline.replace("\\,", "<ESCAPECOMMA>")
    for part in cmdline.split(","):
        part = part.replace("<ESCAPECOMMA>", ",")
        yexpr = "data[:,1]"

        it = iter(part.split())
        for token in it:
            if token == "file":
                path = next(it)
                if path not in dataset:
                    m = np.loadtxt(path, ndmin=2)
                    row_numbers = np.arange(m.shape[0])
                    dataset[path] = np.insert(m, 0, row_numbers, axis=1)
            elif token == "y":
                yexpr = next(it)
                for c in "0123456789":
                    yexpr = yexpr.replace(f"${c}", f"data[:,{c}]")
            else:
                print(f"Invalid keyword {repr(token)}", file=sys.stderr)
                sys.exit(1)

        plot = PlotData(
            path=path,
            x=dataset[path][:, 0],
            y=eval(yexpr, dict(data=dataset[path], col=lambda c: dataset[path][:, c])),
        )

        plots.append(plot)

    return plots


def main():
    plots = parse_args(" ".join(sys.argv[1:]))
    for p in plots:
        print(p)
        plt.plot(p.x, p.y, label=p.path)
    plt.legend()
    plt.show()


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
    plots = parse_args("file testdata/tens.txt y data[:\\,1], y col(1), y $1")
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


if __name__ == "__main__":
    main()
