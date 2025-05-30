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

    for part in cmdline.split(","):
        current = PlotData()
        it = iter(part.split())
        for token in it:
            if token == "file":
                current.path = next(it)
                current.y = np.loadtxt(current.path)
                current.x = np.arange(0, len(current.y))
            else:
                print(f"Invalid keyword {repr(token)}", file=sys.stderr)
                sys.exit(1)
        plots.append(current)

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


if __name__ == "__main__":
    main()
