from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import Iterator


def tokenize(cmdline: str | list[str]) -> Iterator[str]:
    if type(cmdline) == str:
        cmdline = cmdline.split()

    for arg in cmdline:
        if arg == ",":
            yield arg
            continue
        if arg.startswith(","):
            yield ","
            arg = arg[1:]
        if arg.endswith(","):
            yield arg[:-1]
            yield ","
        else:
            yield arg


@dataclass
class PlotData:
    path: str = ""
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))


def parse_args(cmdline: str | list[str]) -> list[PlotData]:
    if type(cmdline) == str:
        cmdline = cmdline.split()
    plots: list[PlotData] = []

    dataset: dict[str, np.ndarray] = {}

    path = ""
    yexpr = "data[:,1]"

    def add_plot():
        plot = PlotData(
            path=path,
            x=dataset[path][:, 0],
            y=eval(yexpr, dict(data=dataset[path], col=lambda c: dataset[path][:, c])),
        )
        plots.append(plot)

    tokens = tokenize(cmdline)
    for token in tokens:
        if token == ",":
            add_plot()
            yexpr = "data[:,1]"

        elif token == "file":
            path = next(tokens)
            if path not in dataset:
                m = np.loadtxt(path, ndmin=2)
                row_numbers = np.arange(m.shape[0])
                dataset[path] = np.insert(m, 0, row_numbers, axis=1)

        elif token == "y":
            yexpr = next(tokens)
            for c in "0123456789":
                yexpr = yexpr.replace(f"${c}", f"data[:,{c}]")

        else:
            print(f"Invalid keyword {repr(token)}", file=sys.stderr)
            sys.exit(1)

    add_plot()

    return plots


def main():
    plots = parse_args(" ".join(sys.argv[1:]))
    for p in plots:
        print(p)
        plt.plot(p.x, p.y, label=p.path)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
