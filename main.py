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


if __name__ == "__main__":
    main()
