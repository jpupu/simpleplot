import dataclasses
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
class GraphSpec:
    xticks: float | list[float] | None = None
    xlim: (float | None, float | None) = (None, None)
    ylim: (float | None, float | None) = (None, None)


@dataclass
class PlotSpec:
    path: str = ""
    xexpr: str = "c0"
    yexpr: str = "c1"
    linestyle: str = "-"
    marker: str = ""
    label: str | None = None


@dataclass
class PlotData:
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))


def parse_float_or_none(s: str) -> float | None:
    if s == "-" or s == "":
        return None
    return float(s)


def parse_specs(cmdline: str | list[str]) -> tuple[GraphSpec, list[PlotSpec]]:
    if type(cmdline) == str:
        cmdline = cmdline.split()
    graph = GraphSpec()
    plots: list[PlotSpec] = []
    reset_plot: list[str] = []
    reset_file: list[str] = []

    spec = dataclasses.replace(PlotSpec(), path="stdin")

    tokens = tokenize(cmdline)
    for token in tokens:
        if token == ",":
            plots.append(spec)
            spec = dataclasses.replace(
                spec, **dict((key, getattr(PlotSpec, key)) for key in reset_plot)
            )
        elif token == "file":
            spec = dataclasses.replace(
                spec, **dict((key, getattr(PlotSpec, key)) for key in reset_file)
            )
            spec.path = next(tokens)
        elif token == "x":
            spec.xexpr = next(tokens)
        elif token == "y":
            spec.yexpr = next(tokens)
        elif token == "ls":
            spec.linestyle = next(tokens)
        elif token == "marker":
            spec.marker = next(tokens)
        elif token == "label":
            spec.label = next(tokens)
        elif token == "--xticks":
            s = next(tokens)
            if "," in s:
                graph.xticks = [float(x) for x in s.split(",")]
            else:
                graph.xticks = float(s)
        elif token == "--xlim":
            graph.xlim = (
                parse_float_or_none(next(tokens)),
                parse_float_or_none(next(tokens)),
            )
        elif token == "--ylim":
            graph.ylim = (
                parse_float_or_none(next(tokens)),
                parse_float_or_none(next(tokens)),
            )
        elif token == "--reset-plot":
            rlist = next(tokens)
            for k, v in [("x", "xexpr"), ("y", "yexpr"), ("ls", "linestyle")]:
                rlist = rlist.replace(k, v)
            reset_plot = rlist.split(",")
        elif token == "--reset-file":
            rlist = next(tokens)
            for k, v in [("x", "xexpr"), ("y", "yexpr"), ("ls", "linestyle")]:
                rlist = rlist.replace(k, v)
            reset_file = rlist.split(",")
        else:
            print(f"Invalid keyword {repr(token)}", file=sys.stderr)
            sys.exit(1)

    plots.append(spec)

    return graph, plots


def eval_expr(expr: str, data: np.ndarray) -> np.ndarray:
    locals = dict(data=data, i=data[:, 0], col=lambda c: data[:, c])
    for i in range(0, min(10, data.shape[1])):
        locals[f"c{i}"] = data[:, i]

    # Evaluate
    result = eval(expr, np.__dict__, locals)

    # Broadcast if scalar
    if np.isscalar(result):
        result = np.ones(data.shape[0]) * result

    return result


def load_datas(specs: list[PlotSpec]) -> list[PlotData]:
    def load_file(path: str) -> np.ndarray:
        if path == "stdin":
            path = sys.stdin
        m = np.loadtxt(path, ndmin=2)
        row_numbers = np.arange(m.shape[0])
        return np.insert(m, 0, row_numbers, axis=1)

    def build_data(spec: PlotSpec) -> PlotData:
        data = dataset[spec.path]
        return PlotData(
            x=eval_expr(spec.xexpr, data),
            y=eval_expr(spec.yexpr, data),
        )

    paths = {s.path for s in specs}
    dataset = {p: load_file(p) for p in paths}

    return [build_data(spec) for spec in specs]


class AutoMultipleLocator(ticker.MultipleLocator):
    def __call__(self):
        ticks = super().__call__()
        while len(ticks) > self.axis.get_tick_space():
            ticks = ticks[::2]
        return ticks


def main():
    graph, specs = parse_specs(sys.argv[1:])

    datas = load_datas(specs)

    ax = plt.subplot()
    for spec, data in zip(specs, datas):
        ax.plot(
            data.x,
            data.y,
            label=f"{spec.path}:{spec.yexpr}" if spec.label is None else spec.label,
            ls=spec.linestyle,
            marker=spec.marker,
        )
    if graph.xticks is not None:
        if type(graph.xticks) == list:
            ax.set_xticks(graph.xticks)
        else:
            ax.xaxis.set_major_locator(AutoMultipleLocator(base=graph.xticks))
    ax.set_xlim(*graph.xlim)
    ax.set_ylim(*graph.ylim)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"interrupted", file=sys.stderr)
