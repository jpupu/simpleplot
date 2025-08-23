import dataclasses
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


class CommandLineError(RuntimeError):
    pass


def tokenize(cmdline: str | list[str]) -> Iterator[str]:
    if type(cmdline) is str:
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
    """PlotSpec is a plot spec"""

    path: str | None = None
    xexpr: str | None = None
    yexpr: str | None = None
    linestyle: str | None = None
    marker: str | None = None
    color: str | None = None
    label: str | None = None

    @classmethod
    def default(cls) -> "PlotSpec":
        return PlotSpec(
            path="-",
            xexpr="c0",
            yexpr="c1",
            linestyle="-",
            marker="",
            color=None,
            label="",
        )

    def set(self, name: str, value: str) -> None:
        setattr(self, name, value)

    def setdefault(self, name: str, value: str) -> None:
        if getattr(self, name) is None:
            setattr(self, name, value)

    @classmethod
    def is_field(cls, name: str) -> bool:
        return name in cls.__dict__

    def update(self, other: "PlotSpec"):
        for f in dataclasses.fields(PlotSpec):
            o = getattr(other, f.name)
            if o is not None:
                setattr(self, f.name, o)

    def replace(self, **kwargs) -> "PlotSpec":
        new = self.copy()
        for k, v in kwargs.items():
            new.set(k, v)
        return new

    def copy(self) -> "PlotSpec":
        p = PlotSpec()
        p.update(self)
        return p

    def is_empty(self) -> bool:
        return all(x is None for x in dataclasses.astuple(self))


@dataclass
class PlotData:
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))


def parse_float_or_none(s: str) -> float | None:
    if s == "-" or s == "":
        return None
    return float(s)


MARKER_STYLES = ".,ov^<>12348spP*hH+xXDd|_"
LINE_STYLES = ["--", "-.", "-", ":"]
COLOR_STYLES = list("bgrcmykwa") + [f"C{n}" for n in range(10)]


def parse_fmt(fmt: str) -> tuple[str | None, str | None, str | None] | None:
    # See https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html
    marker = ""
    line = ""
    color = None

    for m in MARKER_STYLES:
        if fmt.startswith(m):
            marker = m
            fmt = fmt[len(m) :]
            break
    for ln in LINE_STYLES:
        if fmt.startswith(ln):
            line = ln
            fmt = fmt[len(ln) :]
            break
    for c in COLOR_STYLES:
        if fmt.startswith(c):
            color = c
            fmt = fmt[len(c) :]
            break

    if fmt:
        return None

    return marker, line, color


def parse_cmdline(cmdline: str | list[str]) -> tuple[list[PlotSpec], GraphSpec]:
    """Parses command line to sparse plot specs and graph spec."""
    graph = GraphSpec()
    plots: list[PlotSpec] = []
    plot: PlotSpec = PlotSpec()

    tokens = tokenize(cmdline)
    for token in tokens:
        if token in (",", "plot"):
            plots.append(plot)
            plot = PlotSpec()
        elif PlotSpec.is_field(token):
            plot.set(token, next(tokens))
        elif token == "fmt":
            fmt = next(tokens)
            mlc = parse_fmt(fmt)
            if mlc is None:
                raise CommandLineError(f"Invalid format: {fmt}")
            plot.marker, plot.linestyle, plot.color = mlc
        # elif token == "scatter":
        #     plot.setdefault("xexpr", "c1")
        #     plot.setdefault("yexpr", "c2")
        #     plot.setdefault("linestyle", "")
        #     plot.setdefault("marker", "o")
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
        else:
            plot.path = token

    if plot:
        plots.append(plot)

    plots = [p for p in plots if not p.is_empty()]

    return plots, graph


def build_plots(attrlist: list[PlotSpec]) -> list[PlotSpec]:
    """Build concrete PlotSpecs from sparse specs."""
    plots: list[PlotSpec] = []
    plot = PlotSpec.default()

    for attrs in attrlist:
        plot.update(attrs)
        plots.append(plot.copy())

    return plots


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
    specs, graph = parse_cmdline(sys.argv[1:])
    specs = build_plots(specs)

    datas = load_datas(specs)

    ax = plt.subplot()
    for spec, data in zip(specs, datas, strict=True):
        ax.plot(
            data.x,
            data.y,
            label=f"{spec.path}:{spec.yexpr}" if spec.label is None else spec.label,
            ls=spec.linestyle,
            marker=spec.marker,
            color=spec.color,
        )
    if graph.xticks is not None:
        if type(graph.xticks) is list:
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
    except CommandLineError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
