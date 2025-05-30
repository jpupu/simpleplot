def main():
    print("Hello from simpleplot!")


def parse_args(cmdline):
    return []


def test_multifile():
    """Multiple files"""
    plots = parse_args("file sin.txt, file cos.txt")
    assert plots[0].path == "sin.txt"
    assert plots[1].path == "cos.txt"


if __name__ == "__main__":
    main()
