from xdiff.printlib import formatter

def test_print_report_renders_each_comparison(monkeypatch):
    rendered = []
    report = ["first", "second"]

    monkeypatch.setattr(formatter, "print_comparison", rendered.append)

    formatter.print_report(report)

    assert rendered == report
