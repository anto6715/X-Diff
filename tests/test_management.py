from types import SimpleNamespace

from nccompare import management

def test_start_from_command_line_interface_executes_and_renders(monkeypatch):
    args = SimpleNamespace(folder1="ref", folder2="cmp")
    report = object()
    rendered = []

    monkeypatch.setattr(management, "get_args", lambda: args)
    monkeypatch.setattr(management.core, "execute", lambda **kwargs: report)
    monkeypatch.setattr(management.formatter, "print_report", rendered.append)

    management.start_from_command_line_interface()

    assert rendered == [report]
