from xdiff import management


def test_start_from_command_line_interface_executes_and_renders(monkeypatch):
    called = []

    monkeypatch.setattr(management, "cli", lambda: called.append(True))

    management.start_from_command_line_interface()

    assert called == [True]
