"""Helpers for mirroring script output to stdout and a log file."""

import sys
from pathlib import Path


class TeeLogger:
    """Write log output to both the active terminal and a file."""

    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.error_terminal = sys.stderr
        self.log_file = open(log_file, 'a', buffering=1, encoding='utf-8')
        self.closed = False

    def write(self, message: str):
        self.terminal.write(message)
        if not self.closed:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        if not self.closed:
            self.log_file.flush()

    def close(self):
        if self.closed:
            return
        sys.stdout = self.terminal
        sys.stderr = self.error_terminal
        self.log_file.close()
        self.closed = True


def setup_logging(run_dir: Path, filename: str = 'training.log') -> TeeLogger:
    """Mirror stdout/stderr to a log file under the run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    tee = TeeLogger(run_dir / filename)
    sys.stdout = tee
    sys.stderr = tee
    return tee


__all__ = ['TeeLogger', 'setup_logging']