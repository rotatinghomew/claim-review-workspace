"""测试隔离：每个用例使用独立临时 state 目录，原始 data/ 保持只读。"""
import tempfile
import unittest
from pathlib import Path

from claimdesk import config


class IsolatedStateCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        config.use_state_root(Path(self._tmp.name))
        config.ensure_dirs()

    def tearDown(self):
        self._tmp.cleanup()
