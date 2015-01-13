

import sys
from unittest.mock import MagicMock

class Mock(MagicMock):
	@classmethod
	def __getattr__(cls, name):
			return Mock()

