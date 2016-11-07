

import sys
from unittest.mock import MagicMock

class Mock(MagicMock):
	@classmethod
	def __getattr__(cls, name):
		if name=='_mock_methods':
			return super().__getattr__(cls, name)
		else:
			return Mock()
