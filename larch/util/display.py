

from .interface_info import ipython_status as _ipython_status
from .rate_limiter import NonBlockingRateLimiter
_message_set = _ipython_status()


class Throttlable():

	def __init__(self, throttle=None):
		if throttle is not None:
			self._throttle = NonBlockingRateLimiter(throttle)
		else:
			self._throttle = True



if 'IPython' in _message_set:
	from IPython.display import display, clear_output, display_html, HTML

	class display_head(Throttlable):

		def __init__(self, text, level=3, throttle=2):
			super().__init__(throttle)
			self.level = level
			self.tag = display(HTML(f'<h{self.level}>{text}</h{self.level}>'), display_id=True)

		def update(self, text, newline=False, force=False):
			if self._throttle or force:
				self.tag.update(HTML(f'<h{self.level}>{text}</h{self.level}>'))

		def __call__(self, *text, **kwargs):
			self.update(*text, **kwargs)

		def linefeed(self):
			pass

	class display_p(Throttlable):

		def __init__(self, text, force=False, throttle=2):
			super().__init__(throttle)
			if isinstance(text, str):
				self.tag = display(HTML(f'<p>{text}</p>'), display_id=True)
			else:
				self.tag = display(text, display_id=True)

		def update(self, text, force=False):
			if self._throttle or force:
				if isinstance(text, str):
					self.tag.update(HTML(f'<p>{text}</p>'))
				else:
					self.tag.update(text)

		def __call__(self, *text, **kwargs):
			self.update(*text, **kwargs)

else:

	class fake_display():

		def __init__(self, *args, **kwargs):
			print(*args)

		def update(self, *args):
			print(*args)

	display = fake_display
	clear_output = lambda *x,**y: None
	display_html = lambda *x,**y: None
	HTML = lambda *x,**y: None

	class display_head(Throttlable):

		def __init__(self, text, level=3, throttle=2):
			super().__init__(throttle)
			self.level = level
			print(f'{text}: ',end="")

		def update(self, text, newline=False, force=False):
			if self._throttle or force:
				if newline:
					print("")
					print(f'{text} ',end="")
				else:
					print(f'{text}: ', end="")

		def __call__(self, *text, **kwargs):
			self.update(*text, **kwargs)

		def linefeed(self):
			print("")

	class display_p(Throttlable):

		def __init__(self, text, force=False, throttle=2):
			super().__init__(throttle)
			if isinstance(text, str):
				print(text)
			else:
				if force:
					print(str(text))

		def update(self, text, force=False):
			if self._throttle or force:
				if isinstance(text, str):
					print(text)
				else:
					if force:
						print(str(text))

		def __call__(self, *text, **kwargs):
			self.update(*text, **kwargs)
