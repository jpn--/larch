
# This file is under the Creative Commons Attribution Share Alike license
# https://creativecommons.org/licenses/by-sa/3.0/
# original source:
# https://stackoverflow.com/questions/20643184/using-python-threads-to-make-thousands-of-calls-to-a-slow-api-with-a-rate-limit

from collections.abc import Iterator
from threading import Lock
import time
import functools


class BlockingRateLimiter(Iterator):
	"""Iterator that yields a value at most once every 'interval' seconds."""
	def __init__(self, interval):
		self.lock = Lock()
		self.interval = interval
		self.next_yield = 0

	def __next__(self):
		with self.lock:
			t = time.monotonic()
			if t < self.next_yield:
				time.sleep(self.next_yield - t)
				t = time.monotonic()
			self.next_yield = t + self.interval


class NonBlockingRateLimiter():
	def __init__(self, interval):
		self.lock = Lock()
		self.interval = interval
		self.next_greenlight = 0

	def __call__(self, fn):
		@functools.wraps(fn)
		def decorated(*args, **kwargs):
			t = time.monotonic()
			if t >= self.next_greenlight:
				with self.lock:
					self.next_greenlight = t + self.interval
					fn(*args, **kwargs)
		return decorated

	def __bool__(self):
		t = time.monotonic()
		if t >= self.next_greenlight:
			with self.lock:
				self.next_greenlight = t + self.interval
				return True
		return False


_global_rate_limiters = {}

def GlobalRateLimiter(tag, interval=1, wait_now=True):
	global _global_rate_limiters
	if tag not in _global_rate_limiters:
		_global_rate_limiters[tag] = RateLimiter(interval)
	if wait_now:
		return next(_global_rate_limiters[tag])
