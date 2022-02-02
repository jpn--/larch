import unicodedata
from difflib import SequenceMatcher as _SequenceMatcher
from heapq import nlargest as _nlargest
from pathlib import Path


def strip_accents(s):
	return ''.join(c for c in unicodedata.normalize('NFD', s)
	               if unicodedata.category(c) != 'Mn')


def supercasefold(s):
	return strip_accents(s).casefold()


def case_insensitive_close_matches(word, possibilities, n=3, cutoff=0.6, excpt=None):
	"""Get a list of the best "good enough" matches.

	Parameters
	----------
	word : str
		A base string for which close matches are desired
	possibilities : Collection[str]
	 	Word list against which to match word.
	n : int, default 3
		Maximum number of close matches to return,  must be > 0.
	cutoff : float, default 0.6
		A float in the range [0, 1].  Possibilities
		that don't score at least that similar to `word` are ignored.

	The best (no more than n) matches among the possibilities are returned
	in a list, sorted by similarity score, most similar first.

	Examples
	--------
	>>> case_insensitive_close_matches("appel", ["ape", "apple", "peach", "puppy"])
	['apple', 'ape']
	>>> import keyword
	>>> case_insensitive_close_matches("wheel", keyword.kwlist)
	['while']
	>>> case_insensitive_close_matches("apples", keyword.kwlist)
	[]
	>>> case_insensitive_close_matches("Accept", keyword.kwlist)
	['except']
	>>> case_insensitive_close_matches("NonLocal", keyword.kwlist)
	['nonlocal']
	"""

	if not n > 0:
		raise ValueError("n must be > 0: %r" % (n,))
	if not 0.0 <= cutoff <= 1.0:
		raise ValueError("cutoff must be in [0.0, 1.0]: %r" % (cutoff,))
	result = []
	s = _SequenceMatcher()
	s.set_seq2(supercasefold(word))
	for x in possibilities:
		x_ = supercasefold(x)
		s.set_seq1(x_)
		if s.real_quick_ratio() >= cutoff and \
						s.quick_ratio() >= cutoff and \
						s.ratio() >= cutoff:
			result.append((s.ratio(), x))

	# Move the best scorers to head of list
	result = _nlargest(n, result)
	# Strip scores for the best n matches
	ret = [x for score, x in result]
	if not excpt is None:
		did_you_mean = "'{}' not found, did you mean {}?".format(word, " or ".join("'{}'".format(s) for s in ret))
		raise excpt(did_you_mean)
	return ret


def max_len(arg, at_least=0):
	if len(arg) == 0:
		return at_least
	try:
		return max(at_least, *map(len, arg))
	except:
		try:
			return max(at_least, *map(len, *arg))
		except:
			print("ERR max_len")
			print("arg=", arg)
			print("type(arg)=", type(arg))
			raise


def grid_str(arg, lineprefix="", linesuffix=""):
	x = []
	try:
		n = 0
		while 1:
			x.append(max_len(tuple(i[n] for i in arg)))
			n += 1
	except IndexError:
		n -= 1
	lines = [lineprefix + " ".join(t[i].ljust(x[i]) for i in range(len(t))) + linesuffix for t in arg]
	return '\n'.join(lines)



def truncate_path_for_display(p, maxlen=30):
	"""Truncates a path if it is too long to display neatly.

	Do not use the trunacted path for actual file operations, it won't work.

	Parameters
	----------
	p : str or pathlib.Path
	maxlen : int
		Maximum number of character to return.  This is a suggestion, as
		the basename itself is never truncated.
	"""
	original = Path(p)
	if len(str(original)) <= maxlen:
		return str(original)
	original_parts_len = len(original.parts)
	trimmer = 1
	z = lambda x: str(Path('⋯', *original.parts[x:]))    # unicode equivalent '\u22EF'
	while len(z(trimmer)) > maxlen:
		trimmer += 1
		if trimmer >= original_parts_len:
			return z(original_parts_len-1)
	return z(trimmer)
