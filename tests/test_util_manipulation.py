import pandas
import os
import gzip
import pickle
import base64
import io
import numpy
from pytest import approx
from larch.util.data_manipulation import periodize

def test_periodize():
	h = pandas.Series(range(21))
	p = periodize(h, default='OP', AM=(6.5, 9), PM=(16, 19))
	assert p.dtype == 'category'
	assert p[0] == 'OP'
	assert p[1] == 'OP'
	assert p[2] == 'OP'
	assert p[3] == 'OP'
	assert p[4] == 'OP'
	assert p[5] == 'OP'
	assert p[6] == 'OP'
	assert p[7] == 'AM'
	assert p[8] == 'AM'
	assert p[9] == 'AM'
	assert p[10] == 'OP'
	assert p[11] == 'OP'
	assert p[12] == 'OP'
	assert p[13] == 'OP'
	assert p[14] == 'OP'
	assert p[15] == 'OP'
	assert p[16] == 'PM'
	assert p[17] == 'PM'
	assert p[18] == 'PM'
	assert p[19] == 'PM'
	assert p[20] == 'OP'
