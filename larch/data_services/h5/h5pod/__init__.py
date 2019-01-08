import os
from pathlib import Path
import tables as tb
import numpy
import pandas
import warnings
from ....util import Dict
from ....util.aster import asterize
from ....util.text_manip import truncate_path_for_display


from ... import _reserved_names_

from .generic import H5Pod, NoKnownShape, IncompatibleShape
from .idca import H5PodCA
from .idco import H5PodCO
from .idcs import H5PodCS
from .idce import H5PodCE
from .idga import H5PodGA
from .idrc import H5PodRC
from .id0a import H5Pod0A

def H5PodFactory(uri:str):
	from urllib.parse import urlparse, parse_qs
	p = urlparse(uri, scheme='file')
	q = parse_qs(p.query)
	if 'type' in q:
		cls = _pod_types[q['type'][0]]
	else:
		cls = H5Pod
	if 'mode' in q:
		mode = q['mode'][0]
	else:
		mode = 'r'
	return cls(filename=p.path, mode=mode, groupnode=p.fragment)


_pod_types = {
	'idco': H5PodCO,
	'idca': H5PodCA,
	'idce': H5PodCE,
	'idga': H5PodGA,
	'idcs': H5PodCS,
	'co': H5PodCO,
	'ca': H5PodCA,
	'ce': H5PodCE,
	'ga': H5PodGA,
	'cs': H5PodCS,
}
