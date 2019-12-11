
import os
from pathlib import Path
import tables as tb
import numpy
import pandas
import logging
from ....util import Dict
from ....util.aster import asterize
from ....util.text_manip import truncate_path_for_display
from ... import _reserved_names_
from ...pod import Pod
from ...general import _sqz_same, selector_len_for
from .... import warning

class IncompatibleShape(ValueError):
	pass

class NoKnownShape(ValueError):
	pass


class CArray(tb.CArray):

	@property
	def DICTIONARY(self):
		return Dict( self._v_attrs.DICTIONARY )

	@DICTIONARY.setter
	def DICTIONARY(self, x):
		self._v_attrs.DICTIONARY = dict(x)

	@property
	def DESCRIPTION(self):
		return self._v_attrs.TITLE

	@DESCRIPTION.setter
	def DESCRIPTION(self, x):
		self._v_attrs.TITLE = str(x)

	@property
	def TITLE(self):
		return self._v_attrs.TITLE

	@TITLE.setter
	def TITLE(self, x):
		self._v_attrs.TITLE = str(x)


	def __repr__(self):
		r = super().__repr__()
		try:
			d = self.DICTIONARY
		except:
			return r
		r += "\n  dictionary := {\n      "
		r += repr(d).replace("\n","\n      ")
		r += "\n  }"
		return r

	def uniques(self, slicer=None, counts=False):
		if isinstance(slicer, (bool, int)) and counts is False:
			counts = bool(slicer)
			slicer = None
		if slicer is None:
			slicer = slice(None)
		action = self[slicer]
		len_action = len(action)
		try:
			action = action[~numpy.isnan(action)]
		except TypeError:
			num_nan = 0
		else:
			num_nan = len_action - len(action)
		if counts:
			x = numpy.unique(action, return_counts=counts)
			try:
				d = self.DICTIONARY
			except AttributeError:
				y = pandas.Series(x[1], x[0])
			else:
				y = pandas.Series(x[1], [(d[j] if j in d else j) for j in x[0]])
			if num_nan:
				y[numpy.nan] = num_nan
			return y
		if num_nan:
			numpy.append(action, numpy.nan)
		return numpy.unique(action)


class H5Pod(Pod):
	def __init__(self, filename=None, mode='a', groupnode=None, *,
	             h5f=None, inmemory=False, temp=False,
	             complevel=1, complib='zlib',
				 do_nothing=False,
				 ident=None,
				 shape=None,
	             ):

		super().__init__(ident=ident)

		if do_nothing:
			return

		if isinstance(filename, H5Pod):
			# Copy / Re-Class contructor
			x = filename
			self._groupnode = x._groupnode
			self._h5f_own = False
			return

		if isinstance(filename, tb.group.Group) and groupnode is None:
			# Called with just a group node, use it
			groupnode = filename
			filename = None

		if isinstance(filename, (str,Path)):
			filename = os.fspath(filename)
			if groupnode is None:
				groupnode = "/"

		if filename is None and mode=='a' and groupnode is None:
			groupnode = '/' # default constructor for temp obj

		if isinstance(groupnode, tb.group.Group):
			# Use the existing info from this group node, ignore all other inputs
			self._groupnode = groupnode
			filename = groupnode._v_file.filename
			mode = groupnode._v_file.mode
			self._h5f_own = False

		elif isinstance(groupnode, str):

			# apply expanduser to filename to allow for home-folder based filenames
			if isinstance(filename,str):
				filename = os.path.expanduser(filename)

			if filename is None:
				temp = True
				from ....util.temporaryfile import TemporaryFile
				self._TemporaryFile = TemporaryFile(suffix='.h5d')
				filename = self._TemporaryFile.name

			if h5f is not None:
				self._h5f_own = False
				self._groupnode = self._h5f.get_node(groupnode)
			else:
				kwd = {}
				if inmemory or temp:
					kwd['driver']="H5FD_CORE"
				if temp:
					kwd['driver_core_backing_store']=0
				if complevel is not None:
					kwd['filters']=tb.Filters(complib=complib, complevel=complevel)
				self._h5f_obj = tb.open_file(filename, mode, **kwd)
				self._h5f_own = True
				try:
					self._groupnode = self._h5f_obj.get_node(groupnode)
				except tb.NoSuchNodeError:
					if isinstance(groupnode, str):
						self._groupnode = self._h5f_obj._get_or_create_path(groupnode, True)
		else:
			raise ValueError('must give groupnode as `str` or `tables.group.Group`')

		self._recoverable = (filename, self._groupnode._v_pathname)

		if shape is not None:
			self.shape = shape

	@property
	def _h5f(self):
		return self._groupnode._v_file

	def __repr__(self):
		from ....util.text_manip import max_len
		s = f"<larch.{self.__class__.__name__}>"
		try:
			s += f"\n |  file: {truncate_path_for_display(self.filename)}"
			if self._groupnode._v_pathname != "/":
				s += f"\n |  node: {self._groupnode._v_pathname}"
			try:
				shape = self.shape
			except NoKnownShape:
				shape = None
			else:
				s += f"\n |  shape: {shape}"
			try:
				metashape = self.metashape
			except (NoKnownShape, AttributeError):
				pass
			else:
				if metashape != shape:
					s += f"\n |  metashape: {metashape}"
			if len(self._groupnode._v_children):
				s += "\n |  data:"
				just = max_len(self._groupnode._v_children.keys())
				for i in sorted(self._groupnode._v_children.keys()):
					try:
						node_dtype = self._groupnode._v_children[i].dtype
					except tb.NoSuchNodeError:
						node_dtype = "<no dtype>"
					s += "\n |    {0:{2}s} ({1})".format(i, node_dtype, just)
			else:
				s += "\n |  data: <empty>"
		except (tb.ClosedNodeError, tb.ClosedFileError):
			s += f"\n |  <file is closed>"
			s += f"\n |  file: {truncate_path_for_display(self._recoverable[0])}"
			s += f"\n |  node: {self._recoverable[1]}"
		return s

	def __xml__(self, no_data=False, descriptions=True, dictionaries=False):
		from xmle import Elem
		x = Elem('div')
		t = x.elem('table', style="margin-top:1px;")
		t.elem('caption', text=f"<larch.{self.__class__.__name__}>", style="caption-side:top;text-align:left;font-family:Roboto;font-weight:700;font-style:normal;font-size:100%;padding:0px;")
		#
		try:
			ident = self.ident
		except AttributeError:
			pass
		else:
			tr = t.elem('tr')
			tr.elem('th', text='ident')
			tr.elem('td', text=ident)
		#
		try:
			filename = self.filename
		except AttributeError:
			pass
		else:
			tr = t.elem('tr')
			tr.elem('th', text='file')
			tr.elem('td', text=truncate_path_for_display(filename))
		#
		try:
			filemode = self.filemode
		except AttributeError:
			pass
		else:
			tr = t.elem('tr')
			tr.elem('th', text='mode')
			tr.elem('td', text=truncate_path_for_display(self.filemode))
		#
		try:
			v_pathname = self._groupnode._v_pathname
		except AttributeError:
			pass
		else:
			if self._groupnode._v_pathname != "/":
				tr = t.elem('tr')
				tr.elem('th', text='node')
				tr.elem('td', text=self._groupnode._v_pathname)
		#
		try:
			str_shape = str(self.shape)
		except NoKnownShape:
			pass
		else:
			tr = t.elem('tr')
			tr.elem('th', text='shape')
			tr.elem('td', text=str_shape)
		#
		try:
			str_shape = str(self.metashape)
		except (NoKnownShape, AttributeError):
			pass
		else:
			tr = t.elem('tr')
			tr.elem('th', text='metashape')
			tr.elem('td', text=str_shape)
		#
		try:
			str_durable_mask = f"0x{self.durable_mask:X}"
		except (AttributeError):
			pass
		else:
			if str_durable_mask!='0x0':
				tr = t.elem('tr')
				tr.elem('th', text='durable_mask')
				tr.elem('td', text=str_durable_mask)
		#
		if not no_data:
			if len(self._groupnode._v_children):
				tr = t.elem('tr')
				tr.elem('th', text='data', style='vertical-align:top;')
				td = tr.elem('td')
				t1 = td.elem('table', cls='dictionary')
				t1head = t1.elem('thead')
				t1headr = t1head.elem('tr')
				t1headr.elem('th', text='name')
				t1headr.elem('th', text='dtype')
				if descriptions:
					t1headr.elem('th', text='description')
				any_sources = 0
				for i in sorted(self._groupnode._v_children.keys()):
					try:
						node_dtype = self._groupnode._v_children[i].dtype
					except (tb.NoSuchNodeError, AttributeError):
						node_dtype = "<no dtype>"
					if i not in _reserved_names_:
						tr1 = t1.elem('tr')
						tr1.elem('td', text=i)
						tr1.elem('td', text=node_dtype)
						if descriptions:
							try:
								title = self._groupnode._v_children[i]._v_attrs['TITLE']
							except:
								title = ""
							else:
								tr1.elem('td', text=title)
						try:
							orig_source = self._groupnode._v_children[i]._v_attrs['ORIGINAL_SOURCE']
						except:
							pass
						else:
							tr1.elem('td', text=orig_source)
							any_sources += 1
				if any_sources:
					t1headr.elem('th', text='source')
			else:
				tr = t.elem('tr')
				tr.elem('th', text='data', style='vertical-align:top;')
				tr.elem('td', text='<empty>')
		return x

	def _repr_html_(self):
		return self.__xml__().tostring()

	def change_mode(self, mode, **kwarg):
		"""Change the file mode of the underlying HDF5 file.

		Can be used to change from read-only to read-write.
		"""
		if mode == self.filemode:
			return
		if mode == 'w':
			raise TypeError("cannot change_mode to w, close the file and delete it")
		filename = self.filename
		groupnode_path = self._groupnode._v_pathname
		self.close()
		self.__init__(filename, mode, groupnode=groupnode_path, **kwarg)
		return self

	def reopen(self, mode='r', **kwarg):
		"""Reopen the underlying HDF5 file.

		Can be used to change from read-only to read-write or to reopen a file that was closed.
		"""
		if mode == self.filemode:
			return
		if mode == 'w':
			raise TypeError("cannot change_mode to w, close the file and delete it")
		filename = self.filename
		groupnode_path = self.groupnode_path
		try:
			self.close()
		except tb.ClosedNodeError:
			pass
		self.__init__(filename, mode, groupnode=groupnode_path, **kwarg)
		return self

	def names(self):
		return [i for i in self._groupnode._v_children.keys() if i not in _reserved_names_]

	def rename_vars(self, *rename_vars):
		"""
		Rename variables according to patterns.

		Parameters
		----------
		rename_vars : 2-tuples
			A sequence of 2-tuples, giving (pattern, replacement) that will be fed to re.sub.
			For example, give ('^','prefix_') to add prefix to all variable names, or
			('^from_this_name$','to_this_name') to change an exact name from one thing to another.

		"""
		import re
		for pattern, replacement in rename_vars:
			q = [(re.sub(pattern, replacement, k),k) for k in self.names()]
			for _to,_from in q:
				if _to != _from:
					self._groupnode._v_children[_from]._f_rename(_to)

	def reshape(self, *shape):
		if len(shape)==0:
			raise ValueError('no shape given')
		if len(shape)==1 and isinstance(shape[0], tuple):
			shape = shape[0]
		if isinstance(shape, int):
			shape = (shape,)
		if not isinstance(shape, tuple):
			raise TypeError('reshape must be int or tuple')
		if len(shape)==1 and shape[0] == -1:
			shape = (int(numpy.product(self.shape)), )
		elif len(shape)==2:
			if shape[0] == -1 and shape[1] > 0:
				shape = (int(numpy.product(self.shape) / shape[1]), shape[1])
			if shape[1] == -1 and shape[0] > 0:
				shape = (shape[0], int(numpy.product(self.shape) / shape[0]))
		if numpy.product(shape) != numpy.product(self.shape):
			raise ValueError(f'incompatible reshape {shape} for current shape {self.shape}')
		#print("reshape to", shape)
		for k in self._groupnode._v_children.keys():
			#print("reshape",k,shape)
			self._groupnode._v_children[k].shape = shape
		self.shape = shape

	def __dir__(self):
		x = super().__dir__()
		x.extend(self.names())
		return x

	@property
	def filename(self):
		try:
			return self._groupnode._v_file.filename
		except (tb.ClosedNodeError, tb.ClosedFileError) as err:
			try:
				return self._last_closed_filename
			except AttributeError:
				raise err

	@property
	def filemode(self):
		try:
			return self._groupnode._v_file.mode
		except (tb.ClosedNodeError, tb.ClosedFileError) as err:
			return None

	@property
	def groupnode_path(self):
		try:
			return self._groupnode._v_pathname
		except (tb.ClosedNodeError, tb.ClosedFileError) as err:
			try:
				return self._last_closed_groupnode_path
			except AttributeError:
				raise err

	@property
	def n_cases(self):
		return self.shape[0]

	@property
	def shape(self):
		"""The shape of the pod.

		"""
		if 'SHAPE' in self._groupnode._v_attrs:
			return tuple(self._groupnode._v_attrs['SHAPE'][:])
		if len(self.names()):
			for v in self._groupnode._v_children.values():
				try:
					found_shape = v.shape
				except:
					pass
				else:
					try:
						self.shape = found_shape
					except:
						pass
					return tuple(found_shape)
		raise NoKnownShape()

	@shape.setter
	def shape(self, x):
		# if self._groupnode._v_nchildren > 0:
		# 	raise ValueError('this pod has shape {!s} but you want to set {!s}'.format(self.shape, x))
		# if self._groupnode._v_nchildren == 0:
		self._groupnode._v_attrs.SHAPE = numpy.asarray(x, dtype=int)

	@property
	def metashape(self):
		"""The actual shape of the data underlying the pod, often same as shape."""
		return self.shape

	def add_expression(self, name, expression, *, overwrite=False, title=None, dictionary=None, dtype=None):
		arr = self[expression]
		if dtype is not None:
			arr = arr.astype(dtype)
			try:
				dtype_str = "("+dtype.__name__+")"
			except:
				dtype_str = ""
			original_source = f'={dtype_str} {expression}'
		else:
			original_source = f'= {expression}'
		if overwrite=='ignore':
			overwrite = False
			if_exists = 'ignore'
		else:
			if_exists = 'raise'
		try:
			self.add_array(name, arr, overwrite=overwrite, title=title, dictionary=dictionary,
						   original_source=original_source, rel_original_source=False)
		except tb.exceptions.NodeError:
			if if_exists=='ignore':
				pass
			else:
				raise

	def add_array(self, name, arr, *, overwrite=False, original_source=None, rel_original_source=True,
	              title=None, dictionary=None, fix_name_problems=True):
		"""Create a new variable in the H5Pod.

		Creating a new variable in the data might be convenient in some instances.
		If you create an array externally, you can add it to the file easily with
		this command.

		Parameters
		----------
		name : str
			The name of the new variable.
		arr : ndarray
			An array to add as the new variable.  Must have the correct shape.
		overwrite : bool
			Should the variable be overwritten if it already exists, default to False.
		original_source : str
			Optionally, give the file name or other description of the source of the data in this array.
		rel_original_source : bool
			If true, change the absolute path of the original_source to a relative path viz this file.
		title : str, optional
			A descriptive title for the variable, typically a short phrase but an
			arbitrary length description is allowed.
		dictionary : dict, optional
			A data dictionary explaining some or all of the values in this field.
			Even for otherwise self-explanatory numerical values, the dictionary
			may give useful information about particular out of range values.

		Raises
		------
		tables.exceptions.NodeError
			If a variable of the same name already exists.
		"""
		if name in _reserved_names_:
			raise ValueError(f'{name} is a reserved name')
		if '/' in name and fix_name_problems:
			import warnings
			warnings.warn(f'the ``/`` character is not allowed in variable names ({name})\n'
						  f'changing it to ``|``')
			name = name.replace('/','|')
		try:
			existing_shape = tuple(self.metashape)
		except NoKnownShape:
			pass
		else:
			if existing_shape != arr.shape:
				# maybe just has extra size-1 dims, check for that...
				arr = arr.squeeze()
				if self.podtype == 'idcs':
					if existing_shape[:-1] != arr.shape:
						raise IncompatibleShape(
							"new array must have shape {!s} but the array given has shape {!s}".format(self.shape, arr.shape))
				else:
					if existing_shape != arr.shape:
						raise IncompatibleShape(
							"new array must have shape {!s} but the array given has shape {!s}".format(self.shape, arr.shape))
		if overwrite:
			self.delete_array(name)
		try:
			h5var = self._h5f.create_carray(self._groupnode, name, obj=arr)
		except ValueError as valerr:
			if "unknown type" in str(valerr) or "unknown kind" in str(valerr):  # changed for pytables 3.3
				try:
					tb_atom = tb.Atom.from_dtype(arr.dtype)
				except ValueError:
					log = logging.getLogger('H5')
					try:
						maxlen = int(len(max(arr.astype(str), key=len)))
					except ValueError:
						import datetime
						if 0:  # isinstance(arr[0], datetime.time):
							log.warning(f"  column {name} is datetime.time, converting to Time32")
							tb_atom = tb.atom.Time32Atom()

							# convert_datetime_time_to_epoch_seconds = lambda tm: tm.hour*3600+ tm.minute*60 + tm.second
							def convert_datetime_time_to_epoch_seconds(tm):
								try:
									return tm.hour * 3600 + tm.minute * 60 + tm.second
								except:
									if numpy.isnan(tm):
										return 0
									else:
										raise

							arr = arr.apply(convert_datetime_time_to_epoch_seconds).astype(numpy.int32).values
						else:
							# import __main__
							# __main__.err_df = df
							raise
					else:
						maxlen = max(maxlen, 8)
						if arr.dtype != object:
							log.warning(f"cannot create column {name} as dtype {arr.dtype}, converting to S{maxlen:d}")
						arr = arr.astype('S{}'.format(maxlen))
						tb_atom = tb.Atom.from_dtype(arr.dtype)
				h5var = self._h5f.create_carray(self._groupnode, name, tb_atom, shape=arr.shape)
				h5var[:] = arr
			else:
				raise
		if rel_original_source and original_source and original_source[0] != '=':
			basedir = os.path.dirname(self.filename)
			original_source = os.path.relpath(original_source, start=basedir)
		if original_source is not None:
			h5var._v_attrs.ORIGINAL_SOURCE = original_source
		if title is not None:
			h5var._v_attrs.TITLE = title
		if dictionary is not None:
			h5var._v_attrs.DICTIONARY = dictionary

	def add_blank(self, name, shape=None, dtype=numpy.float64, **kwargs):
		"""Create a new variable in the H5Pod.

		Creating a new variable in the data might be convenient in some instances.
		If you create an array externally, you can add it to the file easily with
		this command.

		Parameters
		----------
		name : str
			The name of the new variable.
		dtype : dtype
			The dtype of the empty array to add as the new variable.
		shape : tuple
			The shape of the empty array to add. Must be compatible with existing
			shape, if any.

		Other keyword parameters are passed through to `add_array`.

		Raises
		------
		tables.exceptions.NodeError
			If a variable of the same name already exists.
		NoKnownShape
			If shape is not given and not already known from the file.
		"""
		if name in _reserved_names_:
			raise ValueError(f'{name} is a reserved name')
		try:
			existing_shape = tuple(self.metashape)
		except NoKnownShape:
			if shape is None:
				raise
		else:
			if shape is None:
				shape = existing_shape
			if existing_shape != tuple(shape):
				raise IncompatibleShape(
					"new array must have shape {!s} but the array given has shape {!s}".format(self.shape, shape))
		return self.add_array(name, numpy.zeros(shape, dtype=dtype), **kwargs)


	def delete_array(self, name, recursive=True):
		"""Delete an existing variable.

		Parameters
		----------
		name : str
			The name of the data node to remove.
		recursive : bool
			If the data node is a group, recursively remove all sub-nodes.

		"""
		if name in _reserved_names_:
			raise ValueError(f'{name} is a reserved name')
		try:
			self._h5f.remove_node(self._groupnode, name, recursive)
		except tb.exceptions.NoSuchNodeError:
			pass


	def flush(self, *arg, **kwargs):
		return self._h5f.flush(*arg, **kwargs)

	def close(self, *arg, **kwargs):
		try:
			self._last_closed_filename = self.filename
			self._last_closed_groupnode_path = self.groupnode_path
		except:
			pass
		return self._h5f.close(*arg, **kwargs)

	@property
	def podtype(self):
		return ''

	def uri(self, mode=None):
		from urllib.parse import urlunparse
		q_dict = {}
		if self.podtype:
			q_dict['type'] = self.podtype
		if mode:
			q_dict['mode'] = mode
		q = "&".join(f'{k}={v}' for k,v in q_dict.items())
		return urlunparse(['file', '', self.filename, '', q, self._groupnode._v_pathname])

	def _remake_command(self, cmd, selector=None, receiver=None):
		from tokenize import tokenize, untokenize, NAME, OP, STRING
		DOT = (OP, '.')
		COLON = (OP, ':')
		COMMA = (OP, ',')
		OBRAC = (OP, '[')
		CBRAC = (OP, ']')
		OPAR = (OP, '(')
		CPAR = (OP, ')')
		from io import BytesIO
		recommand = []

		if receiver:
			recommand += [(NAME, receiver), OBRAC, COLON, CBRAC, (OP, '='), ]

		try:
			cmd_encode = cmd.encode('utf-8')
		except AttributeError:
			cmd_encode = str(cmd).encode('utf-8')
		dims = len(self.shape)
		g = tokenize(BytesIO(cmd_encode).readline)
		if selector is None:
			screen_tokens = [COLON,]
		else:
			# try:
			# 	slicer_encode = selector.encode('utf-8')
			# except AttributeError:
			# 	slicer_encode = str(selector).encode('utf-8')
			# screen_tokens = [(toknum, tokval) for toknum, tokval, _, _, _ in tokenize(BytesIO(slicer_encode).readline)]
			screen_tokens = [(NAME, 'selector'), ]
		for toknum, tokval, _, _, _ in g:
			if toknum == NAME and tokval in self._groupnode:
				# replace NAME tokens
				partial = [(NAME, 'self'), DOT, (NAME, '_groupnode'), DOT, (NAME, tokval), OBRAC, ]
				partial += screen_tokens
				if len(self._groupnode._v_children[tokval].shape)>1:
					partial += [COMMA, COLON, ]
				if len(self._groupnode._v_children[tokval].shape)>2:
					partial += [COMMA, COLON, ]
				if len(self._groupnode._v_children[tokval].shape)>3:
					partial += [COMMA, COLON, ]
				partial += [CBRAC,]


				recommand.extend(partial)
			else:
				recommand.append((toknum, tokval))
		# print("<recommand>")
		# print(recommand)
		# print("</recommand>")
		ret = untokenize(recommand).decode('utf-8')
		return asterize(ret, mode="exec" if receiver is not None else "eval"), ret


	def _evaluate_single_item(self, cmd, selector=None, receiver=None):
		j, j_plain = self._remake_command(cmd, selector=selector, receiver='receiver' if receiver is not None else None)
		# important globals
		from ....util.aster import inXd
		from numpy import log, exp, log1p, absolute, fabs, sqrt, isnan, isfinite, logaddexp, fmin, fmax, nan_to_num, sin, cos, pi
		from ....util.common_functions import piece, normalize
		try:
			if receiver is not None:
				exec(j)
			else:
				return eval(j)
		except Exception as exc:
			args = exc.args
			if not args:
				arg0 = ''
			else:
				arg0 = args[0]
			arg0 = arg0 + '\nwithin parsed command: "{!s}"'.format(cmd)
			arg0 = arg0 + '\nwithin re-parsed command: "{!s}"'.format(j_plain)
			if selector is not None:
				arg0 = arg0 + '\nwith selector: "{!s}"'.format(selector)
			if "max" in cmd:
				arg0 = arg0 + '\n(note to get the maximum of arrays use "fmax" not "max")'.format(cmd)
			if "min" in cmd:
				arg0 = arg0 + '\n(note to get the minimum of arrays use "fmin" not "min")'.format(cmd)
			if isinstance(exc, NameError):
				badname = str(exc).split("'")[1]
				goodnames = dir()
				from ....util.text_manip import case_insensitive_close_matches
				did_you_mean_list = case_insensitive_close_matches(badname, goodnames, n=3, cutoff=0.1, excpt=None)
				if len(did_you_mean_list) > 0:
					arg0 = arg0 + '\n' + "did you mean {}?".format(
						" or ".join("'{}'".format(s) for s in did_you_mean_list))
			exc.args = (arg0,) + args[1:]
			raise

	def __contains__(self, item):
		if item in self._groupnode:
			return True
		return False

	def dtype_of(self, name):
		"""dtype of raw data for a particular named data item."""
		if name in self._groupnode._v_children:
			return self._groupnode._v_children[name].dtype
		raise KeyError(f"{name} not found")

	def load_data_item(self, name, result, selector=None):
		"""Load a slice of the pod arrays into an array in memory"""
		# convert a single name string to a one item list
		_sqz_same(result.shape, [selector_len_for(selector, self.shape[0]), *self.shape[1:]])
		try:
			result[:] = self._evaluate_single_item(name, selector)
		except IndexError:
			# https://github.com/PyTables/PyTables/issues/310
			_temp = self._evaluate_single_item(name, None)
			try:
				result[:] = _temp[selector]
			except Exception as err:
				raise ValueError(f'_temp.shape={_temp.shape}  selector.shape={selector.shape}') from err
		return result

	def load_meta_data_item(self, name, result, selector=None):
		"""Load a slice of the pod arrays into an array in memory"""

		if selector is not None:
			import warnings
			warnings.warn('selector not compatible for load_meta_data_item')

		# convert a single name string to a one item list
		_sqz_same(result.shape, self.metashape)
		try:
			result[:] = self._evaluate_single_item(name, selector)
		except IndexError:
			# https://github.com/PyTables/PyTables/issues/310
			result[:] = self._evaluate_single_item(name, None)[selector]
		return result

	def get_data_dictionary(self, name):
		"""dictionary of raw data for a particular named data item."""
		if name in self._groupnode._v_children:
			return self._groupnode._v_children[name].DICTIONARY
		raise KeyError(f"{name} not found")

	def __getitem__(self, item):
		if isinstance(item, tuple) and len(item)>=2 and isinstance(item[-1], slice):
			names, slice_ = item[:-1], item[-1]
		else:
			names = item
			slice_ = None

		# convert a single name string to a one item list
		if isinstance(names, str):
			names = [names,]

		dtype = numpy.float64

		result = numpy.zeros( [selector_len_for(slice_, self.shape[0]), *self.shape[1:], len(names)], dtype=dtype)

		for i, cmd in enumerate(names):
			result[...,i] = self._evaluate_single_item(cmd, slice_)

		return result

	def _load_into(self, names, slc, result):
		"""Load a slice of the pod arrays into an array in memory"""
		# convert a single name string to a one item list
		if isinstance(names, str):
			names = [names, ]
		_sqz_same(result.shape,[selector_len_for(slc, self.shape[0]), *self.shape[1:], len(names)])
		for i, cmd in enumerate(names):
			result[..., i] = self._evaluate_single_item(cmd, slc)
		return result

	def load_into(self, names, selector, result):
		"""Load a slice of the pod arrays into an array in memory"""
		# convert a single name string to a one item list
		if isinstance(names, str):
			names = [names, ]
		_sqz_same(result.shape, [selector_len_for(selector, self.shape[0]), *self.shape[1:], len(names)])
		for i, cmd in enumerate(names):
			result[..., i] = self._evaluate_single_item(cmd, selector)
		return result

	def __getattr__(self, item):
		if item in self._groupnode._v_children:
			ret = self._groupnode._v_children[item]
			if isinstance(ret, tb.CArray):
				ret.__class__ = CArray
			return ret
		raise AttributeError(item)

	class _dataframe_factory():
		def __init__(self, obj):
			self.obj = obj
		def __getattr__(self, item):
			return getattr(self.obj,item)
		def __getitem__(self, item):
			if len(self.obj.shape) > 1:
				try:
					metashape = self.obj.metashape
				except AttributeError:
					raise TypeError('dataframe access currently only compatible with 1d, use regular arrays for higher dimensions')
				else:
					if len(metashape) > 1:
						raise TypeError('dataframe access currently only compatible with 1d, use regular arrays for higher dimensions')

			if isinstance(item, tuple) and len(item)>=2 and isinstance(item[-1], slice):
				names, slice_ = item[:-1], item[-1]
			else:
				names = item
				slice_ = None

			# convert a single name string to a one item list
			if isinstance(names, str):
				names = [names,]

			result = pandas.DataFrame()

			for i, cmd in enumerate(names):
				j = self.obj._evaluate_single_item(cmd, selector=slice_)
				try:
					#result.loc[:,cmd] = j
					result = result.assign(**{cmd:j})
				except:
					print()
					print(f"An error in tacking {cmd} to result")
					print(f"j.dtype is {j.dtype}")
					print(f"j.shape is {j.shape}")
					print(f"result.shape is {result.shape}")
					print()
					raise

			return result

	@property
	def dataframe(self):
		return self._dataframe_factory(self)

	def astype(self, t:str):
		from . import _pod_types
		cls = _pod_types[t.lower()]
		return cls(self)

	def statistics_for(self, var, histogram=True, selector=None, **kwargs):
		a = self.get_data_item(var)
		if isinstance(selector, str):
			selector = self.get_data_item(selector, None, dtype=bool)
		if selector is not None:
			a = a[selector]
		from ....util.statistics import statistics_for_array
		try:
			dictionary = self._groupnode._v_children[var]._v_attrs.DICTIONARY
		except:
			dictionary = None
		try:
			descrip = self._groupnode._v_children[var]._v_attrs.TITLE
		except:
			descrip = None
		result = statistics_for_array(a, histogram=histogram, varname=var, dictionary=dictionary, **kwargs)
		if descrip is not None and descrip!="":
			result.description = descrip
		if dictionary is not None:
			result.dictionary = Dict(dictionary)
		return result

	def statistics(self, vars=None, histogram=False, selector=None):
		if vars is None:
			vars = self.names()
		from ....util import Dict
		from ....util.arraytools import scalarize
		import numpy.ma as ma
		stats = pandas.DataFrame(
			columns=[
				'n',
				'minimum',
				'maximum',
				'median',
				'mean',
				'stdev',
				'nonzero_minimum',
				'nonzero_maximum',
				'nonzero_mean',
				'nonzero_stdev',
				'zeros',
				'positives',
				'negatives',
			] + (['histogram'] if histogram else []),
			index = vars,
		)
		for var in vars:
			if selector is not None:
				if isinstance(selector, slice):
					a = self[var, selector]
				else:
					a = self[var][selector]
			else:
				a = self[var]
			stats.loc[var,'n'] = scalarize(a.shape[0])
			stats.loc[var,'minimum'] = scalarize(numpy.nanmin(a, axis=0))
			stats.loc[var,'maximum'] = scalarize(numpy.nanmax(a, axis=0))
			stats.loc[var,'median'] = scalarize(numpy.nanmedian(a, axis=0))
			if histogram:
				from ....util.histograms import sizable_histogram_figure, seems_like_discrete_data
				try:
					dictionary = self.get_data_dictionary(var)
				except:
					dictionary = None
				stats.loc[var,'histogram'] = sizable_histogram_figure(
					a,
					title=None, xlabel=var, ylabel='Frequency',
					discrete=seems_like_discrete_data(a, dictionary)
					)
			ax = ma.masked_array(a, mask=~numpy.isfinite(a))
			stats.loc[var,'mean'] = scalarize(numpy.mean(ax, axis=0))
			stats.loc[var,'stdev'] = scalarize(numpy.std(ax, axis=0))
			stats.loc[var, 'zeros'] = scalarize(numpy.sum(numpy.logical_not(ax), axis=0))
			stats.loc[var, 'positives'] = scalarize(numpy.sum(ax>0, axis=0))
			stats.loc[var, 'negatives'] = scalarize(numpy.sum(ax<0, axis=0))
			ax.mask |= (ax==0)
			stats.loc[var,'nonzero_minimum'] = scalarize(numpy.min(ax, axis=0))
			stats.loc[var,'nonzero_maximum'] = scalarize(numpy.max(ax, axis=0))
			stats.loc[var,'nonzero_mean'] = scalarize(numpy.mean(ax, axis=0))
			stats.loc[var,'nonzero_stdev'] = scalarize(numpy.std(ax, axis=0))
		if histogram:
			from ....util.dataframe import DataFrameViewer
			return DataFrameViewer(stats)
		return stats

	def get_row(self, rownum, lookup=True):
		result = Dict()
		for i in self.names():
			result[i] = self._groupnode._v_children[i][rownum]
			if lookup:
				try:
					d = self._groupnode._v_children[i]._v_attrs.DICTIONARY
				except (KeyError, AttributeError):
					pass
				else:
					if result[i] in d:
						result[i] = f"{result[i]} ({d[result[i]]})"
		return result

	@property
	def vault(self):
		try:
			return self.__vault
		except:
			from ..h5util import get_or_create_group
			from ..h5vault import H5Vault
			v = get_or_create_group( self._h5f, self._groupnode, name='_VAULT_', title='', filters=None, createparents=False, skip_on_readonly=False )
			self.__vault = H5Vault(v)
			return self.__vault

