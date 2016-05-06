
try:
	import docx
	from docx.enum.style import WD_STYLE_TYPE
except ImportError:

	class DocxModelReporter():
		pass

else:
	from ..utilities import category, pmath, rename
	from ..core import LarchError, ParameterAlias


	def _append_to_document(self, other_doc):
		while not isinstance(other_doc, docx.document.Document) and hasattr(other_doc, '_parent'):
			other_doc = other_doc._parent
		if not isinstance(other_doc, docx.document.Document):
			raise larch.LarchError('other_doc is not a docx.Document or a part thereof')
		for element in other_doc._body._element:
			self._body._element.append(element)
		return self

	docx.document.Document.append = _append_to_document



	def document_larchstyle():
		document = docx.Document()

		monospaced_small = document.styles.add_style('Monospaced Small',WD_STYLE_TYPE.TABLE)
		monospaced_small.base_style = document.styles['Normal']
		monospaced_small.font.name = 'Courier New'
		monospaced_small.font.size = docx.shared.Pt(9)
		monospaced_small.paragraph_format.space_before = docx.shared.Pt(0)
		monospaced_small.paragraph_format.space_after  = docx.shared.Pt(0)
		monospaced_small.paragraph_format.line_spacing = 1.0

		return document


	def docx_table(*arg, header_text=None, header_level=None,  **kwarg):
		doc = document_larchstyle()
		if header_text is not None:
			if header_level is None:
				header_level = 1
			doc.add_heading(header_text, level=header_level)
		tbl = doc.add_table(*arg, **kwarg)
		return tbl



	class DocxModelReporter():

		def docx_params(self, groups=None, display_inital=False, **format):

			# keys fix
			existing_format_keys = list(format.keys())
			for key in existing_format_keys:
				if key.upper()!=key: format[key.upper()] = format[key]
			if 'PARAM' not in format: format['PARAM'] = '< 12.4g'
			if 'TSTAT' not in format: format['TSTAT'] = ' 0.2f'

			number_of_columns = 5
			if display_inital:
				number_of_columns += 1


			if groups is None and hasattr(self, 'parameter_groups'):
				groups = self.parameter_groups

			table = docx_table(rows=1, cols=number_of_columns, style='Monospaced Small',
							   header_text="Model Parameter Estimates", header_level=2)

			def append_simple_row(name, initial_value, value, std_err, tstat, nullvalue, holdfast):
				row_cells = table.add_row().cells
				i = 0
				row_cells[i].text = name
				i += 1
				if display_inital:
					row_cells[i].text = "{:{PARAM}}".format(initial_value, **format     )
					i += 1
				row_cells[i].text = "{:{PARAM}}".format(value , **format)
				i += 1
				if holdfast:
					row_cells[i].text = "fixed value"
					row_cells[i].merge(row_cells[i+1])
					i += 2
				else:
					row_cells[i].text = "{:.3g}".format(std_err   , **format)
					i += 1
					row_cells[i].text = "{:{TSTAT}}".format(tstat , **format  )
					i += 1
				row_cells[i].text = "{:.1f}".format(nullvalue , **format)

			def append_derivative_row(name, initial_value, value, refers_to, multiplier):
				row_cells = table.add_row().cells
				i = 0
				row_cells[i].text = name
				i += 1
				if display_inital:
					row_cells[i].text = "{:{PARAM}}".format(initial_value, **format     )
					i += 1
				row_cells[i].text = "{:{PARAM}}".format(value , **format)
				i += 1
				row_cells[i].text = "= {} * {}".format(refers_to,multiplier)
				row_cells[i].merge(row_cells[i+2])
				i += 3

			hdr_cells = table.rows[0].cells
			i = 0
			hdr_cells[i].text = 'Parameter'
			i += 1
			if display_inital:
				hdr_cells[i].text = 'Initial Value'
				i += 1
			hdr_cells[i].text = 'Estimated Value'
			i += 1
			hdr_cells[i].text = 'Std Error'
			i += 1
			hdr_cells[i].text = 't-Stat'
			i += 1
			hdr_cells[i].text = 'Null Value'
			i += 1
			for cell in hdr_cells:
				cell.paragraphs[0].runs[0].bold = True


			if groups is None:
				for par in self.parameter_names():
					append_simple_row(
						par,
						self.parameter(par).initial_value,
						self.parameter(par).value,
						self.parameter(par).std_err,
						self.parameter(par).t_stat,
						self.parameter(par).null_value,
						self.parameter(par).holdfast
					)

			else:
				
				## USING GROUPS
				listed_parameters = set([p for p in groups if not isinstance(p,category)])
				for p in groups:
					if isinstance(p,category):
						listed_parameters.update( p.complete_members() )
				unlisted_parameters = (set(self.parameter_names()) | set(self.alias_names())) - listed_parameters


				def write_param_row(p, *, force=False):
					if p is None: return
					if force or (p in self) or (p in self.alias_names()):
						if isinstance(p,category):
							row_cells = table.add_row().cells
							row_cells[0].merge(row_cells[-1])
							row_cells[0].text = p.name
							#row_cells[0].style = "parameter_category"
							for subp in p.members:
								write_param_row(subp)
						else:
							if isinstance(p,rename):
								append_simple_row(par,
									self[p].initial_value,
									self[p].value,
									self[p].std_err,
									self[p].t_stat,
									self[p].null_value,
									self[p].holdfast
								)
							else:
								pwide = self.parameter_wide(p)
								if isinstance(pwide,ParameterAlias):
									append_derivative_row(pwide.name,
										self.metaparameter(pwide.name).initial_value,
										self.metaparameter(pwide.name).value,
										pwide.refers_to,
										pwide.multiplier
									)
								else:
									append_simple_row(pwide.name,
										pwide.initial_value,
										pwide.value,
										pwide.std_err,
										pwide.t_stat,
										pwide.null_value,
										pwide.holdfast
									)


				# end def
				for p in groups:
					write_param_row(p)
				if len(groups)>0 and len(unlisted_parameters)>0:
					write_param_row(category("Other Parameters"),force=True)
				if len(unlisted_parameters)>0:
					for p in unlisted_parameters:
						write_param_row(p)
			return table
		docx_param = docx_parameters = docx_params




