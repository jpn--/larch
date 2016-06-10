



%extend Fountain {
%pythoncode %{

	def dataframe_idco(self, *vars, **kwargs):
		"""
		Load a selection of :ref:`idco` data into a :class:`pandas.DataFrame`.
		
		This function passes all parameters through to :meth:`Fountain.array_idco`.
		"""
		data = self.array_idco(*vars, **kwargs)
		import pandas
		df = pandas.DataFrame(data=data, index=self.caseids(), columns=vars, dtype=None, copy=False)
		return df
		
	def dataframe_idca(self, *vars, wide=False, **kwargs):
		"""
		Load a selection of :ref:`idca` data into a :class:`pandas.DataFrame`.
		
		:param wide: If True (defaults False), the resulting data array will be pivoted to be :ref:`idco`,
					with one row per case and a hierarchical columns definition.
		:type wide:  bool
	
		This function passes all other parameters through to :meth:`Fountain.array_idca`.
		
		"""
		import pandas
		data = self.array_idca(*vars, **kwargs)
		data = data.reshape(-1,data.shape[-1])
		mi = pandas.MultiIndex.from_product([self.caseids(), self.alternative_codes()], names=['caseid', 'altid'])
		df = pandas.DataFrame(data=data, index=mi, columns=vars, dtype=None, copy=False)
		if wide:
			return df.unstack(level=-1)
		else:
			return df

	def dataframe_all(self):
		"""
		Load all data (idca and idco) to one big idco format :class:`pandas.DataFrame`.
		
		No effort is made to prevent duplication of data in this DataFrame.
		(e.g. if there are idco variables stacked to make a single idca
		variable, these will appear in the output twice).  If there is a
		lot of data, this DataFrame could be very large.
		
		"""
		import pandas
		dfco = self.dataframe_idco(*self.variables_co())
		dfca = self.dataframe_idca(*self.variables_ca(), wide=True)
		dfca.columns = ['_'.join(str(c) for c in col).strip() for col in dfca.columns.values]
		df = pandas.concat([dfco, dfca], axis=1)
		for col in df.columns:
			if numpy.all(df[col].astype(int) == df[col]):
				df[col] = df[col].astype(int)
		return df

	def export_all(self, *arg, **kwarg):
		"""
		Export all data (idca and idco) to one big idco format csv file.
		
		This method takes the dataframe from :meth:`dataframe_all` and writes
		it out to a csv file. All arguments are passed through to :meth:`pandas.DataFrame.to_csv`.
		No effort is made to prevent duplication of data in this export
		(e.g. if there are idco variables stacked to make a single idca
		variable, these will appear in the output twice).
		
		"""
		import pandas
		if 'index_label' not in kwarg:
			kwarg['index_label']='caseid'
		self.dataframe_all().to_csv(*arg, **kwarg)
		
%}
};




