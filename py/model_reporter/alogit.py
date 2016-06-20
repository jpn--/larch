


class AlogitModelReporter():


	def alogit_control_file(self):
		"""
		Generate an ALOGIT control file for use with this model.
		
		This feature is entirely experimental.  The user should have no expectation
		that the generated file should be ready to eat by Alogit; some manual 
		tweaking will almost certainly be required for a successful transition.
		"""
	
		import io
		alo = io.StringIO()
		alo.write( "$title {}\n".format(self.title) )
		alo.write( "$subtitle (@file.alo)\n\n" )
		alo.write( "$print transforms\n" )
		alo.write( "$gen.stats utilities all\n" )
		alo.write( "$estimate\n" )

		alo.write( "\n" )
		
		try:
			df_alogit = self.df.alogit_control_file()
		except:
			raise
		else:
			alo.write(df_alogit)

		alo.write( "\n\n" )
		
		for mp in self:
			alo.write( "$COEFF parameter_{} {} {}\n".format(mp.name_, "=" if mp.holdfast else "#", mp.value) )

		g = self.graph

		alo.write( "\n" )

		nesting_nodes = {k:tuple(v.keys()) for k,v in g.succ.items() if len(v)>0}
		for nestcode, downcodes in nesting_nodes.items():
			nestname = g.node[nestcode]['name'].replace(" ","_")
			if nestcode==self.root_id:
				nest_param = ""
			else:
				nest_param = "parameter_" + self.nest[nestcode].param.replace(" ","_")
			members = " ".join("node_"+g.node[j]['name'].replace(" ","_") for j in downcodes)
			alo.write( "$NEST node_{} ({}) {}\n".format(nestname, nest_param, members) )

		alo.write( "\n" )

		# Availability
		# Alogit uses ifeq instead of =, and similar for < > <= >=, so it's easier to
		# just export the computed availability factors.
		#for aname in self.alternative_names():
		#	alo.write( "Avail(node_{0}) = ifeq(avail_{0},1)\n".format(aname.replace(" ","_")) )

		# Choice
		# We will export a column larchchoice that corresponds to the choice "slot"
		# in the alternative_names tuple.
		anames_tuple = tuple(i.replace(" ","_") for i in self.alternative_names())
		#anames = ", ".join(anames_tuple)
		#alo.write( "choice = recode(larchchoice, {})\n".format(anames) )
		#
		#alo.write( "\n" )

		# Array (idca)

		#vars_idca = set(self.df.variables_ca())
		#try:
		#	vars_idca.remove('caseid')
		#except KeyError:
		#	pass
		#try:
		#	vars_idca.remove('altid')
		#except KeyError:
		#	pass
		#
		#if len(vars_idca):
		#	alo.write( "$array" )
		#	for idca_name in vars_idca:
		#		alo.write(" {}(alts)".format(idca_name.replace(" ","_")))
		#	alo.write( "\n" )
		#	for idca_name in vars_idca:
		#		for aname in anames_tuple:
		#			alo.write("\n{0}(node_{1}) = {0}_{1}".format(idca_name.replace(" ","_"), aname, ))
		#
		#	alo.write( "\n" )

		# Utility
		for aname,acode in zip(anames_tuple, self.alternative_codes()):
			padding_size = 0
			alo.write( "\nUTIL(node_{}) = ".format(aname) )
			if acode in self.utility.co:
				for k in self.utility.co[acode]:
					alo.write(" "*padding_size)
					alo.write( "parameter_{} * {} +\n".format(k.param, k.data) )
					padding_size = len("UTIL(node_{}) = ".format(aname))
			for k in self.utility.ca:
				alo.write(" "*padding_size)
				alo.write( "     parameter_{} * {} +\n".format(k.param, k.data) )
				padding_size = len("UTIL(node_{}) = ".format(aname))
			alo.write(" "*padding_size)
			alo.write( "0\n" )

		return alo.getvalue()