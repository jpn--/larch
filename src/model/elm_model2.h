/*
 *  elm_model2.h
 *
 *  Copyright 2007-2015 Jeffrey Newman 
 *
 *  Larch is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  Larch is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with Larch.  If not, see <http://www.gnu.org/licenses/>.
 *  
 */

#ifndef __ELM_MODEL2_H__
#define __ELM_MODEL2_H__

#ifdef SWIG
#define NOSWIG(x) 
#define FOSWIG(x) x
#else
#define NOSWIG(x) x
#define FOSWIG(x) 
#endif // def SWIG



#ifndef SWIG

#include "etk.h"
#include "elm_vascular.h"
#include "elm_sql_scrape.h"
#include "sherpa.h"
#include "elm_runstats.h"
#include "elm_parameter2.h"
#include "elm_inputstorage.h"
#include "elm_model2_options.h"
#include "elm_packets.h"
#include "elm_darray.h"
#include "larch_cache.h"

namespace etk {
  class dispatcher;
  class workshop;
}

#define MODELFEATURES_NESTING       0x1
#define MODELFEATURES_ALLOCATION    0x2
#define MODELFEATURES_QUANTITATIVE  0x4

#endif // ndef SWIG


// Include the header for this module in the wrapper
#ifdef SWIG
%{
#include "elm_model2_options.h"
#include "elm_model2.h"
%}

%feature("kwargs", 1) elm::Model2::_set_estimation_statistics;

#endif // def SWIG


namespace elm {

	
	
	class Model2
	: public sherpa
	{

		#ifdef SWIG
		%feature("pythonappend") Model2(elm::Fountain& d) %{
			try:
				self._ref_to_db = args[0]
			except IndexError:
				self._ref_to_db = None
			from .logging import easy_logging_active
			if easy_logging_active():
				self.logger(True)
			try:
				self._pull_graph_from_db()
			except LarchError:
				pass
			from .util import dicta
			self.descriptions = dicta()
		%}
		%feature("pythonappend") delete_data_fountain() %{
			self._ref_to_db = None
		%}
		%feature("pythonprepend") setUp %{
			if self._ref_to_db is not None and self.is_provisioned()==0 and and_load_data:
				self.provision()
				self.setUpMessage = "autoprovision yes (setUp)"
				if self.logger(): self.logger().info("autoprovisioned data from database")
		%}
		%feature("pythonprepend") estimate %{
			if self._ref_to_db is not None and self.is_provisioned()==0:
				self.provision()
				self.setUpMessage = "autoprovision yes (estimate)"
				if self.logger(): self.logger().info("autoprovisioned data from database")
		%}
		#endif // def SWIG


#ifndef SWIG

	public:
	
		friend class elm::ComponentList;
		
	public:
		ParameterList* _self_as_ParameterListPtr();
				
	public:		
		VAS_System Xylem;
		// this is the nesting structure for the model. even MNL models have a
		// xylem structure, if very simple. 


#endif // ndef SWIG


	public:
		const elm::VAS_System& _xylem() {return Xylem;}

		void _sayweakself(const std::string& marker_message="");
		void _setweakself(PyObject* ref_to_self);

	public:		
		elm::cellcode _get_root_cellcode() const;
		void _set_root_cellcode(const elm::cellcode& r);

	public:		
		unsigned _nCases_recall;

#ifndef SWIG
		
		
//////// MARK: MODEL ATTRIBUTES //////////////////////////////////////////////////////
		
	public:
		unsigned nCases;
		// This is the number of cases used in the model. It can be smaller than
		//  the number of cases available in the data, which is called 
		//  bootstrapping. The recall version saves the value last seen in
		//  a provisioning.
		
		unsigned nElementals;
		// This is the number of elemental alternatives.
		
		unsigned nNests;
		// This is the number of nests, excluding elemental alternatives and the
		//  root node.
		
		unsigned nNodes;
		// This is the total number of nodes, including elemental alternatives,
		//  and the root node. It should equal nElementals + nNests + 1.
				
		unsigned nThreads;
		// How many threads should be spawned for running long-time calculations?
		
	public:
		unsigned features;
//		bool use_network() const;
//		bool use_quantitative() const;
		
//////// MARK: PARAMETER ARRAYS //////////////////////////////////////////////////////
				
	protected:
		paramArray  Params_UtilityCA  ;
		paramArray  Params_UtilityCO  ;
		paramArray  Params_SamplingCA ;
		paramArray  Params_SamplingCO ;
		paramArray  Params_QuantityCA ; 
		paramArray  Params_QuantLogSum;
		paramArray  Params_LogSum     ;
		paramArray  Params_Edges      ;

		// hold the linkages to the freedoms for the various parameters
		//  used to create the model parmeters at each iteration from the freedoms
		
		void pull_and_exp_from_freedoms(const paramArray& par,       double* ops, const double* fr, const bool& log=false);
		void pull_from_freedoms(const paramArray& par,       double* ops, const double* fr, const bool& log=false);
		void push_to_freedoms  (      paramArray& par, const double* ops,       double* fr);
		void pull_coefficients_from_freedoms();
		void _setUp_coef_and_grad_arrays();

	
	protected:
		etk::ndarray      Coef_UtilityCA  ;
		etk::ndarray      Coef_UtilityCO  ;
		etk::ndarray      Coef_SamplingCA  ;
		etk::ndarray      Coef_SamplingCO  ;
		etk::ndarray      Coef_QuantityCA ;
		etk::ndarray      Coef_QuantLogSum;
		etk::ndarray      Coef_LogSum     ;
		etk::ndarray      Coef_Edges      ;
		// holds the calculated parameters themselves
		
#endif // ndef SWIG

	public:
		virtual void freshen();
	
	public:
		PyObject*  CoefUtilityCA() {return Coef_UtilityCA.get_object();}
		PyObject*  CoefUtilityCO() {return Coef_UtilityCO.get_object();}

		const etk::ndarray* Coef(const std::string& label);
			
#ifndef SWIG
				
		
//////// MARK: DATA ARRAYS ///////////////////////////////////////////////////////////

	public:
//		Facet* _Data;
		Fountain* _Fount;
		inline Fountain* _fountain() {return _Fount;}
		inline const Fountain* _fountain() const {return _Fount;}
		// a pointer to the relevant master data object

#endif // ndef SWIG

	public:
		#ifdef SWIG
		%feature("pythonappend") needs() const %{
			temp = {}
			for i,j in val.items(): temp[i] = j
			val = temp
		%}
		%feature("pythonprepend") provision %{
			if len(args)==0:
				if hasattr(self,'db') and isinstance(self.db,(DB,DT)):
					args = (self.db.provision(self.needs()), )
				else:
					raise LarchError('model has no db specified for provisioning')
		%}
		#endif // def SWIG
		std::map<std::string, elm::darray_req> needs() const;
		void provision(const std::map< std::string, boosted::shared_ptr<const elm::darray> >&);
		void provision();
		int is_provisioned(bool ex=false) const;
	private:
		std::string _subprovision(const std::string& name, boosted::shared_ptr<const darray>& storage,
							 	  const std::map< std::string, boosted::shared_ptr<const darray> >& input,
								  const std::map<std::string, darray_req>& need,
								  std::map<std::string, size_t>& ncases);
		int _is_subprovisioned(const std::string& name, const elm::darray_ptr& arr, const std::map<std::string, darray_req>& requires, const bool& ex) const;

	public:
		const elm::darray* Data(const std::string& label);
		elm::darray* DataEdit(const std::string& label);

		elm::darray_export_map Data_UtilityCE;
		elm::darray_export_map Data_UtilityCE_builtin;
		

#ifndef SWIG

	public:
		elm::darray_ptr  Data_UtilityCA;
		elm::darray_ptr  Data_UtilityCO;
		elm::darray_ptr  Data_SamplingCA;
		elm::darray_ptr  Data_SamplingCO;
		elm::darray_ptr  Data_Allocation;
		elm::darray_ptr  Data_QuantityCA;
//		elm::darray_ptr  Data_QuantLogSum;
//		elm::darray_ptr  Data_LogSum;
		
		elm::darray_ptr  Data_Choice;
		elm::darray_ptr  Data_Weight;
		elm::darray_ptr  Data_Avail;


	private:
		void scan_for_multiple_choices();
		
		
	
	public:
		etk::bitarray Data_MultiChoice; // For each observations, is the choice unique and binary?
		
		boosted::shared_ptr<elm::darray>  Data_Weight_rescaled;
		
		
		
		inline elm::darray_ptr Data_Weight_active() {return (Data_Weight_rescaled ? Data_Weight_rescaled : Data_Weight);}


//		elm::darray_ptr Darray_UtilityCA;
//		elm::darray_ptr Darray_UtilityCO;
//		elm::darray_ptr Darray_SamplingCA;
//		elm::darray_ptr Darray_SamplingCO;
//		
//		elm::darray_ptr Darray_Choice;
//		elm::darray_ptr Darray_Weight;
//		elm::darray_ptr Darray_Avail;
		
	
	public:
		
		double weight_scale_factor;
#endif // ndef SWIG
		double get_weight_scale_factor() const;
		std::string auto_rescale_weights(const double& mean_weight=1.0);
		void restore_scale_weights();
#ifndef SWIG

		
//		std::string weight_CO_variable;
//		bool weight_autorescale;
		
		
		// KEEP
		std::string choseness_CA_variable;
		std::string availability_ca_variable;
						
//////// MARK: CALCULATION ARRAYS ////////////////////////////////////////////////////
		
	public:
		etk::ndarray Utility;
		etk::ndarray Probability;
		etk::ndarray Cond_Prob;
		etk::ndarray Allocation;
		etk::ndarray Quantity;

		etk::ndarray CaseLogLike;

		etk::ndarray SamplingWeight;
		etk::ndarray AdjProbability;
		
		etk::memarray_raw Workspace;
		
		etk::memarray_raw Grad_UtilityCA  ;
		etk::memarray_raw Grad_UtilityCO  ;
		etk::memarray_raw Grad_QuantityCA ;
		etk::memarray_raw Grad_QuantLogSum;
		etk::memarray_raw Grad_LogSum     ;
		
//////// MARK: CALCULATION METHODS ///////////////////////////////////////////////////

	public:
		virtual double objective();
		virtual const etk::memarray& gradient (const bool& force_recalculate=false) ;


//		virtual void calculate_hessian();

		void calculate_probability();
		void calculate_utility_only();



		cache_set _cached_results;


#endif // ndef SWIG

	public:
		void clear_cache();
		
		double loglike();
//		double loglike_cached();
//		double loglike_nocache();
//		double loglike(std::vector<double> v);
//		double loglike_cached(std::vector<double> v);
//		double loglike_nocache(std::vector<double> v);
		std::shared_ptr<etk::ndarray> loglike_casewise();
		std::shared_ptr<etk::ndarray> loglike_casewise(std::vector<double> v);
		
		std::shared_ptr<etk::ndarray> _gradient_casewise();
		std::shared_ptr<etk::ndarray> _gradient_casewise(std::vector<double> v);

		std::shared_ptr<etk::ndarray> finite_diff_gradient () ;
		std::shared_ptr<etk::ndarray> finite_diff_gradient (std::vector<double> v) ;

//		std::shared_ptr<etk::ndarray> calc_utility(datamatrix_t* uco, datamatrix_t* uca=nullptr, datamatrix_t* av=nullptr) const;
		std::shared_ptr<etk::ndarray> calc_utility(etk::ndarray* utilitydataco, etk::ndarray* utilitydataca=nullptr, etk::ndarray* availability=nullptr) const;

	private:
		std::shared_ptr<etk::ndarray> _calc_utility(const etk::ndarray* utilitydataco, const etk::ndarray* utilitydataca, const etk::ndarray* availability) const;

	public:
		std::shared_ptr<etk::ndarray> calc_utility() const;
		std::shared_ptr<etk::ndarray> calc_utility(const std::map< std::string, boosted::shared_ptr<const elm::darray> >&);
		
		std::shared_ptr<etk::ndarray> calc_probability(etk::ndarray* u) const;
		std::shared_ptr<etk::ndarray> calc_logsums(etk::ndarray* u) const;
		
//		std::shared_ptr<etk::ndarray> calc_utility_probability(datamatrix_t* uco, datamatrix_t* uca=nullptr, datamatrix_t* av=nullptr) const;
		std::shared_ptr<etk::ndarray> calc_utility_probability(etk::ndarray* utilitydataco, etk::ndarray* utilitydataca=nullptr, etk::ndarray* availability=nullptr) const;
//		std::shared_ptr<etk::ndarray> calc_utility_logsums(datamatrix_t* uco, datamatrix_t* uca=nullptr, datamatrix_t* av=nullptr) const;
		std::shared_ptr<etk::ndarray> calc_utility_logsums(etk::ndarray* utilitydataco, etk::ndarray* utilitydataca=nullptr, etk::ndarray* availability=nullptr) const;


	public:
		double loglike_given_utility( );
		std::shared_ptr<etk::ndarray> negative_d_loglike_given_utility ();


#ifdef SWIG
		// in the swig-exposed verion of this function, always update_freedoms
		void calculate_parameter_covariance();
#endif

		std::shared_ptr<etk::ndarray> _mnl_gradient_full_casewise();
		std::shared_ptr<etk::ndarray> _ngev_gradient_full_casewise();
		

#ifndef SWIG
		// in the internal verion of this function, allow update_freedoms or not
		void calculate_parameter_covariance(bool update_freedoms=true);


		void mnl_probability();
//		void mnl_gradient   ();
		void mnl_gradient_v2();
		//void mnl_hessian    ();

		void nl_probability();
		void nl_gradient   ();

		void ngev_probability();
		void ngev_probability_given_utility();
		void ngev_gradient   ();

		void calculate_hessian_and_save();
		
		//bool any_holdfast() ;
		//size_t count_holdfast() ;
		//void hessfull_to_hessfree(const etk::symmetric_matrix* full_matrix, etk::symmetric_matrix* free_matrix) ;
		//void hessfree_to_hessfull(etk::symmetric_matrix* full_matrix, const etk::symmetric_matrix* free_matrix) ;
	
		
//		std::vector<sidecar*> sidecars;

		boosted::shared_ptr<etk::dispatcher> probability_dispatcher;
		boosted::shared_ptr<etk::dispatcher> probability_given_utility_dispatcher;
		boosted::shared_ptr<etk::dispatcher> gradient_dispatcher;
		boosted::shared_ptr<etk::dispatcher> loglike_dispatcher;
		
		boosted::shared_ptr<etk::workshop> make_shared_workshop_accumulate_loglike ();
		boosted::shared_ptr<etk::workshop> make_shared_workshop_mnl_probability ();
		boosted::shared_ptr<etk::workshop> make_shared_workshop_nl_probability ();
		boosted::shared_ptr<etk::workshop> make_shared_workshop_nl_gradient ();
		boosted::shared_ptr<etk::workshop> make_shared_workshop_ngev_probability ();
		boosted::shared_ptr<etk::workshop> make_shared_workshop_ngev_gradient ();

	private:
		// Private variables to use in likelihood accumulation
		double accumulate_LogL;
		etk::ndarray* PrToAccum;

	protected:
		
//		void case_gradient_mnl_layer(const unsigned& c, double* d_LL, const double& alpha, const double* cProbTotal);
//		void case_gradient_mnl(const unsigned& c);
//		void case_gradient_mnl_multichoice(const unsigned& c);
		double accumulate_log_likelihood() /*const*/;
		
	private:
		
		
//////// MARK: MODEL OPTIONS /////////////////////////////////////////////////////////

	public:
//		bool is_option(const std::string& optname) const;
//		bool bool_option(const std::string& optname) const;
//		unsigned unsigned_option(const std::string& optname) const;
//		int int_option(const std::string& optname) const;
//		double double_option(const std::string& optname) const;
//		std::string string_option(const std::string& optname) const;
	protected:
		etk::ss_map _options;
		void process_options();
		static std::set<std::string> _valid_options;
	public:
//		void setOption(const std::string& optname, const std::string& optvalue="");
//		std::string getOption(const std::string& optname) const;
		/*
		 This map lists the various options for the model:
		 
		directory
			A path to the working directory. By default, the modelfile, datafile,
			and logfile are all in this directory, but this can be changed.		
		modelfile
			The relative path (from directory) to the location of the model file		
		datafile
			The relative path (from directory) to the location of the data file		
		logfile
			The relative path (from directory) to the location of the log file
		*/
	public:
		static std::set<std::string> valid_options();
		static bool is_valid_option(const std::string& optname);
		
	// Linking options to a python dict is causing errors
//	protected:
//		PyObject* Option;
//	public:
//		void _Option(PyObject* x);
//	
//		double Option_double (const std::string& optname) const;
//		long   Option_integer(const std::string& optname) const;
//		bool   Option_bool   (const std::string& optname) const;
		


//////// MARK: SETUP /////////////////////////////////////////////////////////////////

	protected:
//		void _setUp_availability_data();
//		void _setUp_choice_data();
//		void _setUp_weight_data();
		void _setUp_utility_data_and_params();
		void _setUp_quantity_data_and_params();
		void _setUp_samplefactor_data_and_params();
		void _setUp_allocation_data_and_params();
		
		void _setUp_MNL();
		void _setUp_QMNL();
		void _setUp_NL();
		void _setUp_NGEV();

	
	public:
		int _is_setUp;
		void _parameter_update();
		void _parameter_push(const std::vector<double>& v);
		void _parameter_log();
		std::string _parameter_report() const;

//////// MARK: RECORDED RESULTS //////////////////////////////////////////////////////

	protected:
		double _LL_null;
		double _LL_nil;
		double _LL_constants;
		runstats _latest_run;
		
	public:
		runstats& RunStatistics();


#endif // ndef SWIG

	public:
		void write_runstats_note(const std::string& comment);
		std::string read_runstats_notes() const;

//////// MARK: TIMING INTERFACE //////////////////////////////////////////////////////

	public:
		void start_timing(const std::string& name);
		void finish_timing();

#ifndef SWIG
		
//////// MARK: USER FUNCS ////////////////////////////////////////////////////////////


	
	private:
		std::string _add_utility_co(const std::string& column_name, 
									const std::string& alt_name, 
									const long long&   alt_code, 
									std::string        freedom_name, 
									const double&      freedom_multiplier);
	
	public:
//		void CleanUtilityCA();
//		void CleanUtilityCO();



//		std::string remove_linear_utility (const std::string& variable_name);

//		std::string add_quantity (const std::string& variable_name, 
//								   const std::string& freedom_name, 
//								   const double& freedom_multiplier=1.0);
//		std::string remove_quantity (const std::string& variable_name);
		
//		std::string add_alloc(const std::string& parentCode,
//						 const std::string& childCode,
//						 const std::string& variable_name,
//						 const std::string& freedom_name, 
//						 const double& freedom_multiplier=1.0);
		
//		std::string add_quant_scale (const unsigned long long& alt_code,
//						 const std::string& freedom_name,
//						 const double& constant_value=1.0);

//		std::string add_alloc(const unsigned long long& parentCode,
//						 const unsigned long long& childCode,
//						 const std::string& variable_name,
//						 const std::string& freedom_name, 
//						 const double& freedom_multiplier=1.0);
		

		
	public:
//		std::string weight(const std::string& varname, bool reweight=false);
//		std::string choice(const std::string& varname); // idca
//		void avail(const std::string& varname);

		PyObject*	_get_choice() const;
		PyObject*	_get_weight() const;
		PyObject*	_get_avail()  const;

//		std::string consistency_check();


	public:
//		void sim_probability
//			( const std::string& tablename
//			, const std::string& columnnameprefix="prob_"
//			, const std::string& columnnamesuffix=""
//			, bool use_alt_codes=true
//			, bool overwrite=false
//			);
//		void sim_conditional_probability
//			( const std::string& tablename
//			, const std::string& columnnameprefix="cprob_"
//			, const std::string& columnnamesuffix=""
//			, bool use_alt_codes=true
//			, bool overwrite=false
//			);
//		void simulate_conditional_probability
//			( const std::string& tablename
//			, const std::string& columnnameprefix="cprob_"
//			, const std::string& columnnamesuffix=""
//			, bool use_alt_codes=true
//			, bool overwrite=false
//			);
//		void _simulate_choices
//			( const std::string& tablename
//			, const std::string& choicecolumnname="simchoice"
//			, const std::string& casecolumnname="caseid"
//			, const std::string& altcolumnname="altid"
//			, const std::string& choices_per_case_columnname=""
//			, bool overwrite=false
//			);



//		runstats recall_estimate() const;
//		double LL_null();
//		double LL_here();

		
		std::vector< std::string > parameter_names() const;


								
								
		/// Direct memory access from python ndarrays
		etk::ndarray* adjprobability(etk::ndarray* params=nullptr);
		etk::ndarray* utility(etk::ndarray* params=nullptr);

		ca_co_packet utility_packet();
		ca_co_packet utility_packet_without_data();
		ca_co_packet quantity_packet();
		ca_co_packet sampling_packet();
		ca_co_packet allocation_packet();

		// note: nest and link are not SWIG'd, they are replaced below
		ELM_RESULTCODE nest(const std::string& nest_name, const elm::cellcode& nest_code,
						 std::string freedom_name="", const double& freedom_multiplier=1.0);
		std::string link (const long long& parentCode, const long long& childCode);



#endif // ndef SWIG

		std::vector< std::string > alias_names() const;

		etk::ndarray* probability(etk::ndarray* params=nullptr);


//////// MARK: SWIG EXPOSED FUNCS ////////////////////////////////////////////////////////////
		
	public:
		etk::symmetric_matrix* _get_hessian_array();
		void _set_hessian_array(etk::symmetric_matrix* in);
		void _del_hessian_array();


	public:
		std::vector<double> parameter_values() const;
		void parameter_values(std::vector<double> v);
		
		PyObject* parameter_values_as_bytes() const;
		
		void utilityca (const std::string& column_name,
							   std::string freedom_name="", 
							   const double& freedom_multiplier=1.0);
		void utilityco (const std::string& column_name,
							   const std::string& alt_name,
							   std::string freedom_name="", 
							   const double& freedom_multiplier=1.0);
		void utilityco (const std::string& column_name, 
							   const long long& alt_code,
							   std::string freedom_name="", 
							   const double& freedom_multiplier=1.0);

	public:
		#ifdef SWIG
		%feature("shadow") Model2::Input_Graph() %{
		def _Model2_Input_Graph_NoSet(self, value):
			raise LarchError("cannot change model.graph directly, edit model.nest or model.link")
		__swig_setmethods__["graph"] = _Model2_Input_Graph_NoSet
		__swig_getmethods__["graph"] = _core.Model2_Input_Graph
		@property
		def graph(self):
			return $action(self)
		%}
		%rename(utility) Input_Utility;
		%rename(quantity) Input_QuantityCA;
		%rename(nest) Input_LogSum;
		%rename(link) Input_Edges;
		%rename(samplingbias) Input_Sampling;
		#endif // def SWIG

		LinearBundle_1              Input_Utility;
		ComponentList               Input_QuantityCA ;
		ComponentCellcodeMap        Input_LogSum;
		LinearCOBundle_2            Input_Edges;
		LinearBundle_1              Input_Sampling;
		ComponentGraphDNA           Input_Graph();

	public:
NOSWIG(	PyObject*         logger (const std::string& logname);)
NOSWIG(	PyObject*         logger (bool z);)
NOSWIG(	PyObject*         logger (int z);)
		PyObject*         logger (PyObject* l=nullptr);
		
		etk::string_sender* _string_sender_ptr;

		model_options_t option;


	//	etk::ndarray* tally_chosen() const;
	//	etk::ndarray* tally_avail() const;
	//	etk::ndarray* tally_chosen();
	//	etk::ndarray* tally_avail();


		runstats _maximize_bhhh();

		double loglike_null();

		runstats estimate();
NOSWIG(	runstats estimate(std::vector<sherpa_pack>* opts); )
		runstats estimate(std::vector<sherpa_pack> opts);
		runstats estimate_tight(double magnitude=8);

		/// The _get* functions are used to get attributes of the model for saving/pickling
		PyObject* _get_parameter() const;
		PyObject* _get_nest() const;
		PyObject* _get_link() const;
		PyObject* _get_utilityca() const;
		PyObject* _get_utilityco() const;
		PyObject* _get_samplingbiasca() const;
		PyObject* _get_samplingbiasco() const;
	    PyObject* _get_logger() const; 

		PyObject* _get_estimation_statistics () const;
		PyObject* _get_estimation_run_statistics () const;
		void _set_estimation_statistics(const double& log_like=NAN,	const double& log_like_null=NAN, const double& log_like_nil=NAN,
										const double& log_like_constants=NAN,	const double& log_like_best=NAN
										);
		void _set_estimation_run_statistics_pickle(PyObject* dict);

	public:
FOSWIG(	%rename(__repr__) representation; )
		std::string prints(const unsigned& precision=5, const unsigned& cell_width=11) const;
		std::string full_report(const unsigned& precision=5, const unsigned& cell_width=11);
//		std::string save(const std::string& filename="", const std::string& objectname="m") const;
		std::string representation() const;
		

//////// MARK: CONSTRUCTOR ///////////////////////////////////////////////////////////

		#ifdef SWIG
		%typemap(out) elm::Model2* {
		   ($1)->weakself = $result = SWIG_NewPointerObj(SWIG_as_voidptr(($1)), $1_descriptor, SWIG_POINTER_NEW |  0 );
		}
		#endif // def SWIG
		
	public:
		Model2();
		Model2(elm::Fountain& d);
		~Model2();

		void delete_data_fountain();
		void change_data_fountain(elm::Fountain& datafile);

	public:
		void setUp(bool and_load_data=true);
		void _pull_graph_from_db();
		
		std::string setUpMessage;
		
		virtual void tearDown();

		std::string title;

		std::string save_buffer() const;
			
	public:
//		void simulate_probability
//			( const std::string& tablename
//			, const std::string& columnnameprefix="prob_"
//			, const std::string& columnnamesuffix=""
//			, bool use_alt_codes=true
//			, bool overwrite=false
//			);
//		void simulate_choices
//			( const std::string& tablename
//			, const std::string& choicecolumnname="simchoice"
//			, const std::string& casecolumnname="caseid"
//			, const std::string& altcolumnname="altid"
//			, const std::string& choices_per_case_columnname=""
//			, bool overwrite=false
//			);


		std::shared_ptr<etk::ndarray> negative_d_loglike() ;
		std::shared_ptr<etk::ndarray> negative_d_loglike(const std::vector<double>& v) ;
		std::shared_ptr<etk::ndarray> negative_d_loglike_cached() ;
		std::shared_ptr<etk::ndarray> negative_d_loglike_cached(const std::vector<double>& v) ;
		std::shared_ptr<etk::ndarray> negative_d_loglike_nocache() ;
		std::shared_ptr<etk::ndarray> negative_d_loglike_nocache(const std::vector<double>& v) ;

		std::shared_ptr<etk::symmetric_matrix> bhhh_cached();
		std::shared_ptr<etk::symmetric_matrix> bhhh_cached(const std::vector<double>& v);
		std::shared_ptr<etk::symmetric_matrix> bhhh_nocache();
		std::shared_ptr<etk::symmetric_matrix> bhhh_nocache(const std::vector<double>& v);
		std::shared_ptr<etk::symmetric_matrix> bhhh();
		std::shared_ptr<etk::symmetric_matrix> bhhh(const std::vector<double>& v);

		std::shared_ptr<etk::ndarray> bhhh_direction();
		double bhhh_tolerance();
		double bhhh_tolerance_nocache();

		std::shared_ptr<etk::ndarray> bhhh_direction(const std::vector<double>& v);
		double bhhh_tolerance(const std::vector<double>& v);
		double bhhh_tolerance_nocache(const std::vector<double>& v);

	};

	#ifndef SWIG
	etk::strvec __identify_needs(const ComponentList& Input_List);
	etk::strvec __identify_needs(const LinearCOBundle_1& Input_List);
	etk::strvec __identify_needs(const LinearCOBundle_2& Input_EdgeMap);
	void __identify_additional_needs(const ComponentList& Input_List, etk::strvec& needs);
	#endif // ndef SWIG

		
} // end namespace elm

#ifdef SWIG
%include "elm_model2_extend.h"			
#endif // def SWIG
			

#endif // __ELM_MODEL2_H__

