

#ifndef __VASCULAR_SYSTEM_I__
#define __VASCULAR_SYSTEM_I__


namespace elm {

	class VAS_System
	{
	public:

		const unsigned& n_elemental() const;
		const unsigned& n_branches()  const;
		const unsigned& n_edges()     const;
		const unsigned& alloc_break(const unsigned& i) const;

		size_t          n_alloc_break() const ;
		size_t          size()        const ;

		std::string display() const;
		std::string display_phenotype() const;
		std::string display_edges() const;
	
		std::vector<std::string> elemental_names() const;
		std::vector<long long> elemental_codes() const;
		std::vector<long long> all_codes() const;

		int add_cell(const cellcode& cod, const std::string& nam="");
		void regrow(  );
	
	};

}


#endif // __VASCULAR_SYSTEM_I__
