

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
		
	};

}


#endif // __VASCULAR_SYSTEM_I__
