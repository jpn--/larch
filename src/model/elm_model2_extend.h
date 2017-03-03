/*
 *  elm_model2.h
 *
 *  Copyright 2007-2017 Jeffrey Newman 
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

#ifndef __ELM_MODEL2_EXTEND_H__
#define __ELM_MODEL2_EXTEND_H__

#ifdef SWIG


namespace elm {

	%extend Model2 {
		
		
		etk::ndarray* Utility() {return &($self->Utility);}
		etk::ndarray* Probability() {return &($self->Probability);}
		etk::ndarray* Cond_Prob() {return &($self->Cond_Prob);}
		etk::ndarray* Allocation() {return &($self->Allocation);}
		etk::ndarray* Quantity() {return &($self->Quantity);}
		
		etk::ndarray* CaseLogLike() {return &($self->CaseLogLike);}
		etk::ndarray* SamplingWeight() {return &($self->SamplingWeight);}
		etk::ndarray* AdjProbability() {return &($self->AdjProbability);}
		
		
		unsigned long long nAlts() const {return $self->nElementals;}
		unsigned long long nCases() const {
			if ($self->nCases) {
				return $self->nCases;
			} else {
				return $self->_nCases_recall;
			}
		}
		
		std::vector<std::string> alternative_names() const {
			return $self->Xylem.elemental_names();
		}
		std::vector<long long> alternative_codes() const {
			return $self->Xylem.elemental_codes();
		}
		
		void _compute_d2_loglike() {
			$self->setUp();
			//if (!$self->_is_setUp) OOPS("Model is not setup, try calling setUp() first.");
			$self->_parameter_update();
			$self->calculate_hessian_and_save();
		}
		void _compute_d2_loglike(std::vector<double> v) {
			$self->setUp();
			//if (!$self->_is_setUp) OOPS("Model is not setup, try calling setUp() first.");
			$self->_parameter_update();
			$self->_parameter_push(v);
			$self->calculate_hessian_and_save();
		}

		
		void teardown() {return $self->tearDown();}
		
	}

#endif // def SWIG
	
} // end namespace elm


#endif // __ELM_MODEL2_EXTEND_H__

