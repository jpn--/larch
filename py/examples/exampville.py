
## Exampville

import numpy, scipy, larch, os
import scipy.stats
from numpy import log, exp, log1p
# flog = larch.logging.flogger(level=30, label="exampville")
def flog(a0, *arg, **kwarg):
	pass
	#print(a0.format(*arg, **kwarg))

__all__ = ['builder',]

_cache_1 = None





def build_year_1(nZones=9, transit_scope = slice(2,8), n_HH = 834, directory=None, seed=0):

	global _cache_1
	if _cache_1 is not None:
		return _cache_1

	flog("EXAMPVILLE Builder (Year 1)")
	flog("  simulating a survey of {} households",n_HH)
	flog("  traveling among {} travel analysis zones",nZones)

	if directory is None:
		from ..util.temporaryfile import TemporaryDirectory
		directory = TemporaryDirectory()

	if isinstance(transit_scope, tuple):
		transit_scope = slice(*transit_scope)

	if transit_scope.stop > nZones:
		raise TypeError('transit_scope too large for nZones')

	# The randomizer seed is reset to zero by default so that we get consistent generated
	# Exampville survey results.  (High levels of randomness are not very important
	# here because we just want to demonstrate the models.)
	numpy.random.seed(seed)

	## Zones
	flog("Zones")
	pop_weight = numpy.fmax(numpy.arange(nZones),numpy.flipud(numpy.arange(nZones))).astype(float)
	pop_weight /= pop_weight.sum()
	wrk_weight = scipy.stats.binom.pmf(numpy.arange(nZones),nZones-1,.5)
	wrk_weight /= wrk_weight.sum()
	
	zone_lat = (-1)**numpy.arange(nZones)
	zone_lon = 11+numpy.arange(nZones)

	## Skims
	flog("Skims")
	distance = numpy.zeros([nZones,nZones])
	gaps = numpy.random.random(nZones)+1
	for z in range(nZones):
		for y in range(z,nZones):
			distance[y,z] = distance[z,y] = gaps[z:y].sum()
		distance[z,z] = numpy.random.random()


	drivetime = (numpy.random.exponential(1,[nZones,nZones])+1) * distance
	drivetime[drivetime<2.0] = 2.0
	hov_time = drivetime.copy()
	hov_time[:nZones//2,nZones//2:] -= numpy.random.random() * 1.5 + 0.5
	hov_time[nZones//2:,:nZones//2] -= numpy.random.random() * 1.5 + 0.5

	transit_range = transit_scope.stop-transit_scope.start
	transittime = numpy.zeros([nZones,nZones])
	transittime[transit_scope,transit_scope] = (numpy.random.random([transit_range,transit_range])+1) * distance[transit_scope,transit_scope]

	transitfare = numpy.zeros([nZones,nZones])
	transitfare[transit_scope,transit_scope] = 1.5



	## Households
	flog("HHs")
	HHidx = numpy.arange(n_HH, dtype=numpy.int64)
	HHid = numpy.asarray([50000+i for i in HHidx])
	HHincome = numpy.round( numpy.random.normal(75000,25000,[n_HH,]), -3 ).astype(numpy.int64)
	HHsize = numpy.floor(numpy.random.exponential(0.8,[n_HH,])+1+numpy.random.random([n_HH,])).astype(numpy.int64)
	HHhomezone = numpy.random.choice(numpy.arange(1,nZones+1), size=[n_HH,], replace=True, p=pop_weight)




	## People
	flog("People")
	n_PER = numpy.sum(HHsize)
	PERidx = numpy.arange(n_PER, dtype=numpy.int64)
	PERid = numpy.asarray([60000+i for i in PERidx])
	PERhhid = numpy.zeros(n_PER, dtype=numpy.int64)
	PERhhidx = numpy.zeros(n_PER, dtype=numpy.int64)
	n2 = 0
	for n1 in range(n_HH):
		PERhhid[n2:(n2+HHsize[n1])] = HHid[n1]
		PERhhidx[n2:(n2+HHsize[n1])] = HHidx[n1]
		n2 += HHsize[n1]
	PERage = (numpy.random.random(n_PER)*80+5).astype(numpy.int64)
	PERworks = ((numpy.random.random(n_PER)>0.2) & (PERage > 16) & (PERage < 70)).astype(numpy.int64)

	zone_employment = numpy.round( PERworks.sum()*wrk_weight, 0 )+1

	total_employment = zone_employment.sum()
	mean_employment = total_employment/nZones
	zone_retail = numpy.fmin(numpy.round(numpy.random.random(nZones) * mean_employment,0), zone_employment)
	zone_nonretail = zone_employment - zone_retail

	PERnworktours = numpy.random.choice([0,1,2,3], size=[n_PER,], replace=True, p=[0.1,0.8,0.07,0.03]) * PERworks
	PERnothertours = numpy.random.choice([0,1,2,3], size=[n_PER,], replace=True, p=[0.2,0.5,0.2,0.1])
	PERntours = PERnworktours + PERnothertours

	## Tours
	flog("Tours")

	n_TOUR = PERntours.sum()

	TOURid = numpy.arange(n_TOUR, dtype=numpy.int64)
	TOURper = numpy.zeros(n_TOUR, dtype=numpy.int64)
	TOURperidx = numpy.zeros(n_TOUR, dtype=numpy.int64)
	TOURhh = numpy.zeros(n_TOUR, dtype=numpy.int64)
	TOURhhidx = numpy.zeros(n_TOUR, dtype=numpy.int64)
	TOURdtaz = numpy.zeros(n_TOUR, dtype=numpy.int64)
	TOURmode = numpy.zeros(n_TOUR, dtype=numpy.int64)
	TOURpurpose = numpy.zeros(n_TOUR, dtype=numpy.int64)

	# Work tours, then other tours
	n2 = 0
	for n1 in range(n_PER):
		TOURper[n2:(n2+PERntours[n1])] = PERid[n1]
		TOURperidx[n2:(n2+PERntours[n1])] = PERidx[n1]
		TOURhh[n2:(n2+PERntours[n1])] = PERhhid[n1]
		TOURhhidx[n2:(n2+PERntours[n1])] = PERhhidx[n1]
		TOURpurpose[n2:(n2+PERnworktours[n1])] = 1
		TOURpurpose[(n2+PERnworktours[n1]):(n2+PERntours[n1])] = 2
		n2 += PERntours[n1]



	#### Utility by mode to various destinations
	flog("Choice Probability")
	nameModes = ['DA','SR','Walk','Bike','Transit']
	mDA = 0
	mSR = 1
	mWA = 2
	mBI = 3
	mTR = 4
	nModes = len(nameModes)

	nModeNests = 3

	paramCOST = -0.312
	paramTIME = -0.123
	paramNMTIME = -0.246
	paramDIST = -0.00357
	paramLNDIST = -0.00642

	paramMUcar = 0.5
	paramMUnon = 0.75
	paramMUmot = 0.8
	paramMUtop = 1.0

	Util = numpy.zeros([n_TOUR, nZones, nModes])
	for n in range(n_TOUR):
		# Mode
#		flog('N {}',n)
#		flog('Util[n,:,:]  &&&')
#		flog('{}',Util[n,:,:])
		otazi = HHhomezone[TOURhhidx[n]]-1
		Util[n,:,mDA] += drivetime[otazi,:] * paramTIME + distance[otazi,:] * 0.20 * paramCOST
		if HHincome[TOURhhidx[n]]>=75000:
			Util[n,:,mDA] += 1.0
			Util[n,:,mTR] -= 0.5
		Util[n,:,mSR] += drivetime[otazi,:] * paramTIME - 1.0 + distance[otazi,:] * 0.20 * 0.5 * paramCOST
		Util[n,:,mWA] += 1.0 + distance[otazi,:] / 2.5 * 60 * paramNMTIME
		Util[n,:,mBI] += -1.25 + distance[otazi,:] / 12 * 60 * paramNMTIME
		Util[n,:,mTR] += -1.5 + transittime[otazi,:] * paramTIME + transitfare[otazi,:] * paramCOST
		# Destination
		Util[n,:,:] += distance[otazi,:,None] * paramDIST + log1p(distance[otazi,:,None]) * paramLNDIST
		if HHincome[TOURhhidx[n]]<=50000:
			Util[n,:,:] += 0.75 * log(zone_retail * 2.71828 + zone_nonretail)[:,None]
		else:
			Util[n,:,:] += 0.75 * log(zone_retail + zone_nonretail * 2.71828)[:,None]
#		flog('Util[n,:,:]  ...')
#		flog('{}',Util[n,:,:])
		# Unavails
		if PERage[TOURperidx[n]] < 16:
			Util[n,:,mDA] = -numpy.inf
		Util[n,transitfare[otazi,:]<=0,mTR] = -numpy.inf
		Util[n,distance[otazi,:]>=3,mWA] = -numpy.inf
		Util[n,distance[otazi,:]>=15,mBI] = -numpy.inf
#		flog('Util[n,:,:]  +++')
#		flog('{}',Util[n,:,:])


	CPr_car = numpy.zeros([n_TOUR, nZones, 2]) # [DA,SR]
	CPr_non = numpy.zeros([n_TOUR, nZones, 2]) # [WA,BI]
	CPr_mot = numpy.zeros([n_TOUR, nZones, 2]) # [TR,Car]
	CPr_top = numpy.zeros([n_TOUR, nZones, 2]) # [Non,Mot]

	NLS_car = numpy.zeros([n_TOUR, nZones, ]) 
	NLS_non = numpy.zeros([n_TOUR, nZones, ]) 
	NLS_mot = numpy.zeros([n_TOUR, nZones, ]) 
	MLS_top = numpy.zeros([n_TOUR, nZones, ]) # Mode choice logsum
	DLS_top = numpy.zeros([n_TOUR, ])         # Dest choice logsum

	Pr_modes = numpy.zeros([n_TOUR, nZones, nModes])
	Pr_dest = numpy.zeros([n_TOUR, nZones])

	for n in range(n_TOUR):
		NLS_car[n,:] = paramMUcar * log( exp(Util[n,:,mDA]/paramMUcar) + exp(Util[n,:,mSR]/paramMUcar) )
		NLS_non[n,:] = paramMUnon * log( exp(Util[n,:,mWA]/paramMUnon) + exp(Util[n,:,mBI]/paramMUnon) )
		NLS_mot[n,:] = paramMUmot * log( exp(NLS_car[n,:] /paramMUmot) + exp(Util[n,:,mTR]/paramMUmot) )
		MLS_top[n,:] = log( exp(NLS_non[n,:]) + exp(NLS_mot[n,:]) )
		DLS_top[n] = log(  numpy.sum( exp( MLS_top[n,:] ) )  )
		
		Pr_dest[n,:] = exp(MLS_top[n,:] - DLS_top[n])
		
		CPr_top[n,:,0] = exp((NLS_non[n,:] - MLS_top[n,:]) / paramMUtop)
		CPr_top[n,:,1] = exp((NLS_mot[n,:] - MLS_top[n,:]) / paramMUtop)
		CPr_mot[n,:,0] = exp((Util[n,:,mTR]- NLS_mot[n,:]) / paramMUmot)
		CPr_mot[n,:,1] = exp((NLS_car[n,:] - NLS_mot[n,:]) / paramMUmot)
		CPr_non[n,:,0] = exp((Util[n,:,mWA]- NLS_non[n,:]) / paramMUnon)
		CPr_non[n,:,1] = exp((Util[n,:,mBI]- NLS_non[n,:]) / paramMUnon)
		CPr_car[n,:,0] = exp((Util[n,:,mDA]- NLS_car[n,:]) / paramMUcar)
		CPr_car[n,:,1] = exp((Util[n,:,mSR]- NLS_car[n,:]) / paramMUcar)
		
		Pr_modes[n,:,mTR] = CPr_mot[n,:,0] * CPr_top[n,:,1] * Pr_dest[n,:]
		Pr_modes[n,:,mWA] = CPr_non[n,:,0] * CPr_top[n,:,0] * Pr_dest[n,:]
		Pr_modes[n,:,mBI] = CPr_non[n,:,1] * CPr_top[n,:,0] * Pr_dest[n,:]
		Pr_modes[n,:,mDA] = CPr_car[n,:,0] * CPr_mot[n,:,1] * CPr_top[n,:,1] * Pr_dest[n,:]
		Pr_modes[n,:,mSR] = CPr_car[n,:,1] * CPr_mot[n,:,1] * CPr_top[n,:,1] * Pr_dest[n,:]

	Pr_modes[numpy.isnan(Pr_modes)] = 0

	## Choices
	flog("Choices")
	for n in range(n_TOUR):
		try:
			ch = numpy.random.choice(nModes*nZones, replace=True, p=Pr_modes[n,:,:].ravel())
		except:
			flog("total prob = {}", Pr_modes[n,:,:].sum())
			raise
		dtazi = ch//nModes
		modei = ch-(dtazi*nModes)
		TOURdtaz[n] = dtazi+1
		TOURmode[n] = modei+1

	### Write Out Data
	flog("Output")

	if not os.path.exists(directory):
		os.makedirs(directory)

	omx = larch.OMX(os.path.join(directory,'exampville.omx'), mode='a')
	omx.shape = (nZones,nZones)
	omx.add_matrix('DIST', distance)
	omx.add_matrix('AUTO_TIME', drivetime)
	omx.add_matrix('RAIL_TIME', transittime)
	omx.add_matrix('RAIL_FARE', transitfare)
	omx.add_lookup('TAZID', numpy.arange(1,nZones+1, dtype=numpy.int64))
	omx.add_lookup('EMPLOYMENT', zone_employment)
	omx.add_lookup('EMP_RETAIL', zone_retail)
	omx.add_lookup('EMP_NONRETAIL', zone_nonretail)
	omx.add_lookup('LAT', zone_lat)
	omx.add_lookup('LON', zone_lon)

	omx.flush()
	omx.close()
	omx = larch.OMX(os.path.join(directory,'exampville.omx'), mode='r')

	f_hh = larch.DT(os.path.join(directory,'exampville_hh.h5'), mode='a')
	f_hh.new_caseids(HHid)
	f_hh.new_idco_from_array('INCOME', HHincome)
	f_hh.new_idco_from_array('HHSIZE', HHsize)
	f_hh.new_idco_from_array('HOMETAZ', HHhomezone)
	f_hh.flush()

	f_pp = larch.DT(os.path.join(directory,'exampville_person.h5'), mode='a')
	f_pp.new_caseids(PERid)
	f_pp.new_idco_from_array('HHID', PERhhid)
	f_pp.new_idco_from_array('AGE', PERage)
	f_pp.new_idco_from_array('WORKS', PERworks)
	f_pp.new_idco_from_array('N_WORKTOURS', PERnworktours)
	f_pp.new_idco_from_array('N_OTHERTOURS', PERnothertours)
	f_pp.new_idco_from_array('N_TOTALTOURS', PERntours)
	f_pp.flush()

	f_tour = larch.DT(os.path.join(directory,'exampville_tours.h5'), mode='a')
	f_tour.new_caseids(TOURid)
	f_tour.new_idco_from_array('HHID', TOURhh)
	f_tour.new_idco_from_array('PERSONID', TOURper)
	f_tour.new_idco_from_array('DTAZ', TOURdtaz)
	f_tour.new_idco_from_array('TOURMODE', TOURmode)
	f_tour.idco.TOURMODE.attrs.DICTIONARY = {
		1:'DA',
		2:'SR',
		3:'Walk',
		4:'Bike',
		5:'Transit',
	}
	f_tour.new_idco_from_array('TOURPURP', TOURpurpose)
	f_tour.idco.TOURPURP.attrs.DICTIONARY = {
		1:'Work Tour',
		2:'Non-Work Tour',
	}
	f_tour.flush()

	flog("EXAMPVILLE Completed Builder (Year 1)")
	flog("   SKIMS  : {}", omx.filename)
	flog("   HHs    : {}", f_hh.source_filename)
	flog("   Persons: {}", f_pp.source_filename)
	flog("   Tours  : {}", f_tour.source_filename)

	_cache_1 = (directory, omx, f_hh, f_pp, f_tour)

	return directory, omx, f_hh, f_pp, f_tour




builder = builder_1 = build_year_1

