#!/usr/bin/env python-mpi
from __future__ import division

from operator import add
from functools import reduce

import sys,os
import logging

from lenstools import ConvergenceMap
from lenstools.index import *
from lenstools.statistics import Ensemble
from lenstools.pipeline import SimulationBatch

import numpy as np
import astropy.units as u
from mpi4py import MPI

from emcee.utils import MPIPool

##########################################################################################################################
##############This function reads in a ConvergenceMap and measures all the descriptors provided in index##################
##########################################################################################################################

def convergence_measure_all(filename,index,mean_subtract,smoothing_scale=None):

	"""
	Measures all the statistical descriptors of a convergence map as indicated by the index instance
	
	"""

	logging.info("Processing {0}".format(filename))

	#Load the map
	conv_map = ConvergenceMap.load(filename)

	if mean_subtract:
		conv_map.data -= conv_map.mean()

	#Smooth the map maybe
	if smoothing_scale is not None:
		logging.info("Smoothing {0} on {1}".format(filename,smoothing_scale))
		conv_map.smooth(smoothing_scale,kind="gaussianFFT",inplace=True)

	#Allocate memory for observables
	descriptors = index
	observables = np.zeros(descriptors.size)

	#Measure descriptors as directed by input
	for n in range(descriptors.num_descriptors):
		
		if type(descriptors[n]) == PowerSpectrum:
			l,observables[descriptors[n].first:descriptors[n].last] = conv_map.powerSpectrum(descriptors[n].l_edges)
			
		elif type(descriptors[n]) == Moments:
			observables[descriptors[n].first:descriptors[n].last] = conv_map.moments(connected=descriptors[n].connected)
			
		elif type(descriptors[n]) == Peaks:
			v,observables[descriptors[n].first:descriptors[n].last] = conv_map.peakCount(descriptors[n].thresholds,norm=descriptors[n].norm)

		elif type(descriptors[n]) == PDF:
			v,observables[descriptors[n].first:descriptors[n].last] = conv_map.pdf(descriptors[n].thresholds,norm=descriptors[n].norm)
		
		elif type(descriptors[n]) == MinkowskiAll:
			v,V0,V1,V2 = conv_map.minkowskiFunctionals(descriptors[n].thresholds,norm=descriptors[n].norm)
			observables[descriptors[n].first:descriptors[n].last] = np.hstack((V0,V1,V2))
		
		elif type(descriptors[n]) == MinkowskiSingle:
			raise ValueError("Due to computational performance you have to measure all Minkowski functionals at once!")
		
		else:
			
			raise ValueError("Measurement of this descriptor not implemented!!!")

	#Return
	return observables

##########################################################################################################################
##############This function measures all the descriptors in a map set#####################################################
##########################################################################################################################

def measure_from_set(filename,map_set,index,mean_subtract=False,smoothing_scale=None):
	return map_set.execute(filename,callback=convergence_measure_all,index=index,mean_subtract=mean_subtract,smoothing_scale=smoothing_scale)


#################################################################################
##############Main execution#####################################################
#################################################################################

if __name__=="__main__":

	logging.basicConfig(level=logging.INFO)
	smoothing_scale = float(sys.argv[1]) * u.arcmin

	#Initialize MPIPool
	try:
		pool = MPIPool()
	except:
		pool = None

	if (pool is not None) and not(pool.is_master()):
		
		pool.wait()
		pool.comm.Barrier()
		MPI.Finalize()
		sys.exit(0)

	#Where 
	model_id = "Om0.260_Ol0.740_Ob0.046_w-1.000_ns0.960_si0.800"
	redshift = 38.00

	#What to measure
	l_edges = np.logspace(2.0,np.log10(1.0e4),20)
	v_pk = np.linspace(-0.07,0.5,50)
	v_mf = np.linspace(-0.07,0.5,50)

	#How many realizations
	num_realizations = 1024
	chunks = 1
	realizations_per_chunk = num_realizations // chunks

	#Build the index
	descriptor_list = list()
	descriptor_list.append(PowerSpectrum(l_edges))
	#descriptor_list.append(Moments())
	#descriptor_list.append(Peaks(v_pk))
	#descriptor_list.append(MinkowskiAll(v_mf))
	#descriptor_list.append(PDF(v_mf))

	idx = Indexer.stack(descriptor_list)

	#Get a handle on the simulation batch
	batch = SimulationBatch.current()
	logging.info("Measuring descriptors for simulation batch at {0}".format(batch.environment.home))

	#Get a handle on the collection
	model = batch.getModel(model_id)
	collection = model.getCollection(box_size=600.0*model.Mpc_over_h,nside=1024)

	#Save for reference
	np.save(os.path.join(collection.home_subdir,"ell.npy"),0.5*(l_edges[1:]+l_edges[:-1]))
	np.save(os.path.join(collection.home_subdir,"th_peaks.npy"),0.5*(v_pk[1:]+v_pk[:-1]))
	np.save(os.path.join(collection.home_subdir,"th_minkowski.npy"),0.5*(v_mf[1:]+v_mf[:-1]))

	#Perform the measurements for all the map sets
	for map_set in collection.mapsets:

		#Log to user
		logging.info("Processing map set {0}".format(map_set.settings.directory_name))

		#Construct an ensemble for each map set
		ensemble_all = list()

		#Measure the descriptors spreading calculations on a MPIPool
		for c in range(chunks):
			ensemble_all.append(Ensemble.fromfilelist([ "WLconv_z{0:.2f}_{1:04d}r.fits".format(redshift,r+1) for r in range(realizations_per_chunk*c,realizations_per_chunk*(c+1)) ]))
			ensemble_all[-1].load(callback_loader=measure_from_set,pool=pool,map_set=map_set,index=idx,smoothing_scale=smoothing_scale)

		#Merge all the chunks
		ensemble_all = reduce(add,ensemble_all)

		#Split ensemble into individual descriptors
		for n,ens in enumerate(ensemble_all.split(idx)):

			savename = os.path.join(map_set.home_subdir,idx[n].name+"_s{0}.npy".format(int(smoothing_scale.value)))
			logging.info("Writing {0}".format(savename))
			ens.save(savename)

	#Close pool and quit
	if pool is not None:
		
		pool.close()
		pool.comm.Barrier()
		MPI.Finalize()
	
	sys.exit(0)





