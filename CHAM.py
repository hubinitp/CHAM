# created by Xue-Wen LIU, Bin HU @10.23 2017
# This code solves non-linear power spectrum using Halo model with different distribution function including Press-Sheth, Sheth-Tormen, Random Walk Formalism(arXiv:0508384)

import numpy as np
import matplotlib.pyplot as plt
import math, time, sys
import pp
import os, time
from sympy import *
import scipy.interpolate
from scipy.interpolate import spline
from scipy.integrate import quad
from scipy.integrate import romberg
from scipy.signal import savgol_filter
import yaml
import warnings

# Load some external config file
with open('./CHAM_params.yaml') as fp:
    CP = yaml.load(fp)

CP['rho_m'] = CP['omg_m'] * CP['rho_c'] / CP['hub']**3.0

# ignore warnings
warnings.filterwarnings("ignore")

#--------------------------------------------------------------------------------------#
# functions 

# sigma(k) integral function 
def f(logk, r, pk_interp):
        k = np.exp(logk)
        tmp = (k**(3.0))*pk_interp(k)/2.0/np.pi**2
        x = k*r
        win = 3.0/(x**(3.0))*(np.sin(x)-x*np.cos(x))    # sharp-k filter
        f = tmp * win **2
        return f

# compute \sigma  --just need to input the linear p(k) filename from EFCAMB
def sigm(pok_file):
	global r_arr, mass_arr
	num_r = CP['num_r']
	pok_dat = np.loadtxt(pok_file)
	pk_interp = scipy.interpolate.interp1d(pok_dat[:,0], pok_dat[:,1])    # linear P(k)
	logr_step_size = np.log10(CP['r_max'])-np.log10(CP['r_min'])
	logr_step_size = logr_step_size/(num_r+0.0)
	r_arr = np.zeros(num_r)
	mass_arr = np.zeros(num_r)
	sigma_arr = np.zeros(num_r)
	for i in range(num_r):
	        logr = np.log10(CP['r_min']) + logr_step_size*i
	        r = 10.0**logr
	        r_arr[i] = r
	        sigma_arr[i] = np.sqrt(romberg(f,np.log(CP['k_min']),np.log(CP['kmax_linear']),args=(r,pk_interp)))
	        m = 4./3.*np.pi*CP['rho_m']*(r**(3.0))
	        mass_arr[i] = m
	return sigma_arr


# Normalized Gaussian function
def Gaus(ave, S):
        gd = numpy.exp(-0.5*ave**2.0/abs(S))
        gd = gd/numpy.sqrt(2.0*numpy.pi*abs(S))
        return gd

# ST formalism -- input sigma and delta_c array return mass function and bias
def STfucplbias(sig_arr, detc_arr):
        nu_arr = detc_arr/sig_arr
        nu2_arr = nu_arr**2.0
	pb, ab, Amp = 0.3, 0.75, 0.322
        mass_lnarr = np.log(mass_arr)
        nu_lnarr = np.log(nu_arr)
        dlsdlm_arr = np.zeros(CP['num_r'])
        dlsdlm = np.diff(nu_lnarr)/np.diff(mass_lnarr)
        dlsdlm_arr[0:CP['num_r']-1] = dlsdlm
        dlsdlm_arr[CP['num_r']-1] = dlsdlm[CP['num_r']-2]

	dist_arr = Amp*np.sqrt(2.0*ab/np.pi)*nu_arr
	dist_arr = dist_arr * (1.0+(ab*nu2_arr)**(-pb))
	dist_arr = dist_arr * np.exp(-ab*nu2_arr/2.0)
        dndlnm_arr = CP['rho_m']/mass_arr*dist_arr*abs(dlsdlm_arr)
	bias_arr = 1.0 + (ab*nu2_arr - 1.0)/detc_arr
	bias_arr = bias_arr + 2.0*pb/detc_arr/(1.0 + (ab*nu2_arr)**pb)

	return (dndlnm_arr, bias_arr)

# EPS formalism 
def EPSfucplbias(sig_arr, detc_arr):
	nu_arr = detc_arr/sig_arr
	nu2_arr = nu_arr**2.0
	mass_lnarr = np.log(mass_arr)
        nu_lnarr = np.log(nu_arr)
        dlsdlm_arr = np.zeros(CP['num_r'])
        dlsdlm = np.diff(nu_lnarr)/np.diff(mass_lnarr)
        dlsdlm_arr[0:CP['num_r']-1] = dlsdlm
        dlsdlm_arr[CP['num_r']-1] = dlsdlm[CP['num_r']-2]
        dSdm_arr = 2.0*dlsdlm_arr/mass_arr*sig_arr**2.0
	
	dist_arr = np.sqrt(2.0/np.pi)*nu_arr
	dist_arr = dist_arr * np.exp(-nu2_arr/2.0)
        dndlnm_arr = CP['rho_m']*dist_arr*abs(dSdm_arr)
	bias_arr = 1.0 + (nu2_arr - 1.0)/detc_arr
	return (dndlnm_arr, bias_arr)

# Random Walk Formalism
def ZHfunc(detc_arr, sig_arr, mss_arr, num_rr, rho_m):
	nu_arr = detc_arr/sig_arr
	nu2_arr = nu_arr**2.0
	Aa, alpha, beta = 0.75, 0.615, 0.485
	mass_lnarr = numpy.log(mss_arr)
	nu_lnarr = numpy.log(nu_arr)
	dlsdlm_arr = numpy.zeros(num_rr)
	dlsdlm = numpy.diff(nu_lnarr)/numpy.diff(mass_lnarr)
	dlsdlm_arr[0:num_rr-1] = dlsdlm
	dlsdlm_arr[num_rr-1] = dlsdlm[num_rr-2]

	sigm2_array = sig_arr**2.0
	sig2new_min = min(sigm2_array) + 0.00001
	sig2new_max = max(sigm2_array) - 0.00001
	step = 0.02
	sigm2_new = numpy.arange(sig2new_min, sig2new_max, step)
	num_sig2new = len(sigm2_new)
	detcsig2_interp = scipy.interpolate.interp1d(sigm2_array, detc_arr)
	detcnew = detcsig2_interp(sigm2_new)
	nu2new_array = detcnew**2.0/sigm2_new

# Our new barrier with considering ellipsoidal collapse effect
        bar_array = numpy.sqrt(Aa)*detcnew
        bar_array = bar_array*(1.0 + beta*(Aa*detcnew**2.0/sigm2_new)**(-alpha))
#------------------------------------------------------------#
# g1 function
        dbards_array = numpy.zeros(num_sig2new)
        dbds = numpy.diff(bar_array)/numpy.diff(sigm2_new)
        dbards_array[0:num_sig2new-1] = dbds
        dbards_array[num_sig2new-1] = dbds[num_sig2new-2]

        gg1_array = bar_array/sigm2_new - 2.0*dbards_array
        gg1_array = gg1_array*Gaus(bar_array,sigm2_new)
# g2 function  g2[i,j] = g_2(S_i, S_j - Delta S/2)
        sigm2dta_new = numpy.zeros(num_sig2new)
        for i in range(1,num_sig2new):
                sigm2dta_new[i] = sigm2_new[i] - step/2.0

        sigm2dta_new[0] = sigm2_new[0] + step/10.0
        barsigm2_interp = scipy.interpolate.interp1d(sigm2_new,bar_array)
        bardta_array = barsigm2_interp(sigm2dta_new)
        gg2_array = numpy.zeros([num_sig2new, num_sig2new])
# (B(S')-B(S))/(S'-S) Array
        barosig2_array = numpy.zeros([num_sig2new, num_sig2new])
        for i in range(num_sig2new):
                for j in range(num_sig2new):
                        barosig2_array[i,j] = (bar_array[i]-bardta_array[j])/(sigm2_new[i]-sigm2dta_new[j])
                        gg2_array[i,j] = 2.0*dbards_array[i] - barosig2_array[i, j]
                        gg2_array[i,j] = gg2_array[i,j]*Gaus(bar_array[i]-bardta_array[j],sigm2_new[i]-sigm2dta_new[j])

# delta array  delta_{i,j} = Delta S/2*g_2(S_i, S_j - Delta S/2)  
        delta_array = step/2.0*gg2_array
        ffunc = numpy.zeros(num_sig2new)
        integ_array = numpy.zeros(num_sig2new)
        ffunc[0] = 0.0
        ffunc[1] = gg1_array[1]*(1-delta_array[1, 1])**(-1.0)
# f(S) function: F = (I - M)^(-1)*G 
        for i in range(2, num_sig2new):
                integ_array[i] = sum(ffunc[j]*(delta_array[i,j]+delta_array[i,j+1]) for j in range(1, i))
                ffunc[i] = (1.0-delta_array[i, i])**(-1.0)
                ffunc[i] = ffunc[i]*(gg1_array[i] + integ_array[i])
	dSdm_array = 2.0*dlsdlm_arr/mss_arr*sigm2_array
	dSdm_interp = scipy.interpolate.interp1d(sigm2_array, dSdm_array)
	dSdmZH_array = dSdm_interp(sigm2_new)
        dndlnm_temp = rho_m*ffunc*abs(dSdmZH_array)
	dndlnm_iterp = scipy.interpolate.interp1d(sigm2_new, dndlnm_temp)
	dndlnm = numpy.zeros(num_rr) + 0.00001    # ensure Log (dndlnm) not equal to infinity
	dndlnm[0] = dndlnm_temp[num_rr-1]
	dndlnm[1:num_rr-1] = dndlnm_iterp(sigm2_array[1:num_rr-1])
	dndlnm[num_rr-1] = dndlnm_temp[0]

        return dndlnm

# Numerical Halo Bias
def ZHfucplbias(sig_arr, detc_arr):
	deriv_step = 0.0001
        step_arr = deriv_step * np.ones(len(detc_arr))
# Liu
	deriv_cof = np.array([detc_arr+2.0*step_arr, detc_arr+step_arr, detc_arr-step_arr, detc_arr-2.0*step_arr, detc_arr])
	ppservers = ()
	ncpus = 8
	ppservers = ("10.0.0.1",)
	job_server = pp.Server(ncpus, ppservers=ppservers)
	start_time = time.time()
	# The following submits 8 jobs and then retrieves the results
	inputs = deriv_cof
	deriv_temp = np.zeros([len(detc_arr), 5])
	jobs = [(input, job_server.submit(ZHfunc,(input, sig_arr, mass_arr, CP['num_r'], CP['rho_m']), (Gaus,), ("numpy","scipy.interpolate",))) for input in inputs]
	i = 0
	for input, job in jobs:
	        #print "Derivative array", "is", job()
	        job()
	        deriv_temp[:, i] = job()
	        i = i + 1
	print "Time elapsed: ", time.time() - start_time, "s"
# Liu
	deriv_nuFu = - np.log(deriv_temp[:, 0]) + 8.0*np.log(deriv_temp[:, 1])
	deriv_nuFu = deriv_nuFu - 8.0*np.log(deriv_temp[:, 2]) + np.log(deriv_temp[:, 3])
        deriv_nuFu = deriv_nuFu / (12.0*deriv_step)
	Bias = 1.0 - deriv_nuFu
	dndlnm = deriv_temp[:, 4]
	for i in range(len(Bias)):
	       if(np.isnan(Bias[i])):
        	        Bias[i] = 0

        return (dndlnm, Bias)

# rho_s and r_s
def rhors(sig_arr, detc_arr, detvir_arr, mss_arr, rho_m, len_r):
	sigdet = numpy.zeros(len_r)
	Masstar = numpy.zeros(len_r)
	for i in range(len_r):
		for j in range(len_r):
        	        sigdet[j] = abs(sig_arr[j] - detc_arr[i])
			Minumpos = numpy.where(sigdet == numpy.min(sigdet))[0][0]
#       print np.min(sigdet)
	        Masstar[i] = mss_arr[Minumpos]
#	print "Mass_*: ", Masstar
# Fitting form of Concertration parameter(c_v = r_v / r_s)
	cv_arr = 9.0*(mss_arr/Masstar)**(-0.13)
# r_s parameter in NFW density profile, do not forget virialized overdensity parameter!
	rZH_vir = (3.0*mss_arr/4.0/numpy.pi/rho_m/(detvir_arr+1.0))**(0.33333)
	rs_arr = rZH_vir / cv_arr
# rho_s parameter in NFW density profile see arXiv:1312.1292
	rhos_arr = 0.33333*rho_m*(detvir_arr+1.0)*cv_arr**3.0
	rhos_arr = rhos_arr*(numpy.log(1.0+cv_arr) - cv_arr/(cv_arr+1.0))**(-1.0)

	return (rs_arr, rhos_arr, cv_arr)

# Furier transformation of NFW density function
def NFW_func(kk_var, sig_arr, detc_arr, detvir_arr, mss_arr, rho_m, len_r):
        # i index for k-space, j index for halo mass M.
        rstmp_arr, rhostmp_arr, cvtmp_arr = rhors(sig_arr, detc_arr, detvir_arr, mss_arr, rho_m, len_r)
	mstmp_arr = mss_arr
        rhoNFW = numpy.zeros(len_r)
        rhotmp = numpy.zeros(len_r)
        krs = numpy.zeros(len_r)
        for j in range(len_r):
        # NFW density profile   
                rhoNFW[j] = 4.0*numpy.pi*rhostmp_arr[j]*rstmp_arr[j]**3.0/mstmp_arr[j]
                krs[j] = kk_var*rstmp_arr[j]
                rhotmp[j] = numpy.sin(krs[j])*(sympy.Si((1.0+cvtmp_arr[j])*krs[j])-sympy.Si(krs[j]))
                rhotmp[j] = rhotmp[j] - numpy.sin(cvtmp_arr[j]*krs[j])/krs[j]/(1.0 + cvtmp_arr[j])
                rhotmp[j] = rhotmp[j] + numpy.cos(krs[j])*(sympy.Ci((1.0+cvtmp_arr[j])*krs[j])-sympy.Ci(krs[j]))
                rhoNFW[j] = rhoNFW[j] * rhotmp[j]
        return rhoNFW


def pf(logm, nlnm, ynfw_interp):
        m = np.exp(logm)
        pf = nlnm(m) * m**2.0/CP['rho_m']**2.0
        pf = pf * ynfw_interp(m)**2.0
        return pf

#I term 
def Itm(logm, nlnm, Bb, ynfw_interp):
        m = np.exp(logm)
        Itm = nlnm(m) * m/CP['rho_m']
        Itm = Itm * ynfw_interp(m) * Bb(m)
        return Itm

#-------------------------------------------------------------------------------------#
# Gravity Model
#-------------------------------------#
# Hu - Sawicki   ------- sigma, delta_c, delta_vir

HS = {}
HS['dtac_file'] = CP['delta_sc']    #pre-calcuated delta_c file
HS['pok_file'] = CP['linear_mpk']  #linear Hu-Sawicki p(k) from EFTCAMB
HS['dtac_dat'] = np.loadtxt(HS['dtac_file'])

GeoG_data = HS['dtac_dat'][:, 2]
ata_data = HS['dtac_dat'][:, 7]
Delta_data = HS['dtac_dat'][:, 8]
deti_data = HS['dtac_dat'][:, 3]

ai = 1.e-3  #initial time set for spherical collapse
Rta_array = ata_data/ai*((1.0 + deti_data)/(1.0 + Delta_data))**0.33333   # Maximum Radius
eta = 2.0*(1.0 - CP['omg_m'])/GeoG_data/CP['omg_m']/ata_data**(-3.0)/(1.0 + Delta_data)  # eta parameter

#print eta
Ss = np.zeros(len(eta))

for i in range(len(eta)):
        x = Symbol('x')
        fx = solve(eta[i]*(2.0*x**3.0 - x) + 1.0 - 2.0*x, x)[1]
        Ss[i] = re(fx)     # Ss = r_vir/r_max
#        print Ss[i]
Rvir_array = Ss*Rta_array
Delvir_array = (1.0 + Delta_data)*(Ss*ata_data)**(-3.0) - 1.0  # \Delta_vir
hsdetvir_iterp = scipy.interpolate.interp1d(HS['dtac_dat'][:,1],  Delvir_array)

HS['sigma'] = sigm(HS['pok_file'])
hsdetc_iterp = scipy.interpolate.interp1d(HS['dtac_dat'][:,1],  HS['dtac_dat'][:,6])
HS['dtac_arr'] = hsdetc_iterp(r_arr)
HS['dtavir_arr'] = hsdetvir_iterp(r_arr)

#-------------------------------------#
'''
# LCDM ----- 

LCDM = {}
LCDM['pok_file'] = CP['lcdm_linear_mpk']  #linear LCDM p(k) from EFTCAMB
LCDM['sigma'] = sigm(LCDM['pok_file'])

lcdm_detc, lcdm_detvir = 1.67567, 392.704
LCDM['dtac_arr'] = lcdm_detc * r_arr**0.0
LCDM['dtavir_arr'] = lcdm_detvir * r_arr**0.0
'''
#-------------------------------------------------------------------------------------#
# Halo Model  ---  Mass function, related bias, NFW density file, Pok
class Halo(object):

	def __init__(self, name, flag):
		print "initializing...."
		self.name = name
		self.flag = flag
		self.deltac = self.name['dtac_arr']
		self.sigma = self.name['sigma']
		self.ovvir = self.name['dtavir_arr']
		
	def msfcbias(self):
		print "Calculating mass function and linear bias, please wait ... ..."
		nlnmbias = {'EPS': EPSfucplbias, 'ST': STfucplbias, 'ZH': ZHfucplbias}
		self.ndlnm, self.bias = eval("nlnmbias[self.flag](self.sigma, self.deltac)")
		mfcbias_arr = np.zeros([CP['num_r'], 3])
		mfcbias_arr[:, 0] = mass_arr
		mfcbias_arr[:, 1] = self.ndlnm
		mfcbias_arr[:, 2] = self.bias
		return mfcbias_arr
	
	def NFWpro(self):
		logk_array = np.linspace(np.log10(CP['k_min']), np.log10(CP['kmax_halo']), CP['num_k'])
		k_array = 10.0**logk_array
		self.karr = k_array

		ppservers = ()
		ncpus = 8
		ppservers = ("10.0.0.1",)
		job_server = pp.Server(ncpus, ppservers=ppservers)
		start_time = time.time()
		# The following submits 8 jobs and then retrieves the results
		inputs = k_array
		rhoNFW_arr = np.zeros([CP['num_k'], CP['num_r']])
		jobs = [(input, job_server.submit(NFW_func,(input, self.sigma, self.deltac, self.ovvir, mass_arr, CP['rho_m'], CP['num_r']), (rhors,), ("numpy","sympy",))) for input in inputs]
		i = 0
		for input, job in jobs:
		        #print "NFW array", input, "is", job()
		        job()
		        rhoNFW_arr[i, :] = job()
		        i = i + 1
		print "Time elapsed: ", time.time() - start_time, "s"
		job_server.print_stats()

                self.NFW = rhoNFW_arr

	def pok(self):

		self.msfcbias()
		self.NFWpro()

		bias_interp = scipy.interpolate.interp1d(mass_arr, self.bias)
		nlnm_interp = scipy.interpolate.interp1d(mass_arr, self.ndlnm)
		pk_dat = np.loadtxt(self.name['pok_file'])
		pk_interp = scipy.interpolate.interp1d(pk_dat[:, 0], pk_dat[:, 1])
		plk_arr = pk_interp(self.karr)     
		
		p1h_arr = np.zeros(CP['num_k'])
		pok_arr = np.zeros(CP['num_k'])
		Ii_arr = np.zeros(CP['num_k'])
		m_min = 1.000001 * min(mass_arr)
		m_max = 0.999999 * max(mass_arr)

		for i in range(CP['num_k']):
		        rho_interp = scipy.interpolate.interp1d(mass_arr, self.NFW[i, :])
		        p1h_arr[i] = romberg(pf,np.log(m_min),np.log(m_max),args=(nlnm_interp,rho_interp))
		        Ii_arr[i] = romberg(Itm,np.log(m_min),np.log(m_max),args=(nlnm_interp,bias_interp,rho_interp))
		        pok_arr[i] = plk_arr[i]*Ii_arr[i]**2.0 + p1h_arr[i]

		pkk_arr = np.zeros([CP['num_k'], 5])
		pkk_arr[:, 0] = self.karr
		pkk_arr[:, 1] = p1h_arr
		pkk_arr[:, 2] = Ii_arr
		pkk_arr[:, 3] = plk_arr
		pkk_arr[:, 4] = plk_arr * (Ii_arr/max(Ii_arr))**2.0 + p1h_arr

		return pkk_arr

#-----------------------------------------------------------------------------------------------#
#  Halo(A, B) ------------ A = {LCDM, HS},  B = {EPS, ST, ZH}

start_time = time.time()

HS_ZHHalo = Halo(HS, "ZH")
Halos = {"HS_ZH":HS_ZHHalo}

for Halo in Halos.keys():
	pok_tmp = Halos[Halo].pok()
	file_tmp = CP['nonlinear_mpk']
	if os.path.exists(file_tmp): os.remove(file_tmp)
	np.savetxt(file_tmp, pok_tmp)

print "Time elapsed: ", time.time() - start_time, "s"

exit()
