# created by Xue-Wen LIU, Bin HU @10.23 2017
# This code solves non-linear power spectrum using Halo model with different distribution function including Press-Sheth, Sheth-Tormen, Random Walk Formalism(arXiv:0508384)

from __future__ import print_function


import math
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import yaml
import scipy.interpolate
import scipy.integrate

from NFWfunc_cython import NFWfunc_loop
from ZHfunc_cython import gg2_func_cython, fS_func_cython
start_time = time.time()


# Load some external config file
with open('./CHAM_params.yaml') as fp:
    CP = yaml.load(fp, Loader=yaml.FullLoader)

CP['rho_m'] = CP['omg_m'] * CP['rho_c'] / CP['hub']**3.0

delta_sc_nondyn = CP['delta_sc_nondyn']
z = CP['redshift']
a = 1.0 / (1+z)
print ('delta_sc_nondyn =', delta_sc_nondyn)
print ('redshift =', z)

# ignore warnings
warnings.filterwarnings("ignore")

#--------------------------------------------------------------------------------------#
# functions 

# compute \sigma  --just need to input the linear p(k) filename from EFTCAMB
def sigm(pok_file):
    global r_arr, mass_arr
    num_r = CP['num_r']
    pok_dat = np.loadtxt(pok_file)
    lnpk_interp = scipy.interpolate.interp1d(np.log(pok_dat[:,0]), np.log(pok_dat[:,1]))   # linear P(k)

    log10_r_step_size = (np.log10(CP['r_max'])-np.log10(CP['r_min'])) /(num_r+0.0)
    log10_r_arr = np.linspace(np.log10(CP['r_min']), np.log10(CP['r_max'])+log10_r_step_size, num=num_r)
    r_arr = 10.0**(log10_r_arr)
    mass_arr = 4.0/3.0*np.pi*CP['rho_m']*(r_arr**(3.0))

    ln_k_intarr = np.linspace(np.log(CP['k_min']), np.log(CP['kmax_linear']), num=10000)
    k_intarr = np.exp(ln_k_intarr)

    tmp_intarr = (k_intarr**(3.0)) * np.exp(lnpk_interp(ln_k_intarr)) / 2.0/np.pi**2

    sigma_arr = np.zeros_like(r_arr)
    for i in range(num_r):
        x = k_intarr * r_arr[i]
        win_intarr = 3.0/(x**(3.0)) * (np.sin(x)-x*np.cos(x))
        f_intarr = tmp_intarr * win_intarr**2

        sigma_arr[i] = np.sqrt(scipy.integrate.trapz(f_intarr, ln_k_intarr))
    return sigma_arr


# Normalized Gaussian function
def Gaus(ave, S):
    gd = np.exp(-0.5*ave**2.0/abs(S))
    gd = gd/np.sqrt(2.0*np.pi*abs(S))
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


def ZHfunc_nondyn(sig_arr, mss_arr, num_rr, rho_m):
    nu_arr = delta_sc_nondyn / sig_arr
    Aa, alpha, beta = 0.75, 0.615, 0.485

    mass_lnarr = np.log(mss_arr)
    nu_lnarr = np.log(nu_arr)
    dlsdlm_arr = np.zeros(num_rr)
    dlsdlm = np.diff(nu_lnarr)/np.diff(mass_lnarr)
    dlsdlm_arr[0:num_rr-1] = dlsdlm
    dlsdlm_arr[num_rr-1] = dlsdlm[num_rr-2]

    sigm2_array = sig_arr**2.0
    sig2new_min = np.min(sigm2_array)
    sig2new_max = np.max(sigm2_array)

    step = 0.01
    sigm2_new = np.linspace(sig2new_min, sig2new_max, int((sig2new_max-sig2new_min)/step))
    num_sig2new = len(sigm2_new)


# Our new barrier with considering ellipsoidal collapse effect
    bar_array = np.sqrt(Aa)*delta_sc_nondyn
    bar_array = bar_array*(1.0 + beta*(Aa*delta_sc_nondyn**2.0/sigm2_new)**(-alpha))
#------------------------------------------------------------#
# g1 function
    dbards_array = np.zeros(num_sig2new)
    dbds = np.diff(bar_array)/np.diff(sigm2_new)
    dbards_array[0:num_sig2new-1] = dbds
    dbards_array[num_sig2new-1] = dbds[num_sig2new-2]

    gg1_array = bar_array/sigm2_new - 2.0*dbards_array
    gg1_array = gg1_array*Gaus(bar_array,sigm2_new)
# g2 function  g2[i,j] = g_2(S_i, S_j - Delta S/2)
    sigm2dta_new = np.zeros(num_sig2new)
    sigm2dta_new = sigm2_new - step/2.0
    sigm2dta_new[0] = sigm2_new[0] + step/10.0
    #barsigm2_interp = scipy.interpolate.interp1d(sigm2_new,bar_array)
    bardta_array = np.sqrt(Aa) * delta_sc_nondyn * (1.0 + beta*(Aa*delta_sc_nondyn**2.0/sigm2dta_new)**(-alpha))
    gg2_array = np.zeros([num_sig2new, num_sig2new])
# (B(S')-B(S))/(S'-S) Array
    barosig2_array = np.zeros([num_sig2new, num_sig2new])
    gg2_func_cython(gg2_array, barosig2_array, num_sig2new, bar_array, bardta_array, sigm2_new, sigm2dta_new, dbards_array)

# delta array  delta_{i,j} = Delta S/2*g_2(S_i, S_j - Delta S/2)  
    delta_array = step/2.0*gg2_array
    ffunc = np.zeros(num_sig2new)
    integ_array = np.zeros(num_sig2new)
    ffunc[0] = 0.0
    ffunc[1] = gg1_array[1]*(1-delta_array[1, 1])**(-1.0)
# f(S) function: F = (I - M)^(-1)*G 
    fS_func_cython(ffunc, num_sig2new, integ_array, delta_array, gg1_array)
    dSdm_array = 2.0*dlsdlm_arr/mss_arr*sigm2_array

    dSdm_interp = scipy.interpolate.interp1d(sigm2_array, dSdm_array)
    dSdmZH_array = dSdm_interp(sigm2_new)
    dndlnm_temp = rho_m*ffunc*np.abs(dSdmZH_array)
    dndlnm_iterp = scipy.interpolate.interp1d(sigm2_new, dndlnm_temp)
    dndlnm = np.zeros(num_rr) + 0.00001    # ensure Log (dndlnm) not equal to infinity
    dndlnm[0] = dndlnm_temp[num_rr-1]

    dndlnm[1:num_rr-1] = dndlnm_iterp(sigm2_array[1:num_rr-1])
    dndlnm[num_rr-1] = dndlnm_temp[0]

    return dndlnm



def ZHfucplbias_nondyn(sig_arr, mss_arr, num_rr, rho_m):
    nu_arr = delta_sc_nondyn/sig_arr
    nu2_arr = nu_arr**2.0
    pb, ab, Amp = 0.3, 0.75, 0.322

    dndlnm_arr = ZHfunc_nondyn(sig_arr, mss_arr, num_rr, rho_m)
    bias_arr = 1.0 + (ab*nu2_arr - 1.0)/delta_sc_nondyn
    bias_arr = bias_arr + 2.0*pb/delta_sc_nondyn/(1.0 + (ab*nu2_arr)**pb)

    return (dndlnm_arr, bias_arr)


#######################################




#rho_s and r_s
def rhors(sig_arr, mss_arr, rho_m, len_r):
    
    diff = np.abs(sig_arr - delta_sc_nondyn)
    min_index = np.where(diff == np.min(diff))[0][0]
    Masstar = mss_arr[min_index]
    
    # Fitting form of Concertration parameter(c_v = r_v / r_s)
    cv_arr = 9.0*a*(mss_arr/Masstar)**(-0.13)

    # r_s parameter in NFW density profile, do not forget virialized overdensity parameter!
    rZH_vir = (3.0*mss_arr/4.0/np.pi/rho_m/(HS['dtavir_arr']+1.0))**(0.33333)
    rs_arr = rZH_vir / cv_arr
    
    # rho_s parameter in NFW density profile see equation (2.32) of arXiv:1312.1292
    rhos_arr = ((1.0/3.0) * rho_m * (HS['dtavir_arr']) * cv_arr**3.0)\
        * (np.log(1.0+cv_arr) - cv_arr/(cv_arr+1.0))**(-1.0)
    

    return (rs_arr, rhos_arr, cv_arr)


#Furier transformation of NFW density function
def NFW_func(kk_var, sig_arr, detvir_arr, mss_arr, rho_m, len_r):
    # i index for k-space, j index for halo mass M.
    
    rstmp_arr, rhostmp_arr, cvtmp_arr = rhors(sig_arr, mss_arr, rho_m, len_r)
    mstmp_arr = mss_arr
    rhoNFW = np.zeros(len_r)
    rhotmp = np.zeros(len_r)
    krs = np.zeros(len_r)

    rhoNFW = np.array(NFWfunc_loop(rhoNFW, len_r, rhostmp_arr, rstmp_arr, mstmp_arr, krs, kk_var, cvtmp_arr, rhotmp))

    return rhoNFW



#-------------------------------------------------------------------------------------#
# Gravity Model
#-------------------------------------#
# Hu - Sawicki   ------- sigma, delta_c, delta_vir

HS = {}
HS['pok_file'] = CP['linear_mpk']  #linear Hu-Sawicki p(k) from EFTCAMB
HS['sigma'] = sigm(HS['pok_file'])
HS['dtac_arr'] = delta_sc_nondyn * np.ones_like(r_arr)
HS['dtavir_arr'] = 400.0 * np.ones_like(r_arr)
#-------------------------------------#



# Halo Model  ---  Mass function, related bias, NFW density file, Pok
# non-dynamical approximation
class HaloModel_nondynApprox(object):

    def __init__(self, name, flag):
        print ("initializing....")
        self.name = name
        self.flag = flag
        self.deltasc = delta_sc_nondyn
        self.sigma = self.name['sigma']
        self.ovvir = self.name['dtavir_arr']

        
    
    def msfcbias(self):
        print ("Calculating mass function and linear bias, please wait ... ...")
        mfcbias_arr = np.zeros([CP['num_r'], 3])
        mfcbias_arr[:, 0] = mass_arr

        nlnmbias = {'ZH': ZHfucplbias_nondyn}
        self.ndlnm, self.bias = nlnmbias[self.flag](self.sigma, mass_arr, CP['num_r'], CP['rho_m'])
        mfcbias_arr[:, 1] = self.ndlnm
        mfcbias_arr[:, 2] = self.bias
        return mfcbias_arr        
    

    def NFWpro(self):
        logk_array = np.linspace(np.log10(CP['k_min']), np.log10(CP['kmax_halo']), CP['num_k'])
        k_array = 10.0**logk_array
        self.karr = k_array
        rhoNFW_arr = np.zeros([CP['num_k'], CP['num_r']])
        
        for i in range(len(k_array)):
            rhoNFW_arr[i, :] = NFW_func(k_array[i], self.sigma, self.ovvir, mass_arr, CP['rho_m'], CP['num_r'])

        self.NFW = rhoNFW_arr


    def pok(self):

        start_time_nb = time.time()
        self.msfcbias()
        print ("Time elapsed (mass function and bias): ", np.round(time.time() - start_time_nb, 4), "s")

        start_time_NFW = time.time()
        self.NFWpro()
        print ("Time elapsed (NFW Fourier Transformation): ", np.round(time.time() - start_time_NFW, 4), "s")


        start_time_Pok = time.time()

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
            rhoNFWfft_interp = scipy.interpolate.interp1d(mass_arr, self.NFW[i, :])
            ln_mass_intarr = np.linspace(np.log(m_min), np.log(m_max), num=1000)
            mass_intarr = np.exp(ln_mass_intarr)
            rhoNFWfft_intarr = rhoNFWfft_interp(mass_intarr)
            nlnm_intarr = nlnm_interp(mass_intarr)
            pf_intarr = nlnm_intarr * mass_intarr**2 / CP['rho_m']**2 * rhoNFWfft_intarr**2.0
            p1h_arr[i] = scipy.integrate.trapz(pf_intarr, ln_mass_intarr)

            bias_intarr = bias_interp(mass_intarr)
            Ii_intarr = nlnm_intarr * mass_intarr / CP['rho_m'] * rhoNFWfft_intarr * bias_intarr
            Ii_arr[i] = scipy.integrate.trapz(Ii_intarr, ln_mass_intarr)

            pok_arr[i] = plk_arr[i]*Ii_arr[i]**2.0 + p1h_arr[i]

        pkk_arr = np.zeros([CP['num_k'], 5])
        pkk_arr[:, 0] = self.karr
        pkk_arr[:, 1] = p1h_arr
        pkk_arr[:, 2] = Ii_arr
        pkk_arr[:, 3] = plk_arr
        pkk_arr[:, 4] = plk_arr * (Ii_arr/max(Ii_arr))**2.0 + p1h_arr

        print ("Time elapsed (non-linear P[k] halo model integration): ", np.round(time.time() - start_time_Pok, 4), "s")

        return pkk_arr


# HaloModel_nondynApprox(name, flag)
pok_tmp = HaloModel_nondynApprox(HS, "ZH").pok()
file_tmp = CP['nonlinear_mpk']
if os.path.exists(file_tmp): 
    os.remove(file_tmp)
np.savetxt(file_tmp, pok_tmp)


print ("output nonlinear P(k) file: ", file_tmp)
print ("Time elapsed (total): ", np.round(time.time() - start_time, 4), "s")
