# created by Xue-Wen LIU, Bin HU @10.23 2017
# This code solves non-linear power spectrum using Halo model with different distribution function including Press-Sheth, Sheth-Tormen, Random Walk Formalism(arXiv:0508384)


import math
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pp
import yaml
import scipy.interpolate
import scipy.integrate
from sympy import *

import NFWfunc_cython
start_time = time.time()

delta_sc_nondyn = 1.692
# Load some external config file
with open('./CHAM_params.yaml') as fp:
    CP = yaml.load(fp, Loader=yaml.UnsafeLoader)

CP['rho_m'] = CP['omg_m'] * CP['rho_c'] / CP['hub']**3.0

# ignore warnings
warnings.filterwarnings("ignore")

#--------------------------------------------------------------------------------------#
# functions 

# compute \sigma  --just need to input the linear p(k) filename from EFCAMB
def sigm(pok_file):
    global r_arr, mass_arr
    num_r = CP['num_r']
    pok_dat = np.loadtxt(pok_file)
    pk_interp = scipy.interpolate.interp1d(pok_dat[:,0], pok_dat[:,1])   # linear P(k)
    log10_r_step_size = (np.log10(CP['r_max'])-np.log10(CP['r_min'])) /(num_r+0.0)
    log10_r_arr = np.linspace(np.log10(CP['r_min']), np.log10(CP['r_max'])+log10_r_step_size, num=num_r)
    r_arr = 10.0**(log10_r_arr)
    mass_arr = 4.0/3.0*np.pi*CP['rho_m']*(r_arr**(3.0))

    ln_k_intarr = np.linspace(np.log(CP['k_min']), np.log(CP['kmax_linear']), num=10000)
    k_intarr = np.exp(ln_k_intarr)

    tmp_intarr = (k_intarr**(3.0))*pk_interp(k_intarr)/2.0/np.pi**2

    sigma_arr = np.zeros_like(r_arr)
    for i in range(num_r):
        x = k_intarr * r_arr[i]
        win_intarr = 3.0/(x**(3.0)) * (np.sin(x)-x*np.cos(x))
        f_intarr = tmp_intarr * win_intarr**2

        sigma_arr[i] = np.sqrt(scipy.integrate.trapz(f_intarr, ln_k_intarr))
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


def ZHfunc_nondyn(sig_arr, mss_arr, num_rr, rho_m):
    from ZHfunc_cython import gg2_func_cython, fS_func_cython
    
    #delta_sc_nondyn = 1.692
    nu_arr = delta_sc_nondyn / sig_arr
    nu2_arr = nu_arr**2.0
    Aa, alpha, beta = 0.75, 0.615, 0.485
    mass_lnarr = numpy.log(mss_arr)
    nu_lnarr = numpy.log(nu_arr)
    dlsdlm_arr = numpy.zeros(num_rr)
    dlsdlm = numpy.diff(nu_lnarr)/numpy.diff(mass_lnarr)
    dlsdlm_arr[0:num_rr-1] = dlsdlm
    dlsdlm_arr[num_rr-1] = dlsdlm[num_rr-2]

    sigm2_array = sig_arr**2.0
    # sig2new_min = numpy.min(sigm2_array) - 0.00001
    # sig2new_max = numpy.max(sigm2_array) + 0.00001
    sig2new_min = numpy.min(sigm2_array)
    sig2new_max = numpy.max(sigm2_array)

    step = 0.01
    #sigm2_new = numpy.arange(sig2new_min, sig2new_max+step, step)
    sigm2_new = numpy.linspace(sig2new_min, sig2new_max, int((sig2new_max-sig2new_min)/step))
    num_sig2new = len(sigm2_new)
    #nu2new_array = delta_sc_nondyn**2.0/sigm2_new

# Our new barrier with considering ellipsoidal collapse effect
    bar_array = numpy.sqrt(Aa)*delta_sc_nondyn
    bar_array = bar_array*(1.0 + beta*(Aa*delta_sc_nondyn**2.0/sigm2_new)**(-alpha))
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
    sigm2dta_new = sigm2_new - step/2.0
    sigm2dta_new[0] = sigm2_new[0] + step/10.0
    #barsigm2_interp = scipy.interpolate.interp1d(sigm2_new,bar_array)
    bardta_array = numpy.sqrt(Aa) * delta_sc_nondyn * (1.0 + beta*(Aa*delta_sc_nondyn**2.0/sigm2dta_new)**(-alpha))
    gg2_array = numpy.zeros([num_sig2new, num_sig2new])
# (B(S')-B(S))/(S'-S) Array
    barosig2_array = numpy.zeros([num_sig2new, num_sig2new])
    gg2_func_cython(gg2_array, barosig2_array, num_sig2new, bar_array, bardta_array, sigm2_new, sigm2dta_new, dbards_array)

# delta array  delta_{i,j} = Delta S/2*g_2(S_i, S_j - Delta S/2)  
    delta_array = step/2.0*gg2_array
    ffunc = numpy.zeros(num_sig2new)
    integ_array = numpy.zeros(num_sig2new)
    ffunc[0] = 0.0
    ffunc[1] = gg1_array[1]*(1-delta_array[1, 1])**(-1.0)
# f(S) function: F = (I - M)^(-1)*G 
    fS_func_cython(ffunc, num_sig2new, integ_array, delta_array, gg1_array)
    dSdm_array = 2.0*dlsdlm_arr/mss_arr*sigm2_array



    

    dSdm_interp = scipy.interpolate.interp1d(sigm2_array, dSdm_array)
    dSdmZH_array = dSdm_interp(sigm2_new)
    dndlnm_temp = rho_m*ffunc*numpy.abs(dSdmZH_array)
    dndlnm_iterp = scipy.interpolate.interp1d(sigm2_new, dndlnm_temp)
    dndlnm = numpy.zeros(num_rr) + 0.00001    # ensure Log (dndlnm) not equal to infinity
    dndlnm[0] = dndlnm_temp[num_rr-1]

    dndlnm[1:num_rr-1] = dndlnm_iterp(sigm2_array[1:num_rr-1])
    dndlnm[num_rr-1] = dndlnm_temp[0]

    return dndlnm



def ZHfucplbias_nondyn(sig_arr, mss_arr, num_rr, rho_m):
    nu_arr = delta_sc_nondyn/sig_arr
    nu2_arr = nu_arr**2.0
    pb, ab, Amp = 0.3, 0.75, 0.322

    # start_time_1 = time.time()
    # print 'first upcross distribution calculation, hajimarimasu!'
    dndlnm_arr = ZHfunc_nondyn(sig_arr, mss_arr, num_rr, rho_m)
    bias_arr = 1.0 + (ab*nu2_arr - 1.0)/delta_sc_nondyn
    bias_arr = bias_arr + 2.0*pb/delta_sc_nondyn/(1.0 + (ab*nu2_arr)**pb)
    # print "first upcross distribution block time elapsed: ", time.time() - start_time_1, "s"

    return (dndlnm_arr, bias_arr)


#######################################



#rho_s and r_s
def rhors(sig_arr, detc_arr, detvir_arr, mss_arr, rho_m, len_r):
    
    delta_sc_nondyn = 1.684
    sigdet = numpy.zeros(len_r)
    Masstar = numpy.zeros(len_r)
    arr_tmp = numpy.abs(sig_arr - delta_sc_nondyn)
    Minumpos = numpy.where(arr_tmp == numpy.min(arr_tmp))[0][0]
    Masstar = mss_arr[Minumpos]
    
    # Fitting form of Concertration parameter(c_v = r_v / r_s)
    cv_arr = 9.0*(mss_arr/Masstar)**(-0.13)

    # r_s parameter in NFW density profile, do not forget virialized overdensity parameter!
    rZH_vir = (3.0*mss_arr/4.0/numpy.pi/rho_m/(detvir_arr+1.0))**(0.33333)
    rs_arr = rZH_vir / cv_arr
    
    # rho_s parameter in NFW density profile see arXiv:1312.1292
    rhos_arr = 0.33333*rho_m*(detvir_arr+1.0)*cv_arr**3.0
    rhos_arr = rhos_arr*(numpy.log(1.0+cv_arr) - cv_arr/(cv_arr+1.0))**(-1.0)
    

    return (rs_arr, rhos_arr, cv_arr)


#Sine and Cosine integral functions used in Fourier transformation of NFW profile
# source: https://en.wikipedia.org/wiki/Trigonometric_integral#Auxiliary_functions

# if 0 < x < 4
def Si_lessThanFour_approxFunc(x):
    x2 = x**2
    x4 = x**4
    x6 = x**6
    x8 = x**8
    x10 = x**10
    x12 = x**12
    x14 = x**14
    
    akane = x * (1.0 - 4.54393409816329991*(1e-2)*(x2) + 1.15457225751016682*(1e-3)*x4 - 1.41018536821330254*(1e-5)*x6 + 9.43280809438713025*(1e-8)*x8 - 3.53201978997168357*(1e-10)*x10 + 7.08240282274875911*(1e-13)*x12 - 6.05338212010422477*(1e-16)*x14) / (1.0 + 1.01162145739225565*(1e-2)*x2 + 4.99175116169755106*(1e-5)*x4 + 1.55654986308745614*(1e-7)*x6 + 3.28067571055789734*(1e-10)*x8 + 4.5049097575386581*(1e-13)*x10 + 3.21107051193712168*(1e-16)*x12)
    return akane

def Ci_lessThanFour_approxFunc(x):
    x2 = x**2
    x4 = x**4
    x6 = x**6
    x8 = x**8
    x10 = x**10
    x12 = x**12
    x14 = x**14
    Euler_const = 0.577215664901532861
    
    
    kaki = Euler_const + numpy.log(x) + x2*(-0.25 + 7.51851524438898291*(1e-3)*x2 - 1.27528342240267686*(1e-4)*x4 + 1.05297363846239184*(1e-6)*x6 - 4.68889508144848019*(1e-9)*x8 + 1.06480802891189243*(1e-11)*x10 - 9.93728488857585407*(1e-15)*x12)/(1.0 + 1.1592605689110735*(1e-2)*x2 + 6.72126800814254432*(1e-5)*x4 + 2.55533277086129636*(1e-7)*x6 + 6.97071295760958946*(1e-10)*x8 + 1.38536352772778619*(1e-12)*x10 + 1.89106054713059759*(1e-15)*x12 + 1.39759616731376855*(1e-18)*x14)
    return kaki

# x > 4
# Si(x) = pi/2 - f(x)*cos(x) - g(x)*sin(x)
# Ci(x) = f(x)*sin(x) - g(x)*cos(x)
def f_func(x):
    xm2 = x**(-2)
    xm4 = x**(-4)
    xm6 = x**(-6)
    xm8 = x**(-8)
    xm10 = x**(-10)
    xm12 = x**(-12)
    xm14 = x**(-14)
    xm16 = x**(-16)
    xm18 = x**(-18)
    xm20 = x**(-20)
    
#    fenzi = 1.0 + 7.44437068161936700618*(1e2)*xm2 \
#                + 1.96396372895146869801*(1e5)*xm4 \
#                + 2.37750310125431834034*(1e7)*xm6 \
#                + 1.43073403821274636888*(1e9)*xm8 \
#                + 4.33736238870432522765*(1e10)*xm10 \
#                + 6.40533830574022022911*(1e11)*xm12 \
#                + 4.20968180571076940208*(1e12)*xm14 \
#                + 1.00795182980368574617*(1e13)*xm16 \
#                + 4.94816688199951963482*(1e12)*xm18 \
#                - 4.94701168645415959931*(1e11)*xm20
#    
#    fenmu = 1.0 + 7.46437068161927678031*(1e2)*xm2 \
#                + 1.97865247031583951450*(1e5)*xm4 \
#                + 2.41535670165126845144*(1e7)*xm6 \
#                + 1.47478952192985464958*(1e9)*xm8 \
#                + 4.58595115847765779830*(1e10)*xm10 \
#                + 7.08501308149515401563*(1e11)*xm12 \
#                + 5.06084464593475076774*(1e12)*xm14 \
#                + 1.43468549171581016479*(1e13)*xm16 \
#                + 1.11535493509914254097*(1e13)*xm18
    
    fff = (1.0  + 7.44437068161936700618*(1e2)*xm2     \
                + 1.96396372895146869801*(1e5)*xm4     \
                + 2.37750310125431834034*(1e7)*xm6     \
                + 1.43073403821274636888*(1e9)*xm8     \
                + 4.33736238870432522765*(1e10)*xm10   \
                + 6.40533830574022022911*(1e11)*xm12   \
                + 4.20968180571076940208*(1e12)*xm14   \
                + 1.00795182980368574617*(1e13)*xm16   \
                + 4.94816688199951963482*(1e12)*xm18   \
                - 4.94701168645415959931*(1e11)*xm20) /\
          (1.0  + 7.46437068161927678031*(1e2)*xm2     \
                + 1.97865247031583951450*(1e5)*xm4     \
                + 2.41535670165126845144*(1e7)*xm6     \
                + 1.47478952192985464958*(1e9)*xm8     \
                + 4.58595115847765779830*(1e10)*xm10   \
                + 7.08501308149515401563*(1e11)*xm12   \
                + 5.06084464593475076774*(1e12)*xm14   \
                + 1.43468549171581016479*(1e13)*xm16   \
                + 1.11535493509914254097*(1e13)*xm18) / x
    
    return fff


def g_func(x):
    xm2 = x**(-2)
    xm4 = x**(-4)
    xm6 = x**(-6)
    xm8 = x**(-8)
    xm10 = x**(-10)
    xm12 = x**(-12)
    xm14 = x**(-14)
    xm16 = x**(-16)
    xm18 = x**(-18)
    xm20 = x**(-20)
    
#    fenzi = 1.0 + 8.13595201151686150*(1e2)*xm2 \
#                + 2.35239181626478200*(1e5)*xm4 \
#                + 3.12557570795778731*(1e7)*xm6 \
#                + 2.06297595146763354*(1e9)*xm8 \
#                + 6.83052205423625007*(1e10)*xm10 \
#                + 1.09049528450362786*(1e12)*xm12 \
#                + 7.57664583257834349*(1e12)*xm14 \
#                + 1.81004487464664575*(1e13)*xm16 \
#                + 6.43291613143049485*(1e12)*xm18 \
#                - 1.36517137670871689*(1e12)*xm20
#
#    fenmu = 1.0 + 8.19595201151451564*(1e2)*xm2 \
#                + 2.40036752835578777*(1e5)*xm4 \
#                + 3.26026661647090822*(1e7)*xm6 \
#                + 2.23355543278099360*(1e9)*xm8 \
#                + 7.87465017341829930*(1e10)*xm10 \
#                + 1.39866710696414565*(1e12)*xm12 \
#                + 1.17164723371736605*(1e13)*xm14 \
#                + 4.01839087307656620*(1e13)*xm16 \
#                + 3.99653257887490811*(1e13)*xm18
    
    ggg =  (1.0 + 8.13595201151686150*(1e2)*xm2     \
                + 2.35239181626478200*(1e5)*xm4     \
                + 3.12557570795778731*(1e7)*xm6     \
                + 2.06297595146763354*(1e9)*xm8     \
                + 6.83052205423625007*(1e10)*xm10   \
                + 1.09049528450362786*(1e12)*xm12   \
                + 7.57664583257834349*(1e12)*xm14   \
                + 1.81004487464664575*(1e13)*xm16   \
                + 6.43291613143049485*(1e12)*xm18   \
                - 1.36517137670871689*(1e12)*xm20) /\
           (1.0 + 8.19595201151451564*(1e2)*xm2     \
                + 2.40036752835578777*(1e5)*xm4     \
                + 3.26026661647090822*(1e7)*xm6     \
                + 2.23355543278099360*(1e9)*xm8     \
                + 7.87465017341829930*(1e10)*xm10   \
                + 1.39866710696414565*(1e12)*xm12   \
                + 1.17164723371736605*(1e13)*xm14   \
                + 4.01839087307656620*(1e13)*xm16   \
                + 3.99653257887490811*(1e13)*xm18) / x**2
    
    return ggg


# Si(x) = pi/2 - f(x)*cos(x) - g(x)*sin(x)
def Si_largerThanFour_approxFunc(x):
    return numpy.pi/2 - f_func(x)*numpy.cos(x) - g_func(x)*numpy.sin(x)

    
# Ci(x) = f(x)*sin(x) - g(x)*cos(x)
def Ci_largerThanFour_approxFunc(x):
    return f_func(x)*numpy.sin(x) - g_func(x)*numpy.cos(x)


########################################

#Furier transformation of NFW density function
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

        if (1.0+cvtmp_arr[j])*krs[j] < 4:
            si_apple = Si_lessThanFour_approxFunc((1.0+cvtmp_arr[j])*krs[j])
            ci_apple = Ci_lessThanFour_approxFunc((1.0+cvtmp_arr[j])*krs[j])
        else:
            si_apple = Si_largerThanFour_approxFunc((1.0+cvtmp_arr[j])*krs[j])
            ci_apple = Ci_largerThanFour_approxFunc((1.0+cvtmp_arr[j])*krs[j])
        
        if krs[j] < 4:
            si_banana = Si_lessThanFour_approxFunc(krs[j])
            ci_banana = Ci_lessThanFour_approxFunc(krs[j])

        else:
            si_banana = Si_largerThanFour_approxFunc(krs[j])
            ci_banana = Ci_largerThanFour_approxFunc(krs[j])

        rhotmp[j] = numpy.sin(krs[j]) * (si_apple - si_banana)
        rhotmp[j] = rhotmp[j] - numpy.sin(cvtmp_arr[j]*krs[j])/krs[j]/(1.0 + cvtmp_arr[j])
        rhotmp[j] = rhotmp[j] + numpy.cos(krs[j])*(ci_apple - ci_banana)

        rhoNFW[j] = rhoNFW[j] * rhotmp[j]

    return rhoNFW


#-------------------------------------------------------------------------------------#
# Gravity Model
#-------------------------------------#
# Hu - Sawicki   ------- sigma, delta_c, delta_vir

HS = {}
HS['dtac_file'] = CP['delta_sc']    #pre-calcuated delta_c file
HS['pok_file'] = CP['linear_mpk']  #linear Hu-Sawicki p(k) from EFTCAMB
HS['dtac_dat'] = np.loadtxt(HS['dtac_file'])

# GeoG_data = HS['dtac_dat'][:, 2] #G_{eff}/G_N
# ata_data = HS['dtac_dat'][:, 7] #scale_factor @ turn_around_point
# Delta_data = HS['dtac_dat'][:, 8] #nonlinear_density @ turn_around_point
# deti_data = HS['dtac_dat'][:, 3] #initial_density

# ai = 1.e-3  #initial time set for spherical collapse
# Rta_array = ata_data/ai*((1.0 + deti_data)/(1.0 + Delta_data))**0.33333   # Maximum Radius
# eta = 2.0*(1.0 - CP['omg_m'])/GeoG_data/CP['omg_m']/ata_data**(-3.0)/(1.0 + Delta_data)  # eta parameter

HS['sigma'] = sigm(HS['pok_file'])
HS['dtac_arr'] = delta_sc_nondyn * np.ones_like(r_arr)
HS['dtavir_arr'] = 8000.0 * np.ones_like(r_arr)
#-------------------------------------#

# LCDM ----- 

LCDM = {}
LCDM['pok_file'] = CP['lcdm_linear_mpk']  #linear LCDM p(k) from EFTCAMB
LCDM['sigma'] = sigm(LCDM['pok_file'])

lcdm_detc, lcdm_detvir = 1.67567, 392.704
LCDM['dtac_arr'] = lcdm_detc * r_arr**0.0
LCDM['dtavir_arr'] = lcdm_detvir * r_arr**0.0



# Halo Model  ---  Mass function, related bias, NFW density file, Pok
# non-dynamical approximation
class HaloModel_nondynApprox(object):

    def __init__(self, name, flag):
        print "initializing...."
        self.name = name
        self.flag = flag
        self.deltasc = self.name['dtac_arr']
        self.sigma = self.name['sigma']
        self.ovvir = self.name['dtavir_arr']
        
    
    def msfcbias(self):
        print "Calculating mass function and linear bias, please wait ... ..."
        mfcbias_arr = np.zeros([CP['num_r'], 3])
        mfcbias_arr[:, 0] = mass_arr

        nlnmbias = {'ZH': ZHfucplbias_nondyn}
        self.ndlnm, self.bias = eval("nlnmbias[self.flag](self.sigma, mass_arr, CP['num_r'], CP['rho_m'])")
        #mfcbias_arr = np.zeros([CP['num_r'], 3])
        #mfcbias_arr[:, 0] = mass_arr
        mfcbias_arr[:, 1] = self.ndlnm
        mfcbias_arr[:, 2] = self.bias
        return mfcbias_arr
        
        # if self.name == 'LCDM':
        #     self.ndlnm, self.bias 
        #     mfcbias_arr[:, 1], mfcbias_arr[:, 2] = STfucplbias(self.sigma, self.deltasc)
        #     return mfcbias_arr

            
    
    def NFWpro(self):
        logk_array = np.linspace(np.log10(CP['k_min']), np.log10(CP['kmax_halo']), CP['num_k'])
        k_array = 10.0**logk_array
        self.karr = k_array

        ppservers = ()
        ncpus = 8
        ppservers = ("10.0.0.1",)
        job_server = pp.Server(ncpus, ppservers=ppservers)
        #start_time = time.time()
        # The following submits 8 jobs and then retrieves the results
        inputs = k_array
        rhoNFW_arr = np.zeros([CP['num_k'], CP['num_r']])
        jobs = [(input, job_server.submit(NFW_func,(input, self.sigma, self.deltasc, self.ovvir, mass_arr, CP['rho_m'], CP['num_r']), (rhors,f_func,g_func,Si_lessThanFour_approxFunc,Ci_lessThanFour_approxFunc,Si_largerThanFour_approxFunc,Ci_largerThanFour_approxFunc), ("numpy","sympy"))) for input in inputs]
        i = 0
        for input, job in jobs:
            #print "NFW array", input, "is", job()
            job()
            rhoNFW_arr[i, :] = job()
            i = i + 1

        #print "Time elapsed: ", time.time() - start_time, "s"
        job_server.print_stats()
        self.NFW = rhoNFW_arr

    def pok(self):

        start_time_nb = time.time()
        self.msfcbias()
        print "Time elapsed (mass function and bias): ", time.time() - start_time_nb, "s"

        start_time_NFW = time.time()
        self.NFWpro()
        print "Time elapsed (NFW Fourier Transformation): ", time.time() - start_time_NFW, "s"


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

        print "Time elapsed (non-linear P[k] halo model integration): ", time.time() - start_time_Pok, "s"

        return pkk_arr
#############################


# HaloModel_nondynApprox(name, flag)
# {name, flag} = {HS, "ZH"} or {LCDM, "ST"}

# f(R) P(k)
pok_tmp = HaloModel_nondynApprox(HS, "ZH").pok()
file_tmp = CP['nonlinear_mpk']
if os.path.exists(file_tmp): 
    os.remove(file_tmp)
np.savetxt(file_tmp, pok_tmp)



print "Time elapsed (total): ", time.time() - start_time, "s"

exit()
