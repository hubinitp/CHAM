cimport cython
from libc.math cimport sin, cos
import sympy

def rhoNFW_func_cython(double[:] rhoNFW, 
                       int len_r, 
                       double[:] rhostmp_arr, 
                       double[:] rstmp_arr, 
                       double[:] mstmp_arr, 
                       double[:] krs, 
                       double kk_var, 
                       double[:] rhotmp, 
                       double[:] cvtmp_arr):
        cdef int j
        for j in range(len_r):
                krs[j] = kk_var * rstmp_arr[j]
                rhotmp[j] = (sin(krs[j]) * (sympy.Si((1.0+cvtmp_arr[j])*krs[j]) - sympy.Si(krs[j])) ) - sin(cvtmp_arr[j] * krs[j]) / krs[j] / (1.0 + cvtmp_arr[j]) + cos(krs[j])*(sympy.Ci((1.0+cvtmp_arr[j])*krs[j]) - sympy.Ci(krs[j]))
                rhoNFW[j] = (4.0*3.1415926535897932384626433 * rhostmp_arr[j] * (rstmp_arr[j]**3.0) / mstmp_arr[j])* rhotmp[j]