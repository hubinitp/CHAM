cimport cython
from libc.math cimport sqrt, abs, exp


def gg2_func_cython(double[:, :] gg2_array, 
                    double[:, :] barosig2_array,
                    int num_sig2new,
                    double[:] bar_array,
                    double[:] bardta_array,
                    double[:] sigm2_new,
                    double[:] sigm2dta_new,
                    double[:] dbards_array):
        cdef int i
        cdef int j
            #for i in range(num_sig2new):
            #for j in range(num_sig2new):
            #             barosig2_array[i,j] = (bar_array[i]-bardta_array[j])/(sigm2_new[i]-sigm2dta_new[j])
            #             gg2_array[i,j] = (2.0*dbards_array[i] - barosig2_array[i, j]) * Gaus(bar_array[i]-bardta_array[j],sigm2_new[i]-sigm2dta_new[j])
            # def Gaus(ave, S):         return numpy.exp(-0.5*ave**2.0/abs(S)) / numpy.sqrt(2.0*numpy.pi*abs(S))

        for i in range(num_sig2new):
                for j in range(num_sig2new):
                        barosig2_array[i, j] = (bar_array[i]-bardta_array[j])/(sigm2_new[i]-sigm2dta_new[j])
                        gg2_array[i, j] = (2.0*dbards_array[i] - barosig2_array[i, j]) * exp(-0.5*(bar_array[i]-bardta_array[j])**2.0 / abs(sigm2_new[i]-sigm2dta_new[j])) / sqrt(2.0*3.1415926535897932384626433*abs(sigm2_new[i]-sigm2dta_new[j]))

def fS_func_cython(double[:] ffunc, 
                   int num_sig2new, 
                   double[:] integ_array, 
                   double[:, :] delta_array, 
                   double[:] gg1_array):
        cdef int i 
        cdef int j
        for i in range(2, num_sig2new):
                for j in range(1, i):
                        integ_array[i] += ffunc[j]*(delta_array[i, j] + delta_array[i, j+1])
                ffunc[i] = (1.0 - delta_array[i, i])**(-1.0)
                ffunc[i] = ffunc[i] * (gg1_array[i] + integ_array[i])
