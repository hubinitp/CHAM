# CHAM

**developed by Bin HU and Xue-Wen LIU**

**2017-Dec**

**optimized by Cheng-Zong RUAN**

**2019-Mar**

###########################
# description

python2.7 or python 3.7

CHAM is a python script for the non-linear matter power spectrum for the alternative theories to LCDM. 

It contains two version of the code. 

v1: Virial_calculator.py and CHAM_ori.py

v2: cham_nondyn.py

In v1, the Virial_calculator gives the critical density data, which is asked by CHAM_ori.py 

In v2, cham_nondyn.py does not rely on the output of Virial_calculator.py any more, it can run independently. 

Both CHAM_ori and cham_nondyn calculate the non-linear spectrum via the screened halo model. 

CHAM_ori: full scenario but a bit slow...

cham_nondyn: using non-dynamical approximation (causing ~0.7% deviation) and other optimization

###########################
# installation request

In order to run CHAM, you need firstly install/import 'sympy' module, 'parallel python' module and 'pyyaml' module. 

The GCC compiler is needed to compile Cython extensions.

###########################
# run (original version)

python Virial_calculator.py

python CHAM_ori.py

###########################
# run (accelerated version)

CC=gcc python setup.py build_ext --inplace

python cham_nondyn.py

###########################
# parameter setup

The parameter file is 'CHAM_params.yaml'. 

Users can modify it. 

###########################
# output data

After running 'Virial_calculator.py', the users shall get 'delta_sc.dat'.

'delta_sc.dat' contains the inter-mediate results: 


| columns | meaning |
| ------- | ------- |
| 0       | index   |
| 1       | r [Mpc] | 
| 2       | G_{eff}/G_N |
| 3       | initial_density |
| 4       | scale_factor @ collapse | 
| 5       | growth_factor(a=1) / growth_factor(a_i) | 
| 6       | critial_density |
| 7       | scale_factor @ turn_around_point | 
| 8       | nonlinear_density @ turn_around_point |


After running 'CHAM.py' or 'cham_nondyn.py', the users shall get 'non-linear_mpk.dat'.

'non_linear_mpk.dat' contains the final non-linear matter power spectrum results: 

| columns | meaning |
| ------- | ------- |
| 0       | k [h/Mpc] | 
| 1       | 1-halo term |
| 2       | I(k) |
| 3       | linear P(k) | 
| 4       | non-linear P(k) |

###########################
# contact

If you have any question, please contact: 

bhu@bnu.edu.cn (Bin Hu)
liuxuewen14@itp.ac.cn (Xue-Wen Liu)
chzruan@mail.bnu.edu.cn (Cheng-Zong Ruan)
