# CHAM
# developed by Xue-Wen LIU and Bin HU
# 2017-Dec

***************************
-description-

CHAM is a python script for the non-linear matter power spectrum for the alternative theories to LCDM. 

It contains two codes: 'Virial_calculator.py' and 'CHAM.py'

The former gives the critical density data, which is asked by the latter. 

The latter calculates the non-linear spectrum via the screened halo model. 

***************************
-installation request-

In order to run CHAM, you need firstly install/import 'sympy' module, 'parallel python' module and 'pyyaml' module. 

***************************
-run-

python Virial_calculator.py

python CHAM.py

***************************
-parameter setup-

The parameter file is 'CHAM_params.yaml'. 

Users can modify it. 

***************************
-output data-

After running 'Virial_calculator.py', the users shall get 'delta_sc.dat'.

'delta_sc.dat' contains the inter-mediate results: 

index, r [Mpc], G_{eff}/G_N, initial_density, scale_factor @ collapse, growth_factor(a=1)/growth_factor(a_i), critial_density, scale_factor @ turn_around_point, nonlinear_density @ turn_around_point

After running 'CHAM.py', the users shall get 'non-linear_mpk.dat'.

'non_linear_mpk.dat' contains the final non-linear matter power spectrum results: 

k [h/Mpc], 1halo term, I(k), linear P(k), non-linear P(k)

***************************
-contact-

If you have any question, please contact: 

liuxuewen14@itp.ac.cn (Xue-Wen Liu)

bhu@bnu.edu.cn (Bin Hu)
