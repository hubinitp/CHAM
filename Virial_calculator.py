# Xue-wen Liu @10.31 2017
# This Python code calculate spherical collapse evolution in
# modify gravity-including characteristic density perturbation, evolution 
# linear density perturbation
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy 
import math
import random
import os, time
import pp
import scipy.optimize

# initial time, matter parameter, initial overdensity
CP = {'r_min':0.005, 'r_max':150.0, 'num_r':110}

# top-hat radius function
def top(x):
        top = (1.0+CP['c2']*x**(-3.0))**(1.0/(CP['alpha']-1.0))
        top = top - (1.0+CP['c3']*x**(-3.0))**(1.0/(CP['alpha']-1.0))
        top = -CP['c1']*x*top
        return top

# six equations 

def equations(p):
        a1, b1, r01, a2, b2, r02 = p
        f1 = CP['Bb']*b1/r01**(a1*(b1-1.0)/b1) - 3.0*CP['c1']*CP['c3']**(1.0/(CP['alpha']-1))/(3.0+2.0*CP['omg_fr'])
        f2 = a1*(b1 - 1.0)/b1 - (4.0-CP['alpha'])/(1.0-CP['alpha'])
        f3 = (3.0*top(r01) - 3.0*top(r01)**2 + top(r01)**3)/(3.0+2*CP['omg_fr'])
        f3 = f3 - CP['Bb']*b1*(2**(1.0/b1) - 1.0)
        f4 = CP['Bb']*b2/r02**(a2*(b2-1.0)/b2) - 3.0*CP['c1']*CP['c2']/(3.0+2.0*CP['omg_fr'])
        f5 = a2*(b2 - 1.0)/b2 + 2.0
        f6 = (3.0*top(r02) - 3.0*top(r02)**2 + top(r02)**3)/(3.0+2*CP['omg_fr'])
        f6 = f6 - CP['Bb']*b2*(2**(1.0/b2) - 1.0)

        return (f1,f2,f3,f4,f5,f6)

def f(p):
        return abs(sum(numpy.array(equations(p))**2)-0.0)

# Solving equations
def Chame(Rsha):
	#final parameter estimation is quite sensitive to the initial value setting
	aa = scipy.optimize.fmin(f,(8.04,6.01,0.32,-5.62,1.42,27.6))
	param_dat = scipy.optimize.fsolve(equations,(aa[0],aa[1],aa[2],aa[3],aa[4],aa[5]))
#	a1, b1, r01, a2, b2, r02 = (param_dat[i] for i in range(6))
	a1, b1, r01, a2, b2, r02 = 7.2583, 28.0987, 0.7555, -24.1235, 1.0904, 13.5432
        if Rsha < (r01 + r02)/2.0:   
                return b1*((Rsha/r01)**a1*(1+(r01/Rsha)**a1)**(1/b1)-(Rsha/r01)**a1)*CP['Bb']+1

        else:
                return b2*((Rsha/r02)**a2*(1+(r02/Rsha)**a2)**(1/b2)-(Rsha/r02)**a2)*CP['Bb']+1


def rk4(x, y, v, fa, dt):
# Coefficients used to compute the independent variable argument of f
        a2  =   2.500000000000000e-01  #  1/4
        a3  =   3.750000000000000e-01  #  3/8
        a4  =   9.230769230769231e-01  #  12/13
        a5  =   1.000000000000000e+00  #  1
        a6  =   5.000000000000000e-01  #  1/2
# Coefficients used to compute the dependent variable argument of f
        b21 =   2.500000000000000e-01  #  1/4
        b31 =   9.375000000000000e-02  #  3/32
        b32 =   2.812500000000000e-01  #  9/32
        b41 =   8.793809740555303e-01  #  1932/2197
        b42 =  -3.277196176604461e+00  # -7200/2197
        b43 =   3.320892125625853e+00  #  7296/2197
        b51 =   2.032407407407407e+00  #  439/216
        b52 =  -8.000000000000000e+00  # -8
        b53 =   7.173489278752436e+00  #  3680/513
        b54 =  -2.058966861598441e-01  # -845/4104
        b61 =  -2.962962962962963e-01  # -8/27
        b62 =   2.000000000000000e+00  #  2
        b63 =  -1.381676413255361e+00  # -3544/2565
        b64 =   4.529727095516569e-01  #  1859/4104
        b65 =  -2.750000000000000e-01  # -11/40
# Coefficients used to compute local truncation error estimate.  These come from subtracting
# a 4th order RK estimate from a 5th order RK estimate.
        r1  =   2.777777777777778e-03  #  1/360
        r3  =  -2.994152046783626e-02  # -128/4275
        r4  =  -2.919989367357789e-02  # -2197/75240
        r5  =   2.000000000000000e-02  #  1/50
        r6  =   3.636363636363636e-02  #  2/55
# Coefficients used to compute 4th order RK estimate
        c1  =   1.157407407407407e-01  #  25/216
        c3  =   5.489278752436647e-01  #  1408/2565
        c4  =   5.353313840155945e-01  #  2197/4104
        c5  =  -2.000000000000000e-01  # -1/5

        y1, v1 = y, v
        ac1 = fa(x, y1, v1)

        y2, v2 = y + b21*v1*dt, v + b21*ac1*dt
        ac2 = fa(x+a2*dt, y2, v2)

        y3, v3 = y + b31*v1*dt + b32*v2*dt, v + b31*ac1*dt + b32*ac2*dt
        ac3 = fa(x+a3*dt, y3, v3)

        y4, v4 = y + b41*v1*dt + b42*v2*dt + b43*v3*dt, v + b41*ac1*dt + b42*ac2*dt + b43*ac3*dt
        ac4 = fa(x+a3*dt, y4, v4)

        y5 = y + b51*v1*dt + b52*v2*dt + b53*v3*dt + b54*v4*dt
        v5 = v + b51*ac1*dt + b52*ac2*dt + b53*ac3*dt + b54*ac4*dt
        ac5 = fa(x+a4*dt, y4, v4)

        y6 = y + b61*v1*dt + b62*v2*dt + b63*v3*dt + b64*v4*dt + b65*v5*dt
        v6 = v + b61*ac1*dt + b62*ac2*dt + b63*ac3*dt + b64*ac4*dt + b65*ac5*dt
        ac6 = fa(x+a5*dt, y4, v4)

        yf = y + (c1*v1 + c3*v3 + c4*v4 + c5*v5)*dt
        vf = v + (c1*ac1 + c3*ac3 + c4*ac4 + c5*ac5)*dt

        r = max(abs( r1 * v1 + r3 * v3 + r4 * v4 + r5 * v5 + r6 * v6 ), abs( r1 * ac1 + r3 * ac3 + r4 * ac4 + r5 * ac5 + r6 * ac6 ))
        return (r, yf, vf)

# linear derivative function
def dverli(x,y,yp):

	dverli = -0.5*yp*(CP['omg_m']*numpy.exp(-3*x)+4*(1.0-CP['omg_m']))/(CP['omg_m']*numpy.exp(-3*x)+(1.0-CP['omg_m'])) 
	dverli = dverli + 1.5*CP['GeoG']*CP['omg_m']*numpy.exp(-3*x)/(CP['omg_m']*numpy.exp(-3*x)+(1.0-CP['omg_m']))*y
	return dverli

#y'' = A*y' + B*y + C..... 
def gRx(x,y,yp):
	gRx = 1.5*CP['omg_m']*numpy.exp(-3*x)/(CP['omg_m']*numpy.exp(-3*x)+(1.0-CP['omg_m']))*yp
	gRx = gRx - 0.5*y*(CP['omg_m']*numpy.exp(-3*x)-2*(1.0-CP['omg_m']))/(CP['omg_m']*numpy.exp(-3*x)+(1.0-CP['omg_m']))
	gRx = gRx - (y+numpy.exp(x)/CP['ai'])*0.5*((1+CP['deti'])*(CP['ai']*numpy.exp(-x)*y+1)**(-3)-1)*CP['GeoG']*CP['omg_m']*numpy.exp(-3*x)/(CP['omg_m']*numpy.exp(-3*x)+(1.0-CP['omg_m']))
	return gRx	

def odesolver(Rr):
        
	x = {}
	yypha = {}
	lngr = {}
	yrsp = {}
	flag = xmin = DaDai = rha = rl = dclnha = flag_max = 0
	
	global CP

	CP = {'c1':0.66667, 'c2':100.0, 'c3':0.1, 'alpha':0.5, 'Bb':0.33333, 'omg_fr':0.0}
	CP['GeoG'] = Chame(Rr)
	print "GeoG, ", CP['GeoG']
	CP['omg_m'] = 0.24
	CP['ai'] = 1.0e-3
	a_max = delt_max = 0
        ppf = -1.25 + 1.25*numpy.sqrt(0.04+0.96*CP['GeoG'])  
        lngr[0, 0], lngr[1, 0] = CP['ai'], CP['ai']*(1.0 + ppf)      # modify initial condition
	metol = 1.e-4	  # when considering R(a) = 0
	CP['deti'] = 2.315e-3
	
	while(flag != 1):
	
        	i, errct = 0, 0
		hmax, hmin, tol, hs = 8.e-4, 6.e-4, 2.e-4, 8.e-4
		flag = 0
		x[0], yrsp[0] = numpy.log(CP['ai']), 1.0  # initial time and position  ai???
		yypha[0, 0], yypha[1, 0] = 0, -CP['deti']*(1.0 + ppf)/3.0
#		print "zero-component, ", yypha[1,0]
#		print "zero-component, ", lngr[1,0]
	        while(x[i] <= 0):
		# x = ln(a), yrsp = r/r_i,  yypha = r/r_i - a/a_i,  lngr = growth function: D(a)
	                (rha, yypha[0,i+1], yypha[1,i+1]) = rk4(x[i], yypha[0,i], yypha[1,i],gRx, hs)
	                if rha <= tol:
	                        hs = hs
# Now compute next step size, and make sure that it is not too big or
# too small.
                	else:
	                        hs = hs * min( max( 0.84 * ( tol / rha )**0.25, 0.1 ), 4.0 )
	                        if hs > hmax: hs = hmax
                	        elif hs < hmin: hs = hmin
	                x[i+1] = x[i] + hs
        	        yrsp[i+1] = yypha[0,i+1] + numpy.exp(x[i+1])/CP['ai']
	                (rl, lngr[0,i+1], lngr[1,i+1]) = rk4(x[i], lngr[0,i], lngr[1,i], dverli, hs)
#			f2 = open("test_A.txt", "a+b")
#	                f2.write(str(i)+" "+str(x[i])+" "+str(CP['deti'])+" "+str(yypha[0, i])+" "+str(lngr[0, i])+"\n")  # i + Radius + GeoG + delta_i + a_c + D(a=0)/D(a_i) + delta_c + a_max + \Delta_max
#        	        f2.close()

        	        if yrsp[i+1] <= yrsp[i]:
                	        if flag_max == 0 :
                        	        a_max = numpy.exp(x[i])
                       	       	        delt_max = (1.0 + CP['deti'])*yrsp[i]**(-3.0)*(a_max/CP['ai'])**3.0 - 1.0
                        	        flag_max = 1

	                        if yrsp[i] <= metol:    # x[i+1] may not satisfy: x[i+1] <= 0
        	                        xmin = x[i]
                	                DaDai = lngr[0,i]/lngr[0,0]
                        	        dclnha = CP['deti']*DaDai
					print "initial_overdensity:, ", CP['deti']
 	                                if numpy.exp(xmin) >= 0.995 and numpy.exp(xmin) <= 1:
        	                                tmin = i 
                	                        flag = 1
						print "finished !"
                        	                break
                    	                else:
						flag = 2
	                                        break
       	        	i = i + 1
                print numpy.exp(x[i])
# choose to modify initial condition
#		flag = 1
                if flag == 1:
                        f1 = open("halocrit.txt", "a+b")
                        f1.write(str(1)+" "+str(Rr)+" "+str(Chame(Rr))+" "+str(CP['deti'])+" "+str(numpy.exp(xmin))+" "+str(DaDai)+" "+str(dclnha)+" "+str(a_max)+" "+str(delt_max)+"\n")  #Radius + GeoG + delta_i + a_c + D(a=0)/D(a_i) + delta_c + a_max + \Delta_max
                        f1.close()
                        errct = 0
                        break
                elif flag == 0:
                        print "retrying-forward"
                        CP['deti'] = CP['deti'] + 3.0e-5
                        if errct == 2:
                                CP['deti'] = random.uniform(1.5e-3, 2.4e-3)
                                errct = 0
                        errct = errct + 1
                elif flag == 2:
                        print "retrying-backward"
                        if numpy.exp(xmin) <= 0.4: CP['deti'] = CP['deti'] - 4.0e-4
                        if numpy.exp(xmin) >0.4 and numpy.exp(xmin) <= 0.65: CP['deti'] = CP['deti'] - 3.0e-4
                        if numpy.exp(xmin) >0.65 and numpy.exp(xmin) <= 0.86: CP['deti'] = CP['deti'] - 1.2e-4
                        if numpy.exp(xmin) >0.86 and numpy.exp(xmin) <= 0.975: CP['deti'] = CP['deti'] - 2.5e-5
                        else:
                                if errct == 1: CP['deti'] = CP['deti'] - 1.5e-6*random.uniform(0.2, 1.2)
                                else: CP['deti'] = CP['deti'] - 3.e-6

        return (numpy.exp(xmin), dclnha, DaDai, a_max, delt_max)

# different radius
logR_arr = numpy.linspace(numpy.log10(CP['r_min']), numpy.log10(CP['r_max']), CP['num_r'])
Rr_arr = 10.0**logR_arr

if os.path.exists("halocrit.txt"): os.remove("halocrit.txt")

ppservers = ()
ncpus = 8
#ppservers = ("10.0.0.1",)
job_server = pp.Server(ncpus, ppservers=ppservers)

start_time = time.time()
# The following submits 8 jobs and then retrieves the results
inputs = Rr_arr

#Test
#test_scale = 0.01
#sigma = odesolver(test_scale)
#print sigma
jobs = [(input, job_server.submit(odesolver, (input,), (Chame,dverli,gRx,rk4,f,equations,top,), ("numpy", "scipy.optimize",))) for input in inputs]

for input, job in jobs:
	print "Output: ", job()

print "Time elapsed: ", time.time() - start_time, "s"
job_server.print_stats()

os.rename("halocrit.txt", "halocritemp.txt")

exit()
