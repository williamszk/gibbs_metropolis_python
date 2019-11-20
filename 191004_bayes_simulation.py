
#preliminaries ##################################################
pwd()
#cd C:\Users\willi>ipython
%cd ~/Dropbox/FEA RP materias/materia bayesiana
%matplotlib tk


#load packages #################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.stats import gamma
from scipy.stats import norm
from random import choices
import math

## generate data ###############################################
"""
consider a problem where we want to estimate the mean of a Gaussian 
distribution
DGP: mean=2; variance=1; n=10 number of observations
"""
np.random.seed(seed=123)
rv1 = np.random.normal(loc=2, scale=1, size=10)#rv1 is our random sample.

#change format to a data frame
rv3 = pd.DataFrame(rv1) 
rv3.shape
rv3.head()

"""
for this problem we assume that we know the variance, variance =1
the objective is to estimate the mean, for this we'll use 
the Ordinary Monte Carlo method
"""

m1 = np.mean(rv1)~/Dropbox/FEA RP materias/materia bayesiana
m1

#############################################################################
## Ordinary Monte Carlo #####################################################
#############################################################################


#create a simulation of size 1000
np.random.seed(seed=321)
rv2 = np.random.normal(loc=m1, scale=1, size=1000) #mean=m1

plt.figure(figsize=(10,5)) #plot histogrm
plt.hist(rv2, bins=100)

#estimation for the mean
np.mean(rv2)

#estimation for credible interval
np.quantile(rv2,[.05,.95])



#############################################################################
## Sampling Importance Resampling (SIR) #####################################
#############################################################################


#suppose that we have a posterior with a Gamma(3,3) format, but we dont know.
#use gaussian in the sampling step
#loc1, scale1 are from the Gaussian
loc1=0
scale1=500 #scale is standard deviation
norm.mean(loc=loc1, scale=scale1)
norm.var(loc=loc1, scale=scale1)
norm.std(loc=loc1, scale=scale1)

np.random.seed(seed=123)
rv1 = np.random.normal(loc=loc1, scale=scale1, size=10000)
rv1[0:10]
np.mean(rv1)
np.std(rv1)
np.var(rv1)
np.max(rv1)
np.min(rv1)

b1, b2 = -5,5
x = np.linspace(b1,b2,100)
y = norm.pdf(x=x, loc=loc1, scale=scale1)
fig, ax = plt.subplots(1, 1) #it creates another plot
ax.plot(x, y,'b-', lw=2, alpha=0.6, label='Gaussian pdf')

#Gamma(3,3) # we call it the true posterior
a = 3
b = 3
scale2=1/b
b1, b2 = gamma.ppf(q=[0.01,0.99], a=a, scale=scale2)
x = np.linspace(b1,b2, 100)
y = gamma.pdf(x=x, a=a, scale=scale2)
#fig, ax = plt.subplots(1, 1) #it creates another plot
ax.plot(x, y,'r-', lw=2, alpha=0.6, label='gamma pdf')

"""
We can see from the figure that the Gaussian distribution is very flat
due to the Gaussian be spread in a large area.
The ideal situation is that on the step of first sampling the support
of sampling is close to the posterior's majority support (places inside 1% and 99%). 
But we don't know the posterior format nor majority support.
Idea: We could make in this an iterative process where we try to find
a close enough area where is the majority support.
"""

#Normal(0,100), what are the .01 and .99 percentiles?
norm.ppf(q=[.01,.99],loc=loc1, scale=scale1)

#p1 probabilities of Gaussian random sample
p1 = norm.pdf(x=rv1,loc=loc1, scale=scale1)
#g1 probabilities of random sample from true distribution Gamma(3,3)
g1 = gamma.pdf(x=rv1, a=a, scale=scale2)
#"w" weights for the resampling step
w=g1/p1
len1 = len(w)
fig, ax = plt.subplots(1, 1) 
ax.plot( w,'r-', lw=2, alpha=0.6, label='graph of weights "w"')
#peaks show where the weights were high

#show proportion of samples that got high weight
y1=np.sort(w)
y2 = y1[-200:-1]
fig, ax = plt.subplots(1, 1) 
ax.plot(y2,'r-', lw=2, alpha=0.6, label='')

#number of points that have more than 0.0001 probability of being drawn
#from a total of 10000 
np.sum(w>0.0001)

#make data frame with weights and value from first draw rv1
df1 = pd.DataFrame({"sample":rv1,
					"weights":w})
df1.head()
#keep only lines that have weights higher than 0.0001

#rvc1 random vector clean 1
rvc1=rv1[w>0.0001]
len(rvc1)
#wc1 weights clean 1
wc1 = w[w>0.0001]

#sampling with reposition and with weights
dd1 = choices(population=rvc1, #values to be drawn
	weights=wc1, #weights of drawn
	k=1000) #number of draws
#dd1

#plot histogram of estimated posterior
plt.figure(figsize=(10,5)) 
plt.grid(True)
plt.hist(dd1, bins=100,color='b',density=True)

#plot histogram of estimated posterior (SIR)
#and true posterior
fig, ax = plt.subplots(1, 1) #it creates another plot
plt.grid(True)
ax.hist(dd1, bins=100,color='b',density=True)
ax.plot(x, y,'r-', lw=3, alpha=1, label='Gaussian pdf')


"""
Compare the moments generated from the SIR method and the true moments.
True mean is 1.
True variance is 0.33
"""
#SIR posterior mean, it should be close to 1
np.mean(dd1)

#SIR posterior variance, it should be close to .33
np.var(dd1)


#########################################################################
#Markov Chain Monte Carlo methods #######################################
#########################################################################


##########################################################################
## Gibbs Sampling ########################################################
##########################################################################

#a simple example:
#we have three parameters 
#alpha -- Gamma(beta,gamma)
#beta -- Exponential(alpha)
#gamma -- Beta(beta, alpha) 


#generate initial random values 
chainSize = 100 #size of chain
alpha=np.zeros(chainSize)
beta=np.zeros(chainSize)
gamma=np.zeros(chainSize)

alpha[[0,1]] = 1 
beta[[0,1]] = 1
gamma[[0,1]] = 1 

ii=2
#generate the chain
for ii in list(range(2, chainSize)):
	
	#artifice to counter 0 or infinite values, give a random beta(1,1) value instead
	if alpha[ii-1] == 0 or alpha[ii-1] == float("inf") or math.isnan(alpha[ii-1]):
		alpha[ii-1] = alpha[ii-2] #do not accept value found
	if beta[ii-1] == 0 or beta[ii-1] == float("inf") or math.isnan(beta[ii-1]):
		beta[ii-1] = beta[ii-2]
	if gamma[ii-1] == 0 or gamma[ii-1] == float("inf") or math.isnan(gamma[ii-1]):
		gamma[ii-1] = gamma[ii-2]
	
	#Monte Carlo Step
	alpha[ii] = np.random.gamma(shape= beta[ii-1], scale=1/gamma[ii-1],size=1)
	beta[ii] = np.random.exponential(scale=1/alpha[ii-1],size=1)
	gamma[ii] = np.random.beta(beta[ii-1], alpha[ii-1], size=1)	


ii += 1 #just for testing the loop

fig, ax = plt.subplots(1, 1) 
ax.plot( alpha,'r-', lw=2, alpha=0.6)
ax.plot( beta,'b-', lw=2, alpha=0.6)
ax.plot( gamma,'g-', lw=2, alpha=0.6)

alpha>np.var(alpha)

np.var(beta)*2
np.var(gamma)*2

#
#This example is not working, chains are not changing
#

##second try ################################################################
# Gaussian, with priors Gaussian and Inverted Gamma #########################
#############################################################################
"""
Idea: a random variable have a Gaussian distribution. Mean is also Gaussian.
Variance have a inverted Gamma so that finding the posterior is easier.
Make a Gibbs sampler from this.
"""

#Build true process, Gaussian(4,2)
#mean=4
#var=2

np.random.seed(seed=123)
#rs1 is our random sample 1
rs1 = np.random.normal(loc=4, scale=2, size=100)




# DO NOT RUN
# begin testing -----------------------------------------------------------
#in the inverse gamma function a=alpha and scale=beta
ig1 = scipy.stats.invgamma.rvs(a=3,loc=0, scale=1, size=1000000)
np.mean(ig1)
np.var(ig1)

a = 3
b = 1
#variance of inverse gamma
b**2/((a-1)**2*(a-2))

mean, var = invgamma.stats(a, loc=0, scale=1, moments='mv')
mean
var


#testing for np.random.normal
std = .5
std**2
rn1 = np.random.normal(loc= 0, scale=std,size=1000)
np.mean(rn1)
np.var(rn1)
# end testing----------------------------------------------------------

#############################################################################
# building loop for chains in Gibbs sampler #################################
#############################################################################

#generate initial random values 
chainSize = 10000 #size of chain
#crate empty lists
mu=np.full(chainSize, np.nan)
sigma_sq=np.full(chainSize, np.nan)

mu[0] = 1
sigma_sq[0] = 1

ii=1
#generate the chain
for ii in list(range(1, chainSize)):
	#Monte Carlo Step
	#--steps for conditional mu
		#assign values for a1 and c1
	a1 = (chainSize*10**6+sigma_sq[ii-1])/(2*sigma_sq[ii-1]*10**6)
	c1 = ((np.sum(rs1))**2*10**6)/(sigma_sq[ii-1]*2*(chainSize*10**6+sigma_sq[ii-1]))

	m_gauss = (c1/a1)**.5
	v_gauss = 1/2*a1

	mu[ii] = np.random.normal(loc=m_gauss, scale=v_gauss**.5,size=1)
	
	#--steps for conditional sigma_sq
	m_ig = (6.002+chainSize)/(2)
	v_ig = (np.sum((rs1-mu[ii-1])**2)+20)/(2)

	sigma_sq[ii] = scipy.stats.invgamma.rvs(a=m_ig,loc=0, scale=v_ig, size=1)


#############################################################################
# builidng chain graphs, statistics of distribution #########################
#############################################################################

#chain graph of mu
fig, ax = plt.subplots(1, 1) 
ax.plot(mu,'r-', lw=2, alpha=0.6)

#chain graph of sigma_sq
fig, ax = plt.subplots(1, 1) 
ax.plot(sigma_sq,'b-', lw=2, alpha=0.6)

#transform np.array into series
mu_pd = pd.Series(mu)
sigma_sq_pd = pd.Series(sigma_sq)

#statistics of mu
mu_pd.describe()

#statistics of sigma_sq
sigma_sq_pd.describe()




















