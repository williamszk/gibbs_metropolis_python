%cd C:\Users\willi\Dropbox\materia bayesiana\python_bayesian
%matplotlib tk
%run 191104_packages.py

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html

#true posterior
a = 1
b = 1
mean, var, skew, kurt = gamma.stats(a=a,scale=1/b,moments='mvsk')

x = np.linspace(gamma.ppf(0.01, a),gamma.ppf(0.99, a), 100)
fig, ax = plt.subplots(1, 1)
ax.plot(x, gamma.pdf(x, a),'r-', lw=5, alpha=0.6, label='gamma pdf')

gamma.pdf(1, a=1, scale=1)


rv = gamma(a,b)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')


#-------------------------------------------------
a = 3
b = 3
scale=1/b
x=1
gamma.pdf(x=x, a=a, loc=loc, scale=scale)

mean, var = gamma.stats(a=a, loc=loc, scale=scale, moments='mv')
b1, b2 = gamma.ppf(q=[0.01,0.99], a=a, scale=scale)
x = np.linspace(b1,b2, 100)
fig, ax = plt.subplots(1, 1) #it creates another plot
y = gamma.pdf(x=x, a=a, scale=scale)
ax.plot(x, y,'r-', lw=2, alpha=0.6, label='gamma pdf')































