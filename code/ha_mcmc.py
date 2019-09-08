import numpy as np
import matplotlib.pyplot as plt
import starry

#import exoplanet as exo
import emcee


from multiprocessing import Pool


time, vels, verr, ha = np.loadtxt('../data/transit.vels', usecols=[0,1,2,4], unpack=True)
time -= 2458706.5

map = starry.Map(ydeg=4, udeg=2, rv=True, lazy=False)
map.reset()


Prot = 2.85             # days
P = 8.1387              # days
e = 0.0
w = 0.0
inc = 90.0

tuse = time + 0.0
euse = verr + 0.0
vuse = vels + 0.0

bnds = ((12000, 24000), (0.04, 0.09), (-1.0, 0.0), (15,25), (0,1),(0,1), (-30,90), (-20,50), (0.0, 20.0), (0.16, 0.175), (-100000, 0))



vsini_mu = 18300
vsini_sig = 1800

ndim = 11
nwalkers = 500

init = np.array([19300, 0.0688, -0.09, 20.79, 0.5, 0.40, 0.0, 20.0, 1.0, 0.166, -35000])



def rmcurve(params):

    vsini, r, b, a, u1, u2, obl, gamma, jitter_good, t0, ha_fac = params
    veq = vsini / np.sin(inc * np.pi / 180.0)
    
    if u1 + u2 > 1.0:
        print('inf')
        return 2700

    map.reset()

    map.inc = inc
    map.obl = obl
    #map.add_spot(spot_amp, sigma=spot_sig, lon=spot_lon, lat=-spot_lat)
    map[1:] = [u1, u2]
    map.veq = veq


    f = (tuse - t0)/P*2*np.pi
    I = np.arccos(b/a)

    zo = a*np.cos(f) 
    yo = -a*np.sin(np.pi/2+f)*np.cos(I)
    xo = a*np.sin(f)*np.sin(I)


    theta = 360.0 / Prot * tuse

    rv_0 = map.rv(xo=xo, yo=yo, zo=zo, ro=r, theta=theta)
    
    trend = ha_fac * (ha-0.055)
    rv = rv_0 + trend + gamma
    
    
    var_good = (euse**2 + jitter_good**2)
   
    goodgauss = 1.0 / np.sqrt(2*np.pi*var_good) * np.exp(-(rv-vuse)**2/(2*var_good))
    lnprob = np.log(goodgauss)

    #print(-1 * lnprob)
    return np.sum(lnprob)

def set_priors(params):
    vsini, r, b, a, u1, u2, obl, gamma, jitter_good, t0, ha_fac = params
    lnprior = 0

    if not all(b[0] < v < b[1] for v, b in zip(params, bnds)):
        return -np.inf

    if u1 + u2 > 1.0:
        return -np.inf

    lnprior -= 0.5*(vsini - vsini_mu)**2/vsini_sig**2

    return lnprior



def lnprob(params):
    lnprior = set_priors(params)
    if np.isfinite(lnprior) == False:
        return -np.inf
    else:
        lnlike = rmcurve(params)
        #print(lnlike, lnprior)
        return lnlike + lnprior

def setup_data(params):
    init = np.zeros(ndim)
    p0 = np.tile(params, nwalkers)
    p0 = p0.reshape(nwalkers, ndim)

    varystd = np.ones(ndim)*0.2
    varystd[0] = 100.0
    varystd[1] = 0.005
    varystd[2] = 0.05
    varystd[4] = 0.02
    varystd[5] = 0.02
    varystd[7] = 2.0
    varystd[9] = 0.001
    varystd[10] = 200.0


    for i in range(len(varystd)):
        p0[:,i] = p0[:,i] + np.random.normal(0, varystd[i], nwalkers)

    return p0

p0 = setup_data(init)

with Pool(24) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
    pos, prob, state = sampler.run_mcmc(p0, 3000, progress=True)


best = np.where(sampler.flatlnprobability == np.max(sampler.flatlnprobability))[0][0]
print(sampler.flatlnprobability[best])

np.save('../runs/chain_ha', sampler.chain)
np.save('pos', pos)
np.save('prob', prob)


