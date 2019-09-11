import numpy as np
import matplotlib.pyplot as plt
import starry

#import exoplanet as exo
import emcee


from multiprocessing import Pool


time, vels, verr = np.loadtxt('../data/vst222259.ascii', usecols=[1,2,3], unpack=True)
s, ha = np.loadtxt('../data/222259_mbcvel.ascii', usecols=[-2, -1], unpack=True)

time = time[:-4]
vels = vels[:-4]
verr = verr[:-4]
ha   = ha[:-7]
s    = s[:-7]

time -= 18706.5

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

bnds = ((12000, 24000), (0.04, 0.09), (-1.0, 0.0), (15,25), (0,1),(0,1), (-30,90), (-500,500), (0.0, 3.0),
        (0, 40.0), (0.0, 1.0), (0.16, 0.175), (-100000, 0), (-1000, 0.0539), (0.0, 10.0),
        (-100000, 0), (-1000, 0.539), (0.0, 10.0))



vsini_mu = 18300
vsini_sig = 1800

rp_mu = 0.057
rp_sig = 0.003

ndim = 18
nwalkers = 500

init = np.array([19300, 0.0688, -0.09, 20.79, 0.5, 0.40, 0.0, 100.0, 1.0, 5.0, 0.7, 0.166, -15000, 0.051, 1.0, -1500, 0.51, 1.0])



def rmcurve(params):

    vsini, r, b, a, u1, u2, obl, gamma, jitter_good, jitter_bad, q, t0, ha_fac, ha_gam, ha_exp, s_fac, s_gam, s_exp = params
    veq = vsini / np.sin(inc * np.pi / 180.0)
    
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
    
    trend = ha_fac * (ha-ha_gam)**ha_exp + s_fac * (s-s_gam)**s_exp + gamma
    rv = rv_0 + trend

    #print((ha-ha_gam)**ha_exp)
    #print(rv)
    
    var_good = (euse**2 + jitter_good**2)
    var_bad  = (euse**2 + jitter_bad**2)
   
    goodgauss = q / np.sqrt(2 * np.pi * var_good) * np.exp(-(rv - vuse) ** 2 / (2 * var_good))
    badgauss = (1-q) / np.sqrt(2 * np.pi * var_bad) * np.exp(-(rv - vuse) ** 2 / (2 * var_bad))
    #badgauss = (1 - q) / np.sqrt(2 * np.pi * var_bad) * np.exp(-(rv_0 * 0.75 + trend - vuse) ** 2 / (2 * var_bad))
    lnprob = np.log(goodgauss + badgauss)

    #print(-1 * lnprob)
    return np.sum(lnprob)

def set_priors(params):
    vsini, r, b, a, u1, u2, obl, gamma, jitter_good, jitter_bad, q, t0, ha_fac, ha_gam, ha_exp, s_fac, s_gam, s_exp = params
    lnprior = 0

    if not all(b[0] < v < b[1] for v, b in zip(params, bnds)):
        return -np.inf

    if u1 + u2 > 1.0:
        return -np.inf

    lnprior -= 0.5*(vsini - vsini_mu)**2/vsini_sig**2

    lnprior -= 0.5*(r-rp_mu)**2/rp_sig**2


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
    varystd[9] = 0.6
    varystd[10] = 0.01
    varystd[11] = 0.001
    varystd[12] = 200.0
    varystd[13] = 0.001
    varystd[14] = 0.025
    varystd[15] = 200.0
    varystd[16] = 0.001
    varystd[17] = 0.025


    for i in range(len(varystd)):
        p0[:,i] = p0[:,i] + np.random.normal(0, varystd[i], nwalkers)

    return p0

p0 = setup_data(init)

with Pool(24) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
    pos, prob, state = sampler.run_mcmc(p0, 3000, progress=True)


best = np.where(sampler.flatlnprobability == np.max(sampler.flatlnprobability))[0][0]
print(sampler.flatlnprobability[best])

np.save('../runs/chain_ha_sk_mm_rpprior', sampler.chain)

np.save('pos', pos)
np.save('prob', prob)


