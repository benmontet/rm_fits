import numpy as np
import matplotlib.pyplot as plt
import starry

import exoplanet as exo
import emcee


from multiprocessing import Pool


time, vels, verr = np.loadtxt('../data/transit.vels', usecols=[0,1,2], unpack=True)
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

bnds = ((12000, 24000), (0.04, 0.07), (-1.0, 0.0), (15,25), (0,1),(0,1), (0,90), (-20,20),
        (50,300), (0.0, 2.0), (2.0, 20.0), (0.0, 1.0), (0.12, 0.20))


vsini_mu = 18.3
vsini_sig = 1.8

ndim = 13
nwalkers = 30

init = np.array([18300.012920585246, 0.058325908653385085, -0.17989438892573809, 20.56513997921657, 1.0, 0.0, 17.311534969903935,
        -6.657428484257285, 185.09863367853995, 1.2029170041130754, 5.885716645051851, 0.01, 0.165])


def rmcurve(params):
    vsini, r, b, a, u1, u2, obl, gamma, gammadot, jitter_good, jitter_bad, q, t0 = params
    veq = vsini / np.sin(inc * np.pi / 180.0)


    map.inc = inc
    map.obl = obl
    # map.add_spot(spot_amp, sigma=spot_sig, lon=spot_lon, lat=-spot_lat)
    map[1:] = [u1, u2]
    map.veq = veq

    #orbit = exo.orbits.KeplerianOrbit(period=P, a=a, t0=t0, b=b, ecc=e, omega=w, r_star=1.0)

    #x, y, z = orbit.get_relative_position(tuse)

    #xo = x.eval()
    #yo = y.eval()
    #33zo = z.eval()
    #theta = 360.0 / Prot * tuse

    xo = np.ones_like(tuse)
    yo = np.ones_like(tuse)
    zo = np.ones_like(tuse)*10.0
    theta = np.ones_like(tuse)

    rv = map.rv(xo=xo, yo=yo, zo=zo, ro=r, theta=theta)
    rv += gamma + gammadot * (tuse - 0.15)

    var_good = (euse ** 2 + jitter_good ** 2)
    var_bad = (euse ** 2 + jitter_bad ** 2)
    gooddata = -0.5 * q * (np.sum((rv - vuse) ** 2 / var_good + np.log(2 * np.pi * var_good)))
    baddata = -0.5 * (1 - q) * (np.sum((rv - vuse) ** 2 / var_bad + np.log(2 * np.pi * var_bad)))
    lnprob = gooddata + baddata

    #print(-1 * lnprob)
    return lnprob

def set_priors(params):
    vsini, r, b, a, u1, u2, obl, gamma, gammadot, jitter_good, jitter_bad, q, t0 = params
    lnprior = 0

    if not all(b[0] < v < b[1] for v, b in zip(params, bnds)):
        return -np.inf
    lnprior -= 0.5*(vsini - vsini_mu)**2/vsini_sig**2

    return lnprior



def lnprob(params):
    lnprior = set_priors(params)
    if np.isfinite(lnprior) == False:
        return -np.inf
    else:
        lnlike = rmcurve(params)
        return lnlike + lnprior

def setup_data(params):
    init = np.zeros(ndim)
    p0 = np.tile(params, nwalkers)
    p0 = p0.reshape(nwalkers, ndim)

    varystd = np.ones(ndim)*0.2
    varystd[0] = 100.0
    varystd[1] = 0.01
    varystd[12] = 0.01
    varystd[7] = 3.0
    varystd[8] = 3.0
    varystd[11] = 0.001

    for i in range(len(varystd)):
        p0[:,i] = p0[:,i] + np.random.normal(0, varystd[i], nwalkers)

    return p0

p0 = setup_data(init)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
pos, prob, state = sampler.run_mcmc(p0, 2000, progress=True)

best = np.where(sampler.flatlnprobability == np.max(sampler.flatlnprobability))[0][0]
print(sampler.flatlnprobability[best])



