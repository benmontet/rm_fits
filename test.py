from numpy import arange, pi
import matplotlib.pylab as plt

from PyAstronomy import modelSuite as ms

rmcl = ms.RmcL()
rmcl.assignValue({"a":6.7, "lambda":7.2/180.0*pi, "epsilon":0.5, \
                  "P":1.74, "T0":0.2, "i":87.8/180.*pi, \
                  "Is":90.0/180.0*pi, "Omega":1.609e-5, "gamma":0.2})

time = arange(100)/100.0 * 0.2 + 0.1
rv = rmcl.evaluate(time)
plt.ylabel("Radial velocity [stellar-radii/s]")

plt.xlabel("Time [d]")

plt.plot(time, rv, '.')
plt.show()

