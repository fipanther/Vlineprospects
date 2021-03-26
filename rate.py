
### Copyright (C) 2021 Fiona Panther <fiona.panther@uwa.edu.au
### This library is free software; you can redistribute it and/or
### modify it under the terms of the GNU Library General Public
### License as published by the Free Software Foundation; either
### version 2 of the License, or (at your option) any later version.
### 
### This library is distributed in the hope that it will be useful,
### but WITHOUT ANY WARRANTY; without even the implied warranty of
### MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
### Library General Public License for more deroll-offss.
###
### You should have received a copy of the GNU Library General Public
### License along with this library; if not, write to the
### Free Software Foundation, Inc., 59 Temple Place - Suite 330,
### Boston, MA 02111-1307, USA.
### 


import os
import numpy as np
import matplotlib.pyplot as plt

CWD = os.getcwd()
mk11 = np.loadtxt(CWD+'/mk11.txt')
rho_crit = 1.46E11 #solar mass per Mpc
density = [i * rho_crit for i in mk11[:,1]]
volume = [(4./3.)*np.pi*i**3 for i in mk11[:,0]]

mass = []
for i in range(len(density)):
    mass.append(np.trapz(density[0:i+1], volume[0:i+1]))

#numbers via Ashley Ruiter
C_factor = 1.63E8
Nsys = 2942.
th = 1.2E10

dmax_ame_15 = 6.96
dmax_cosi_15 = 4.08
dmax_int_15 = 2.33

ys = np.linspace(1E-5,10, 100)

def conv_to_rate(mass):
    rate = []
    for i in mass:
        nstars = i/C_factor
        nsys_th = Nsys*nstars
        rate.append(nsys_th/th)
    return rate
rate = conv_to_rate(mass)

plt.figure(figsize = (7,7))
plt.scatter(mk11[:,0], [i*10 for i in rate], c = 'k', s = 50, marker = '+')

plt.plot([dmax_ame_15]*100, ys, 'k', label = 'AMEGO $d_\mathrm{max}$')
plt.plot([dmax_cosi_15]*100, ys, 'k--',label = 'COSI $d_\mathrm{max}$')
plt.plot([dmax_int_15]*100, ys, 'k:',label = 'INTEGRAL $d_\mathrm{max}$')

plt.plot(np.linspace(0, 30, 10), [0.1]*10, 'k', alpha = 0.5)
plt.text(22, 0.13, '10% chance per 10 yr', ha="center", va="center", rotation=0,
            size=14,
            bbox=None)

plt.xlabel('Radius/Mpc', fontsize = 18)
plt.ylabel('Rate lower limit per 10 yr', fontsize = 18)
plt.yscale('log')
plt.tick_params(
        labelsize = 18,
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        #bottom=False,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        #labelbottom=False)
)
plt.legend(loc = 'best', fontsize = 14)
plt.xlim([0, 30])
plt.grid(b=True, color='#999999', linestyle='-', alpha=0.2)
plt.savefig('rate.eps')
