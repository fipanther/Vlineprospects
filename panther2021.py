import os
CWD = os.getcwd()


import numpy as np

import user_functions as usr

#remove any existing logfiles
try:
	os.remove(CWD+'/checksandbalances.log') 
	print("old log file removed successfully")
except:
	print('log file cannot be found or no log file to remove')

print('Please remember to check the log file - checksandbalances.log')

import logging
print('regenerating log file')
logging.basicConfig(filename='checksandbalances.log', encoding='utf-8', level=logging.INFO)


import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["savefig.facecolor"] = "white"

#CONSTANTS
h = 4.135E-15 #eV/Hz
MeV = 624151.
secondsperday = 24*3600.
MeVtoHz = 2.417990504024e+20 #convert MeV to Hz
HztokeV = 4.1357E-18 #convert Hz to keV
c = 3E5 #speed of light in km/s
h_erg =6.626196E-27 #planck constant in erg
dnu = 5.42e+16 #size of the bins in frequency space

#LINES IN MEV FROM NUDAT
Va_lines = [0.983, 1.312] #line energies
Va_strengths = [0.982, 0.9998]
Co_lines = [1.238, 1.037, 0.846]
Co_strengths = [0.6646, 0.1405, 0.999]
Ni_lines = [0.749, 0.811, 1.561]
Ni_strengths = [0.495, 0.86, 0.14]

#Some input values you need:
v_offset = 10000. #average ejecta velocity, defines blueshift of emission line center
v_min = 5000. #Lower limit of line width in velocity space
v_max = 20000. #Upper limit of line width in velocity space

#FUNCTIONS

def center_shift(f0, v):
	"""
	Add a doppler shift to the lab frame line center
		Parameters:
			f0 (float): frequency of line as list
			v (float): velocity for doppler shift in km/s
	"""
 	return [i+i*(v/c) for i in f0]

def define_window(f_center, v_low, v_high):
	"""
	Define the window over which you integrate
		Parameters:
			f_center (float): Define bin center frequency
			v_low (float): minimum velocity of line in km/s
			v_high (float): maximum velocity of line in km/s
	"""
	fmin = f_center-((v_low/c)*f_center)
	fmax = f_center+((v_high/c)*f_center)
	return fmin, fmax

def conv_fnu_to_fph(fnu, dnu, nu, dE):
	"""
	Convert fluxes from optical units to gamma-ray units
	fnu: flux in optical astro units erg/cm^2/s/Hz
	dnu: bin size
	nu: bin center
	E: Bin width in keV

	returns flux in ph/cm^2/s/keV
	"""
	return fnu*dnu/(h_erg*nu*dE)

def broadband(bandwidth, res):
	# calculate how the bandwidth reduces the resolution
    return np.sqrt(bandwidth/res)

def dmax(flux_canonical_1mpc, sensitivity):
    return np.sqrt(flux_canonical_1mpc/sensitivity)


#LOAD MODEL FROM SIM+2012
raw_data = np.loadtxt(CWD+"/gamma_spec-ELDD-L.out")

times=raw_data[0]
times=times[1:]#These are the times at which the spectra are calculated in days
frequencies=raw_data[:,0]
frequencies=frequencies[1:]#multitiply by 1E-20 to make it a sensible number
fluxes=raw_data[1:,1:] #multiply by 1E28 to make this a sensible number

frequencies_orig = raw_data[:,0]
frequencies_orig = frequencies_orig[1:]


#energy of the V lines in Hz
lines = [Va_lines[0]*MeVtoHz, Va_lines[1]*MeVtoHz]

#all the bins are the same size so just pick some
binsize_keV = (frequencies_orig[1]-frequencies_orig[0])*4.1357E-18
logging.info('Spectrum binsize is {} keV'.format(binsize_keV))

# these are the lines we are interested in
shift_10 = center_shift(lines, v_offset)


#needed for plotting things nicely - convert to Hz, 
#calculate doppler shift and then convert to keV
Ni_lines_new = [Ni_lines[0]*MeVtoHz, Ni_lines[1]*MeVtoHz,Ni_lines[2]*MeVtoHz]
shift_10_Ni = center_shift(Ni_lines_new, v_offset)
Co_lines_new = [Co_lines[0]*MeVtoHz, Co_lines[1]*MeVtoHz,Co_lines[2]*MeVtoHz]
shift_10_Co = center_shift(Co_lines_new, v_offset)

#converting to keV 
shift_10_keV = [i*HztokeV for i in shift_10]
shift_10_keV_Ni = [i*HztokeV for i in shift_10_Ni]
shift_10_keV_Co = [i*HztokeV for i in shift_10_Co]


define_window_983_min, define_window_983_max  = define_window(shift_10[0], v_max, v_min)
define_window_1312_min, define_window_1312_max  = define_window(shift_10[1], v_max, v_min)


#convert frequencies to keV
energies = [i*4.1357E-18 for i in frequencies_orig]


time_arr_tmp = [5, 15, 25, 35]
idx_arr_tmp = []
for i in time_arr_tmp:
    a = usr.find_nearest(np.asarray(times), i)[0]
    idx_arr_tmp.append(a)

save_fluxes_tmp = []
for p in idx_arr_tmp:
    flux_test = fluxes[:,p]
    flux_ph = [conv_fnu_to_fph(i, dnu, j, binsize_keV) for i,j in zip(flux_test, frequencies_orig)]
    #normalize these so that plotting is easier by multiplying by 10^5
    flux_ph_tmp = [l*1E5 for l in flux_ph]
    save_fluxes_tmp.append(flux_ph_tmp)

#define the integration bandwidth
bandwidth_983 = (define_window_983_max*HztokeV)-(define_window_983_min*HztokeV)
bandwidth_1312 = (define_window_1312_max*HztokeV)-(define_window_1312_min*HztokeV)

### CALCULATE POSITRON LIGHTCURVE ###
# Positron light curve
min_511 = usr.find_nearest(np.asarray(energies), 500)
max_511 = usr.find_nearest(np.asarray(energies), 540)
min_idx_511, max_idx_511 = min_511[0], max_511[0]
int_energies_511 = energies[min_idx_511:max_idx_511]




#flux in ph/cm^2/s
f_511 = []

for l in range(len(times)):
    flux_i = fluxes[:,l]
    
    flux_ph = [conv_fnu_to_fph(i, 5.42e+16, j, 0.22) for i,j in zip(flux_i, frequencies_orig)]
    
    flux_temp_511 = flux_ph[min_idx_511:max_idx_511]
    
    integ_511 = np.trapz(flux_temp_511, int_energies_511)
    
    f_511.append(integ_511)
fname_posLC_dat = 'pos_LC.dat'
np.savetxt(fname_posLC_dat, np.column_stack([times,f_511]))
logging.info('Positron LC data saved as {}'.format(fname_posLC_dat))

### CALCULATE THE VANADIUM LIGHT CURVES:
min_983 = usr.find_nearest(np.asarray(energies), define_window_983_min*HztokeV)
max_983 = usr.find_nearest(np.asarray(energies), define_window_983_max*HztokeV)
min_idx_983, max_idx_983 = min_983[0], max_983[0]

min_1312 = usr.find_nearest(np.asarray(energies), define_window_1312_min*HztokeV)
max_1312 = usr.find_nearest(np.asarray(energies), define_window_1312_max*HztokeV)
min_idx_1312, max_idx_1312 = min_1312[0], max_1312[0]

int_energies_983 = energies[min_idx_983:max_idx_983]
int_energies_1312 = energies[min_idx_1312:max_idx_1312]


#flux in ph/cm^2/s
f_983 = []
f_1312 = []

for l in range(len(times)):
    flux_i = fluxes[:,l]
    
    flux_ph = [conv_fnu_to_fph(i, 5.42e+16, j, 0.22) for i,j in zip(flux_i, frequencies_orig)]
    
    flux_temp_983 = flux_ph[min_idx_983:max_idx_983]
    flux_temp_1312 = flux_ph[min_idx_1312:max_idx_1312]
    
    integ_983 = np.trapz(flux_temp_983, int_energies_983)
    integ_1312 = np.trapz(flux_temp_1312, int_energies_1312)
    
    f_983.append(integ_983)
    f_1312.append(integ_1312)

#SAVE THE LIGHTCURVES TO DATA FILES
V_lc_fname = 'LC.dat'
np.savetxt('LC.dat', np.column_stack([times,f_983,f_1312]))
logging.info('Saved Vanadium lightcurves out to file {}'.format(V_lc_fname))


### WE WANT TO START AT THE TIME THAT RESULTS IN THE BEST TIME, SO PLOT HOW INTEGRATED FLUX 
### CHANGES AS A FUNCTION OF START TIMES

times_seconds = [i*3600*24 for i in times]

window = [0.25E6, 0.5E6, 1E6, 1.5E6, 3E6]
logging.info('investigating optimal integration time out of {} s observations'.format(window))


test_times = np.arange(45,80)
logging.info('investigating optimal integration start time out of {} to {} days'.format(times_seconds[int(min(test_times))], times_seconds[int(max(test_times))]))


all_1312 = []
for j in window:
    intflux_save_1312 = []
    for i in test_times:
        start_t = int(i)
        t_end = usr.find_nearest(np.asarray(times_seconds), j+times_seconds[start_t])[0]
        window_s = (times_seconds[t_end]-times_seconds[start_t])
        window_shift = [i-(times[start_t]*24*3600) for i in times_seconds[start_t:t_end]]
        f_temp_time_1312 = f_1312[start_t:t_end]
        time_temp_seconds = times_seconds[start_t:t_end]

        int_1312_time = np.trapz(f_temp_time_1312, window_shift)/window_s
        intflux_save_1312.append(int_1312_time)
    all_1312.append(intflux_save_1312)
    
all_983 = []
for j in window:
    intflux_save_983 = []
    for i in test_times:
        start_t = int(i)
        t_end = usr.find_nearest(np.asarray(times_seconds), j+times_seconds[start_t])[0]
        window_s = (times_seconds[t_end]-times_seconds[start_t])
        window_shift = [i-(times[start_t]*24*3600) for i in times_seconds[start_t:t_end]]
        f_temp_time_983 = f_983[start_t:t_end]
        time_temp_seconds = times_seconds[start_t:t_end]

        int_983_time = np.trapz(f_temp_time_983, window_shift)/window_s
        intflux_save_983.append(int_983_time)
    all_983.append(intflux_save_983)

#NOTE THAT IT WAS VERY OBVIOUS THAT 1 - 1.5 S WERE THE OPTIMAL TIMES

logging.info('finding max of 1 Ms for 983...')
intflux_983_1 = all_983[2]
indx_max_1_983 = intflux_983_1.index(max(intflux_983_1))
max_983_1 = intflux_983_1[indx_max_1_983]
logging.info('max flux: {} ph/cm^2/s'.format(max_983_1))
logging.info('idx: {}'.format(indx_max_1_983))
logging.info('Optimal start time: {} days'.format(times[indx_max_1_983+45]))

logging.info('finding max of 1.5 Ms for 983...')
intflux_983_15 = all_983[3]
indx_max_15_983 = intflux_983_15.index(max(intflux_983_15))
max_983_15 = intflux_983_15[indx_max_15_983]
logging.info('max flux: {} ph/cm^2/s'.format(max_983_15))
logging.info('idx: {}'.format(indx_max_15_983))
logging.info('Optimal start time: {} days'.format(times[indx_max_15_983+45]))
    
    
logging.info('find max of 1 Ms for 1312...')
intflux_1312_1 = all_1312[2]
indx_max_1 = intflux_1312_1.index(max(intflux_1312_1))
max_1312_1 = intflux_1312_1[indx_max_1]
logging.info('max flux: {} ph/cm^2/s'.format(max_1312_1))
logging.info('idx: {}'.format(indx_max_1))
logging.info('Optimal start time: {} days'.format(times[indx_max_1+45]))
    
logging.info('find max of 1.5 Ms for 1312...')
intflux_1312_15 = all_1312[3]
indx_max_15 = intflux_1312_15.index(max(intflux_1312_15))
max_1312_15 = intflux_1312_15[indx_max_15]
logging.info('max flux: {} ph/cm^2/s'.format(max_1312_15))
logging.info('idx: {}'.format(indx_max_15))
logging.info('Optimal start time: {} days'.format(times[indx_max_15+45]))

max_fluxes = [max_983_1, max_983_15, max_1312_1, max_1312_15]
energy_saved = [shift_10_keV[0], shift_10_keV[0], shift_10_keV[1], shift_10_keV[1]]
idx_saved = [indx_max_1_983, indx_max_15_983, indx_max_1, indx_max_15]
time_saved = [times[indx_max_1_983+45],times[indx_max_15_983+45],times[indx_max_1+45],times[indx_max_15+45]]
window_saved = [1E6, 1.5E6, 1E6, 1.5E6]


#HOW DOES THE RESOLUTION REDUCE WITH THE BROAD BAND USED FOR THE OBSERVATION
#TAKE THE FWHM OF THE LINES WHICH IS ABOUT 34 KEV
bb_degrade_int_983 = broadband(34,2.2)
bb_degrade_int_1312 = broadband(34,2.2)
bb_degrade_ame_983 = broadband(34,0.01*1E3)
bb_degrade_ame_1312 = broadband(34,0.01*1E3)
bb_degrade_cos_983 = broadband(34,4)
bb_degrade_cos_1312 = broadband(34,4)

INTEGRAL = np.loadtxt(CWD+'/sensitivitycurves/SPILINE.txt')
AMEGO = np.loadtxt(CWD+'/sensitivitycurves/AMEGOLINE.txt')
FERMI = np.loadtxt(CWD+'/sensitivitycurves/FERMI.txt')

AMEGO_1 = usr.find_nearest(AMEGO[:,0], Va_lines[1])#will need to add offset here
INTEGRAL_1 = usr.find_nearest(INTEGRAL[:,0], Va_lines[1])

AMEGO_sens = AMEGO[AMEGO_1[0],1]
INTEGRAL_sens = INTEGRAL[INTEGRAL_1[0], 1]


indx_int_983 = usr.find_nearest(np.asarray(INTEGRAL[:,0]),shift_10_keV[0]/1E3)[0]
indx_ame_983 = usr.find_nearest(np.asarray(AMEGO[:,0]), shift_10_keV[0]/1E3)[0]#shift_10_keV[0])[0]
indx_int_1312 = usr.find_nearest(np.asarray(INTEGRAL[:,0]), shift_10_keV[1]/1E3)[0]
indx_ame_1312 = usr.find_nearest(np.asarray(AMEGO[:,0]), shift_10_keV[1]/1E3)[0]
INT_1_983 = INTEGRAL[indx_int_983,1]
AME_1_983 = AMEGO[indx_ame_983,1]
INT_1_1312 = INTEGRAL[indx_int_983,1]
AME_1_1312 = AMEGO[indx_ame_983,1]
COS_1_983= 1E-5
COS_1_1312= 1E-5


int_1_983_band = INT_1_983*bb_degrade_int_983
int_1_1312_band = INT_1_1312*bb_degrade_int_1312

ame_1_983_band = AME_1_983*bb_degrade_ame_983
ame_1_1312_band = AME_1_1312*bb_degrade_ame_1312

cos_1_983_band = COS_1_983*bb_degrade_cos_983
cos_1_1312_band = COS_1_1312*bb_degrade_cos_1312

int_15_983_band = INT_1_983*bb_degrade_int_983/np.sqrt(1.5)#
int_15_1312_band = INT_1_1312*bb_degrade_int_1312/np.sqrt(1.5)#

ame_15_983_band = AME_1_983*bb_degrade_ame_983/np.sqrt(1.5)#
ame_15_1312_band = AME_1_1312*bb_degrade_ame_1312/np.sqrt(1.5)#

cos_15_983_band = COS_1_983*bb_degrade_cos_983/np.sqrt(1.5)#
cos_15_1312_band = COS_1_1312*bb_degrade_cos_1312/np.sqrt(1.5)#


logging.info('Dmaxes for 983 line saving...')
dmaxes_983 = np.column_stack([[1E6, 1.5E6], 
                 [dmax(max_fluxes[0],int_1_983_band), dmax(max_fluxes[1],int_15_983_band)],
                       [dmax(max_fluxes[0],cos_1_983_band), dmax(max_fluxes[1],cos_15_983_band)],
                [dmax(max_fluxes[0],ame_1_983_band), dmax(max_fluxes[1],ame_15_983_band)],
                ])
fname_dmax_983 = 'dmax_983.dat'
np.savetxt(fname_dmax_983, dmaxes_983)

logging.info('Dmaxes for 1312 line saving...')
fname_dmax_1312 = 'dmax_1312.dat'
dmaxes_1312 = np.column_stack([[1E6, 1.5E6], 
                 [dmax(max_fluxes[2],int_1_1312_band),dmax(max_fluxes[3],int_1_1312_band) ],
                [dmax(max_fluxes[2],cos_1_1312_band),dmax(max_fluxes[3],cos_15_1312_band)],
                [dmax(max_fluxes[2],ame_1_1312_band), dmax(max_fluxes[3],ame_15_1312_band)]
                ])

np.savetxt(fname_dmax_1312, dmaxes_1312)

logging.info('Beginning plotting routines... Please wait... ')
#####################################
#		PLOT THE RAW SPECTRA.       #
#####################################
aprps=dict(facecolor='k',arrowstyle='-')
ymin = 0
ymax = 2.2
xmin = 400
xmax = 1450

fig, axs = plt.subplots(2, 2, sharey = 'row', sharex = 'col',
                        gridspec_kw={'hspace': 0, 'wspace': 0}, figsize = (20,20))



(ax1, ax2), (ax3, ax4) = axs


# FIRST PLOT
ax1.hist(energies, weights = save_fluxes_tmp[0], 
                 bins = energies, color = 'darkslategrey', histtype='step', lw = 3,
         label = 'Radiative Transfer Model', zorder = 5)

ax1.axvspan(define_window_983_min*HztokeV, define_window_983_max*HztokeV, alpha=0.5, color='gainsboro')
ax1.axvspan(define_window_1312_min*HztokeV, define_window_1312_max*HztokeV, 
            alpha=0.5, color='gainsboro', label = 'Integration Band')
ax1.grid(b=True, color='#999999', linestyle='-', alpha=0.2)
ax1.set_xlim([xmin, xmax])
ax1.set_ylim([ymin, ymax])


# SECOND PLOT
ax2.hist(energies, weights = save_fluxes_tmp[1], 
                 bins = energies, color = 'darkslategrey', 
         histtype='step', lw = 3, zorder = 5)
# ax2.plot([shift_10_keV[0]]*100, yarr_new, 'k--', label = 'Line Center')
# ax2.plot([shift_10_keV[1]]*100, yarr_new, 'k--')
ax2.axvspan(define_window_983_min*HztokeV, define_window_983_max*HztokeV, alpha=0.5, color='gainsboro')
ax2.axvspan(define_window_1312_min*HztokeV, define_window_1312_max*HztokeV, 
            alpha=0.5, color='gainsboro', label = 'Integration Band')
ax2.grid(b=True, color='#999999', linestyle='-', alpha=0.2)
ax2.set_xlim([xmin, xmax])
ax2.set_ylim([ymin, ymax])


# THIRD PLOT
ax3.hist(energies, weights = save_fluxes_tmp[2], 
                 bins = energies,color = 'darkslategrey',  histtype='step', lw = 3
        , zorder = 5)
ax3.grid(b=True, color='#999999', linestyle='-', alpha=0.2)
ax3.axvspan(define_window_983_min*HztokeV, define_window_983_max*HztokeV, alpha=0.5, color='gainsboro')
ax3.axvspan(define_window_1312_min*HztokeV, define_window_1312_max*HztokeV, 
            alpha=0.5, color='gainsboro', label = 'Integration Band')
ax3.set_xlim([xmin, xmax])
ax3.set_ylim([ymin, ymax])


# FOURTH PLOT
ax4.hist(energies, weights = save_fluxes_tmp[3], 
                 bins = energies, color = 'darkslategrey', histtype='step', lw = 3, zorder = 5)
ax4.grid(b=True, color='#999999', linestyle='-', alpha=0.2)
ax4.axvspan(define_window_983_min*HztokeV, define_window_983_max*HztokeV, alpha=0.5, color='gainsboro')
ax4.axvspan(define_window_1312_min*HztokeV, define_window_1312_max*HztokeV, 
            alpha=0.5, color='gainsboro', label = 'Integration Band')
ax4.set_xlim([xmin, xmax])
ax4.set_ylim([ymin, ymax])


# SET THE Y-LABELS
ax1.set_ylabel('flux x 10$^{-5}$ ph/cm$^2$/s/keV', fontsize=18)
ax3.set_ylabel('flux x 10$^{-5}$ ph/cm$^2$/s/keV', fontsize=18)

# SET THE X-LABELS
ax3.set_xlabel('energy/keV', fontsize=18)
ax4.set_xlabel('energy/keV', fontsize=18)

# WHERE DO THE ANNOTATIONS FLOAT
yalign_annotate = 1.8
yalign_line = 0.5
annotate_fsize = 16
title_label_align = 2

ax1.text(900, title_label_align, '{:1.0f} days'.format(times[idx_arr_tmp[0]]), ha="center", va="center", rotation=0,
            size=25,
            bbox=None)

#nightmare annotation block
# VANADIUM
ax1.annotate('$^{48}$V (1312 keV)',xy=(shift_10_keV[1], yalign_line),
             xytext=(shift_10_keV[1], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
ax1.annotate('$^{48}$V (983 keV)',xy=(shift_10_keV[0], yalign_line),
             xytext=(shift_10_keV[0], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
#POSITRON LINE
ax1.annotate('e$^{+}$e$^{-}$ (511 keV)',xy=(511, yalign_line),
             xytext=(511, yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
#NICKEL
ax1.annotate('$^{56}$Ni (749 keV)',xy=(shift_10_keV_Ni[0], yalign_line),
             xytext=(shift_10_keV_Ni[0], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
ax1.annotate('$^{56}$Ni (811 keV)',xy=(shift_10_keV_Ni[1], yalign_line),
             xytext=(shift_10_keV_Ni[1], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
ax1.annotate('$^{56}$Ni (1561 keV)',xy=(shift_10_keV_Ni[2], yalign_line),
             xytext=(shift_10_keV_Ni[2], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
#COBALT
ax1.annotate('$^{56}$Co (1238 keV)',xy=(shift_10_keV_Co[0], yalign_line),
             xytext=(shift_10_keV_Co[0], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
ax1.annotate('$^{56}$Co (1837 keV)',xy=(shift_10_keV_Co[1], yalign_line),
             xytext=(shift_10_keV_Co[1], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
ax1.annotate('$^{56}$Co (846 keV)',xy=(shift_10_keV_Co[2], yalign_line),
             xytext=(shift_10_keV_Co[2], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)


#VANADIUM
ax2.annotate('$^{48}$V (1312 keV)',xy=(shift_10_keV[1], yalign_line),
             xytext=(shift_10_keV[1], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
ax2.annotate('$^{48}$V (983 keV)',xy=(shift_10_keV[0], yalign_line),
             xytext=(shift_10_keV[0], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
#POSITRONS
ax2.annotate('e$^{+}$e$^{-}$ (511 keV)',xy=(511, yalign_line+0.4),
             xytext=(511, yalign_annotate+0.3), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
#NICKEL
ax2.annotate('$^{56}$Ni (749 keV)',xy=(shift_10_keV_Ni[0], yalign_line),
             xytext=(shift_10_keV_Ni[0], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
ax2.annotate('$^{56}$Ni (811 keV)',xy=(shift_10_keV_Ni[1], yalign_line),
             xytext=(shift_10_keV_Ni[1], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
ax2.annotate('$^{56}$Ni (1561 keV)',xy=(shift_10_keV_Ni[2], yalign_line),
             xytext=(shift_10_keV_Ni[2], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
#COBALT
ax2.annotate('$^{56}$Co (1238 keV)',xy=(shift_10_keV_Co[0], yalign_line),
             xytext=(shift_10_keV_Co[0], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
ax2.annotate('$^{56}$Co (1037 keV)',xy=(shift_10_keV_Co[1], yalign_line),
             xytext=(shift_10_keV_Co[1], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
ax2.annotate('$^{56}$Co (846 keV)',xy=(shift_10_keV_Co[2], yalign_line),
             xytext=(shift_10_keV_Co[2], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)


#VANADIUM
ax3.annotate('$^{48}$V (1312 keV)',xy=(shift_10_keV[1], yalign_line),
             xytext=(shift_10_keV[1], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
ax3.annotate('$^{48}$V (983 keV)',xy=(shift_10_keV[0], yalign_line),
             xytext=(shift_10_keV[0], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
# POSITRONS
ax3.annotate('e$^{+}$e$^{-}$ (511 keV)',xy=(511, yalign_line+0.4),
             xytext=(511, yalign_annotate+0.3), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
# NICKEL
ax3.annotate('$^{56}$Ni (749 keV)',xy=(shift_10_keV_Ni[0], yalign_line),
             xytext=(shift_10_keV_Ni[0], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
ax3.annotate('$^{56}$Ni (811 keV)',xy=(shift_10_keV_Ni[1], yalign_line),
             xytext=(shift_10_keV_Ni[1], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = 14, arrowprops=aprps)
ax3.annotate('$^{56}$Ni (1561 keV)',xy=(shift_10_keV_Ni[2], yalign_line),
             xytext=(shift_10_keV_Ni[2], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
# COBALT
ax3.annotate('$^{56}$Co (1238 keV)',xy=(shift_10_keV_Co[0], yalign_line),
             xytext=(shift_10_keV_Co[0], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
ax3.annotate('$^{56}$Co (1037 keV)',xy=(shift_10_keV_Co[1], yalign_line),
             xytext=(shift_10_keV_Co[1], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
ax3.annotate('$^{56}$Co (846 keV)',xy=(shift_10_keV_Co[2], yalign_line),
             xytext=(shift_10_keV_Co[2], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)

# VANADIUM
ax4.annotate('$^{48}$V (1312 keV)',xy=(shift_10_keV[1], yalign_line),
             xytext=(shift_10_keV[1], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
ax4.annotate('$^{48}$V (983 keV)',xy=(shift_10_keV[0], yalign_line),
             xytext=(shift_10_keV[0], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
# POSITRONS
ax4.annotate('e$^{+}$e$^{-}$ (511 keV)',xy=(511, yalign_line+0.4),
             xytext=(511, yalign_annotate+0.3), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
# NICKEL
ax4.annotate('$^{56}$Ni (749 keV)',xy=(shift_10_keV_Ni[0], yalign_line),
             xytext=(shift_10_keV_Ni[0], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
ax4.annotate('$^{56}$Ni (811 keV)',xy=(shift_10_keV_Ni[1], yalign_line),
             xytext=(shift_10_keV_Ni[1], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
ax4.annotate('$^{56}$Ni (1561 keV)',xy=(shift_10_keV_Ni[2], yalign_line),
             xytext=(shift_10_keV_Ni[2], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
# COBALT
ax4.annotate('$^{56}$Co (1238 keV)',xy=(shift_10_keV_Co[0], yalign_line),
             xytext=(shift_10_keV_Co[0], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
ax4.annotate('$^{56}$Co (1037 keV)',xy=(shift_10_keV_Co[1], yalign_line),
             xytext=(shift_10_keV_Co[1], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)
ax4.annotate('$^{56}$Co (846 keV)',xy=(shift_10_keV_Co[2], yalign_line),
             xytext=(shift_10_keV_Co[2], yalign_annotate), 
             ha='center', rotation='vertical',va='top', fontsize = annotate_fsize, arrowprops=aprps)



ax2.text(900, title_label_align, '{:1.0f} days'.format(times[idx_arr_tmp[1]]), ha="center", va="center", rotation=0,
            size=25,
            bbox=None)
ax3.text(900, title_label_align, '{:1.0f} days'.format(times[idx_arr_tmp[2]]), ha="center", va="center", rotation=0,
            size=25,
            bbox=None)
ax4.text(900, title_label_align, '{:1.0f} days'.format(times[idx_arr_tmp[3]]), ha="center", va="center", rotation=0,
            size=25,
            bbox=None)



ax1.tick_params(
        labelsize = 20,
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        #bottom=False,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        #labelbottom=False)
)
ax2.tick_params(
        labelsize = 20,
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        #bottom=False,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        #labelbottom=False)
)
ax3.tick_params(
        labelsize = 20,
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        #bottom=False,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        #labelbottom=False)
)
ax4.tick_params(
        labelsize = 20,
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        #bottom=False,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        #labelbottom=False)
)

ax3.set_xticks([400, 600, 800, 1000, 1200, 1400])
ax4.set_xticks([600, 800, 1000, 1200, 1400])
ax1.set_yticks([0.5, 1, 1.5, 2])
ax3.set_yticks([0.5, 1, 1.5, 2])

fname_spec = 'specevonew.eps'
plt.savefig(fname_spec)
logging.info('raw spectra saved as {}'.format(fname_spec))



#####################################
#.    PLOT POSITRON LIGHT CURVE     #
#####################################
plt.figure(figsize = (7,7))
plt.plot(times, f_511, 'k+', label = '511 keV line 500 - 540 keV')
plt.xlabel('time/days', fontsize = 20)
plt.ylabel('flux/$\,ph\,cm^{-2}\,s^{-1}$', fontsize = 20)
plt.legend(loc = 'best', fontsize = 14)
plt.grid(b=True, color='#999999', linestyle='-', alpha=0.2)
plt.tick_params(
        labelsize = 18,
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        #bottom=False,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        #labelbottom=False)
)


fname_posLC = 'poslightcurve.png'
plt.savefig(fname_posLC)
logging.info('Positron LC saved as {}'.format(fname_posLC))




#####################################
#.    PLOT POSITRON LIGHT CURVE     #
#####################################
plt.figure(figsize = (7,7))
plt.plot(times, [i/1E-4 for i in f_983], 'k+', label = '983 keV window')
plt.plot(times, [i/1E-4 for i in f_1312], 'r+', label = '1312 keV window')
plt.xlabel('time/days', fontsize = 20)
plt.ylabel('flux x $10^{-4}/\,ph\,cm^{-2}\,s^{-1}$', fontsize = 20)
plt.legend(loc = 'best', fontsize = 14)
plt.grid(b=True, color='#999999', linestyle='-', alpha=0.2)
plt.tick_params(
        labelsize = 18,
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        #bottom=False,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        #labelbottom=False)
)
fname_VLC = 'Vlightcurve.eps'
plt.savefig(fname_VLC)
logging.info('Vanadium LC data saved as {}'.format(fname_VLC))




#####################################
#.    OPTIMIZE WHEN TO START INT    #
#####################################
plt.figure(figsize = (7,7))
# plt.plot(times[45:80], intflux_save_983, '+k')
plt.plot(times[45:80], [i*1E4 for i in all_1312[0]], 'k', label = '0.25 Ms integration')
plt.plot(times[45:80], [i*1E4 for i in all_1312[1]], 'k--', label = '0.5 Ms integration')
plt.plot(times[45:80], [i*1E4 for i in all_1312[2]], 'k:', label = '1 Ms integration')
plt.plot(times[45:80], [i*1E4 for i in all_1312[3]], 'k', linestyle = 'dashdot', label = '1.5 Ms integration')
plt.plot(times[45:80], [i*1E4 for i in all_1312[4]], 'k', linestyle = (0, (5, 10)), label = '3 Ms integration')
#plt.plot(times[45+indx_max], intflux_save_983[indx_max], 'o')
plt.xlabel('Integration start time/days', fontsize = 18)
plt.ylabel(r'Integrated flux at 1 Mpc x 10$^{-4}$/ph/cm$^{2}$/s$^{1}$', fontsize = 18)
plt.legend(loc='best', fontsize = 14)
#plt.yscale('log')
plt.xlim([3.5,19])
plt.tick_params(
        labelsize = 18,
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        #bottom=False,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        #labelbottom=False)
)
plt.grid(b=True, color='#999999', linestyle='-', alpha=0.2)
fname_optimizer = 'inttimeoptimize.eps'
plt.savefig(fname_optimizer)
logging.info('Optimized integration start plot saved as {}'.format(fname_optimizer))

logging.info('Analysis complete! Have a nice day')