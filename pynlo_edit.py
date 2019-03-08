import numpy as np
import matplotlib.pyplot as plt
import pynlo
## assumption that betas, gamma and alpha are calculated at the central frequencty only
gas_type = {"Ar":9.8e-24,"N2":7.9e-24,"Kr":0}
def beta():
    return beta2,beta3,beta4
def gamma(rad,pressure,gas):
    n2 = gas_type[str(gas)]*pressure
    w0 = 2*np.pi*3e8/(800e-9)
    Aeff = 1/2*np.pi*rad**2
    gamma = n2*w0/(3e8*Aeff)
    return gamma
def alpha(inner_radius,lamb0,ref_index_ext,ref_index_gas):
    new = ref_index_ext/ref_index_gas
    alpha = (2.405/(2*np.pi))**2*lamb0**2/(inner_radius**3)*(new**2+1)/(np.sqrt(new**2-1))
    return alpha
cent_lamb = 800e-9
ref_index_glass = 1.5
ref_index_gas = {"Ar":1.0028,"N2":1,"Kr":1}
pulseWL = 800   # pulse central wavelength (nm)
print("The following variables have been fixed already.")
print("The central wavelength is 800nm, the outer material of the fiber is glass with refractive index 1.5.\n The input pulse has no group delay or higher dispersion in it. The other parameters will be calculated according the following inputs from the user.")
print("You can use default parameters to simulate also. To simulate using default parameters enter 'y' else press anything else.")
g = input()
if(g=='y'):
    gas ="Ar"
    p=1
    l=1
    r=125e-6
    fiber_radius = r
    FWHM = 0.050
    EPP = 400e-6
else:
    print("Enter gas type(Ar/N2/Kr):")
    gas = input()
    print("Enter gas pressure(in atm):")
    p = float(input())
    print("Enter fiber length (in m) :")
    l = float(input())
    print("Enter inner diameter of fiber(in micrometer): ")
    d = float(input())
    r = d/2*10**-6
    fiber_radius = r
    print("Enter FWHM time of input pulse(in fs):")
    FWHM    = float(input())/1000  # pulse duration (ps)
    print("Enter energy per pulse(in micro joule):")
    EPP     = float(input())*10**-6 # Energy per pulse (J)
GDD     = 0.0    # Group delay dispersion (ps^2)
TOD     = 0.0    # Third order dispersion (ps^3)

Window  = 10#10.0   # simulation window (ps)
Steps   = 100     # simulation steps
Points  = 2**13  # simulation points

beta2   =  -9.127e-9#-120     # (ps^2/km)
beta3   =  1.19053e-11    # (ps^3/km)
beta4   =  -2.07049e-14#0.005    # (ps^4/km)

Length  = l*1000   # length in mm

##Alpha   = 0.139568*4.34/100     # attentuation coefficient (dB/cm)
Gamma   = gamma(r,p,gas)*1000 #3.28581e-6  # Gamma (1/(W km)

fibWL   = pulseWL # Center WL of fiber (nm)

Raman   = True    # Enable Raman effect?
Steep   = True    # Enable self steepening?

alpha = alpha(fiber_radius,cent_lamb,ref_index_glass,ref_index_gas[str(gas)])#np.log((10**(Alpha * 0.1))) * 100  # convert from dB/cm to 1/m

######## This is where the PyNLO magic happens! ############################

# create the pulse!
pulse = pynlo.light.DerivedPulses.SechPulse(1, FWHM/1.76, pulseWL, time_window_ps=Window,
                  GDD=GDD, TOD=TOD, NPTS=Points, frep_MHz=100, power_is_avg=False)
pulse.set_epp(EPP) # set the pulse energy

# create the fiber!
fiber1 = pynlo.media.fibers.fiber.FiberInstance()
fiber1.generate_fiber(Length * 1e-3, center_wl_nm=fibWL, betas=(beta2, beta3, beta4),
                              gamma_W_m=Gamma * 1e-3, gvd_units='ps^n/km', gain=-alpha)

# Propagation
evol = pynlo.interactions.FourWaveMixing.SSFM.SSFM(local_error=0.001, USE_SIMPLE_RAMAN=True,
                 disable_Raman=np.logical_not(Raman),
                 disable_self_steepening=np.logical_not(Steep))

y, AW, AT, pulse_out = evol.propagate(pulse_in=pulse, fiber=fiber1, n_steps=Steps)

########## That's it! Physic done. Just boring plots from here! ################

F = pulse.W_mks

zW = ( np.transpose(AW) )
zT = ( np.transpose(AT) )
c=3e8
y = y * 1e3 # convert distance to mm
lamb = abs(2*np.pi*3e8/F)

plt.subplot(121)

plt.plot(lamb,abs(zW[0]),label="z=0m")
plt.plot(lamb,abs(zW[-1]),label="z=1m")
plt.xlim(700e-9,1000e-9)
plt.legend()
plt.xlabel("Frequency")
plt.ylabel("Intensity(a.u.)")
plt.title("freq.")
ts = pulse.T_ps*10e-13
plt.subplot(122)

tnew = np.fft.fftshift(np.fft.fftfreq(Points,d=F[1]-F[0]))
spect = np.fft.fftshift(np.fft.fft(abs(zW[-1])))
##plt.imshow(abs(spect))
plt.plot(ts,abs(zT[0]),label="z=0m")
plt.plot(ts,abs(spect),label="z=1m")
plt.legend()
plt.title("time")
plt.xlabel("s")
plt.ylabel("Intensity")
print("FWHM of input pulse is:")
print(2*abs(ts[np.where(max(zT[0])/2)]))
print("FWHM of output pulse is:")
print(2*abs(ts[np.where(max(spect)/2)]))

plt.show()
