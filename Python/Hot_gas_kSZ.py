import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import quad
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP7
from astropy.cosmology import Planck18 as planck
from astropy import constants as const
from astropy import units as u
from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology('planck18', persistence = 'r')
from colossus.lss import peaks
from colossus.halo import mass_defs, mass_so

M_sun = float(const.M_sun/(u.kg))
pc = float(const.pc/(u.m))
c = float(const.c/(u.m/u.s))
sigma_T = float(const.sigma_T/(u.m**2))
uma = float(const.u/(u.kg))
h0 = cosmo.H0/100

z = 0.2
log10Mc_arr = np.linspace(11, 15, 5)
Mc_arr = (10**log10Mc_arr)*u.solMass.to(u.kg)
Mc = float((10**13)*u.solMass.to(u.kg))
logM200_arr = np.linspace(11, 15, 5)
M200_arr = (10**logM200_arr)*u.solMass.to(u.kg)
M200 = (10**logM200_arr[2])*u.solMass.to(u.kg)
mu_arr = np.linspace(0, 2, 3)
mu = 1
delta_arr = np.linspace(1, 11, 11)
delta = 5
gamma = 1.0510
theta_c = 0.3
alpha = 0.3081
rho_0 = 1e13*M_sun/(1e6*pc)**3 #kg/m³
vp = 1.06e-3*const.c #m/s
r0 = 0.0
r1 = float(1e6*u.parsec.to(u.m))

def nu(M200):
    """
    Compute the peak height of the halo

    input virial mass : M200 float or array-like
    
    Returns float or array-like : nu
    """
    k = 10**np.linspace(-5.0, 2.0, 500)
    #Pk = cosmo.matterPowerSpectrum(k)
    #sigma_tophat = cosmo.sigma(r200(M200), z=0.0, ps_args=Pk)
    return peaks.peakHeight(M200/M_sun, z=z)

def eps(M200):
    """
    Compute the truncation radius parameter

    input float or array-like : M200

    Returns float or array-like : eps
    """
    return 4+0.5*nu(M200)

def beta(M200, Mc, mu):
    """
    Compute the slope of the mass density profile
    Parameters
    ----------
    virial mass : M200 float
    pivot mass : Mc float
    transition steepness : mu float
    
    Returns float : beta
    """
    return (3*(M200/Mc)**mu/(1+(M200/Mc)**mu))

def r200(M200):
    rho_c = float(WMAP7.critical_density(0).to(u.kg/u.m**3)/(u.kg/u.m**3))
    return (3*M200/(4*np.pi*200*rho_c))**(1/3)  #m

def R200c_col(M200):
    M200c_h = M200*h0/M_sun
    R200c_h = mass_so.M_to_R(M200c_h, z, '200c')
    R200c = R200c_h/h0
    return R200c*pc*1e3

def R200_kpc(M200):
    return R200c_col(M200)/(pc*1e3)

def R200_Mpc(M200):
    return R200c_col(M200)/(pc*1e6)

def r200vir(M200c):
    Mv, Rv, cv = mass_defs.changeMassDefinition(M200c, 1, z, '200c', 'vir')
    return Rv

def gas_density(r, M200, Mc, mu, delta):
    """
    Calculation of the gas mass density according a truncated NFW profile
    
    Parameters
    ----------
    distance to halo center: r float or array-like
    virial mass : M200 float
    pivot mass : Mc float
    transition steepness : mu float
    peak height of the halo : nu float
    outer slope : delta float
    
    Returns float or array-like : gas mass density rho_g
    """
    rho_g = rho_0*(1+(r/(theta_c*R200c_col(M200)))**alpha)**(-beta(M200, Mc, mu)/alpha)*(1+(r/(eps(M200)*R200c_col(M200)))**gamma)**(-delta/gamma)  #kg.m⁻³
    return rho_g

def e_density(r, M200, Mc, mu, delta):
    return (1.76/(2*uma))*gas_density(r, M200, Mc, mu, delta)
  
def opt_depth(M200, Mc, mu, delta, b):
    """
    Compute optical depth by integrating electron density

    Parameters
    ----------
    virial mass : M200 float
    pivot mass : Mc float
    transition steepness : mu float
    peak height of the halo : nu float
    outer slope : delta float
    projected radius : b array-like

    Returns float tau
    """
    lmax = (5 * R200c_col(M200))  #m
    def integrand_per_m(l):
        r = np.sqrt(b**2 + l**2) #m
        n_e = e_density(r, M200, Mc, mu, delta)  #m⁻³
        return n_e
    val, _ = quad(integrand_per_m, -lmax, lmax)
    return sigma_T/(1+z)*val

def cmb_temperature_at_z(z=0):
    """
    Dependencies of the CMB temperature with redshift
    
    input z=0 by default
    
    Returns float: cmb_temperature
    """
    T0 = 2.725
    return T0 * (1 + z)

def kSZ(vp, M200, Mc, mu, delta, b):
    """
    Compute kSZ effect in assuming the peculiar velocity to be factored out of the line of sight integral

    Parameters
    ----------
    peculiar velocity : vp float
    virial mass : M200 float
    pivot mass : Mc float
    transition steepness : mu float
    peak height of the halo : nu float
    outer slope : delta float
    projected radius : b array-like

    Returns T_kSZ
    """
    return cmb_temperature_at_z(z)*vp/const.c*opt_depth(M200, Mc, mu, delta, b)

b_grid = np.linspace(0.02, 3.0, 40)*1e6*pc  #m
theta = (b_grid/(cosmo.angularDiameterDistance(z=z)*1e6*pc))
theta_arcmin = np.degrees(theta) * 60

gradient = np.linspace(0, 1, len(M200_arr))
colors = [(gradient[i], gradient[0], gradient[0]) for i in range(len(M200_arr))]



fig, axes = plt.subplots(1, 4, figsize=(25, 125/12))

ax = axes[0]
ax.set_xlabel(r'$\theta$ [arcmin]')
ax.set_ylabel(r'$T_{kSZ}$ [$\mu$K.arcmin²]')
ax.set_title(r'Virial mass $M_{200}$' '\n' 
            rf'Fixed parameters :' '\n' r'$\log{M_c/M_☉} =$' f'{np.log10(Mc/M_sun)}' '\n' rf'$\mu =$ {mu}' '\n' rf'$\delta=$ {delta}')
#ax.set_xlim
#ax.set_ylim
for mass in M200_arr:
    TkSZ = np.array([kSZ(vp, mass, Mc, mu, delta, b)*1e6 for b in b_grid])
    tauint_SZ = TkSZ*0
    for i in range(len(TkSZ)):
        for j in range(i):
            tauint_SZ[i] += ((R200_kpc(mass)*10**(np.log10(theta[j])+0.05))**2-(R200_kpc(mass)*10**(np.log10(theta[j])-0.05))**2)*np.pi*TkSZ[j]
    ax.semilogy(theta_arcmin, tauint_SZ, label=r'$\log{M_{200}/M_☉}$ = ' f'{np.log10(mass/M_sun)}', color=colors[int(np.where(M200_arr==mass)[0][0])])
ax.legend()

ax = axes[1]
gradient = np.linspace(0, 1, len(Mc_arr))
colors = [(gradient[i], gradient[0], gradient[0]) for i in range(len(Mc_arr))]
ax.set_xlabel(r'$\theta$ [arcmin]')
ax.set_title(rf'Pivot mass $M_c$''\n'
            rf'Fixed parameters :' '\n' r'$\log{M_{200}/M_☉} =$' f'{np.log10(M200/M_sun)}' '\n' rf'$\mu =$ {mu}' '\n' rf'$\delta=$ {delta}')
#ax.set_xlim
#ax.set_ylim
for mass in Mc_arr:
    TkSZ = np.array([kSZ(vp, M200, mass, mu, delta, b)*1e6 for b in b_grid])
    tauint_SZ = TkSZ*0
    for i in range(len(TkSZ)):
        for j in range(i):
            tauint_SZ[i] += ((R200_kpc(M200)*10**(np.log10(theta[j])+0.05))**2-(R200_kpc(M200)*10**(np.log10(theta[j])-0.05))**2)*np.pi*TkSZ[j]
    ax.semilogy(theta_arcmin, tauint_SZ, label=r'$\log{M_c/M_☉}$ = ' f'{np.log10(mass/M_sun)}', color=colors[int(np.where(Mc_arr==mass)[0][0])])
ax.legend()

ax = axes[2]
gradient = np.linspace(0, 1, len(mu_arr))
colors = [(gradient[i], gradient[0], gradient[0]) for i in range(len(mu_arr))]
ax.set_xlabel(r'$\theta$ [arcmin]')
ax.set_title(rf'Steepness transition $\mu$''\n'
            rf'Fixed parameters :' '\n' r'$\log{M_{200}/M_☉} =$' f'{np.log10(M200/M_sun)}' '\n' r'$\log{M_c/M_☉} =$' f'{np.log10(Mc/M_sun)}' '\n' rf'$\delta=$ {delta}')
#ax.set_xlim
#ax.set_ylim
for steep in mu_arr:
    TkSZ = np.array([kSZ(vp, M200, Mc, steep, delta, b)*1e6 for b in b_grid])
    tauint_SZ = TkSZ*0
    for i in range(len(TkSZ)):
        for j in range(i):
            tauint_SZ[i] += ((R200_kpc(M200)*10**(np.log10(theta[j])+0.05))**2-(R200_kpc(M200)*10**(np.log10(theta[j])-0.05))**2)*np.pi*TkSZ[j]
    ax.semilogy(theta_arcmin, tauint_SZ, label=r'$\mu$ = ' f'{steep}', color=colors[int(np.where(mu_arr==steep)[0][0])])
ax.legend()

ax = axes[3]
gradient = np.linspace(0, 1, len(delta_arr))
colors = [(gradient[i], gradient[0], gradient[0]) for i in range(len(delta_arr))]
ax.set_xlabel(r'$\theta$ [arcmin]')
ax.set_title(rf'Outer slope $\delta$''\n'
            rf'Fixed parameters :' '\n' r'$\log{M_{200}/M_☉} =$' f'{np.log10(M200/M_sun)}' '\n' r'$\log{M_c/M_☉} =$' f'{np.log10(Mc/M_sun)}' '\n' rf'$\mu=$ {mu}')
#ax.set_xlim
#ax.set_ylim
for slope in delta_arr:
    TkSZ = np.array([kSZ(vp, M200, Mc, mu, slope, b)*1e6 for b in b_grid])
    tauint_SZ = TkSZ*0
    for i in range(len(TkSZ)):
        for j in range(i):
            tauint_SZ[i] += ((R200_kpc(M200)*10**(np.log10(theta[j])+0.05))**2-(R200_kpc(M200)*10**(np.log10(theta[j])-0.05))**2)*np.pi*TkSZ[j]
    ax.semilogy(theta_arcmin, tauint_SZ, label=r'$\delta$ = ' f'{slope}', color=colors[int(np.where(delta_arr==slope)[0][0])])
ax.legend()

plt.suptitle(r'kSZ effect evolution with angular line of sight and other parameters at redshift $z =$' f'{z}')
plt.tight_layout()
plt.savefig('temp_ksz_profile.png', dpi=130)
plt.show()




