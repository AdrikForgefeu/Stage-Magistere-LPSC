import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from astropy.constants import sigma_T, m_e, c, k_B, G, m_p
from astropy.cosmology import Planck18 as cosmo
from scipy.integrate import quad

plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 11, 'axes.grid': True,
                     'grid.alpha': 0.3})

def cmb_temperature_at_z(z=0):
    """
    Dependencies of the CMB temperature with redshift
    
    input z=0 by default
    
    Returns float: cmb_temperature
    """
    T0 = 2.725
    return T0 * (1 + z)


h_planck = const.h.value     # J s (NIST CODATA 2022, exact)
k_B = k_B.value
z = 0.55

nu = np.linspace(30e9, 600e9, 500)       # Hz
x = (h_planck * nu) / (k_B * cmb_temperature_at_z(z))
f_x = x * np.cosh(x/2) / np.sinh(x/2) - 4.0      # = x coth(x/2) - 4

h70 = cosmo.H0.value / 70.0
P0 = 8.403 * h70**-1.5
c500 = 1.177
gamma = 1.0510
alpha = 0.3081
beta = 5.4905
mu, mu_e = 0.59, 1.14

def R500_of_M(M500, z):
    # spherical overdensity radius: M500 = (4/3) pi 500 rho_c(z) R500^3
    rho_c = cosmo.critical_density(z)
    R3 = (3 * M500 / (4 * np.pi * 500 * rho_c)).to(u.Mpc**3)
    return (R3 ** (1/3)).to(u.Mpc)

def P500_of_M(M500, z):
    E = cosmo.efunc(z)
    M14 = (M500 / (3e14 * h70**-1 * u.Msun)).to_value(u.dimensionless_unscaled)
    return 1.65e-3 * E**(8/3) * M14**(2/3) * h70**2 * u.Unit('keV cm-3')

def P_thermal(r, M500, z):
    x = (r / R500_of_M(M500, z)).to_value(u.dimensionless_unscaled)
    p = P0 / ((c500 * x)**gamma * (1 + (c500 * x)**alpha)**((beta - gamma)/alpha))
    return P500_of_M(M500, z) * p

def E(redshift):
    return(np.sqrt(cosmo.Om0*(1+redshift)**3+(1-cosmo.Om0)))

M500 = 5e14 * u.Msun
R500 = R500_of_M(M500, z)
nu_null = nu[np.argmin(np.abs(f_x))] / 1e9

plt.figure()
plt.plot(nu/1e9, f_x)
plt.axhline(0, color='k', lw=0.8)
plt.axvline(nu_null, color='tomato', ls='--', label=f'null at {nu_null:.0f} GHz')
plt.xlabel('frequency [GHz]'); plt.ylabel(r'$f(x)=x\coth(x/2)-4$')
plt.title('tSZ spectral distortion (non-relativistic)')
plt.legend()
plt.savefig('tsz_spectral_function.png', dpi=130)
plt.show()


logM500_arr = np.linspace(11, 15, 5)
M500_arr = (10**logM500_arr) * u.Msun
pref = (sigma_T*1e4 / (m_e * c**2)).to(u.cm**2 / u.keV)   # sigma_T / m_e c^2  [cm^2/keV]
Mpc_in_cm = (1 * u.Mpc).to_value(u.cm)

def y_of_b(b_Mpc, M500):
    # y(b) = (sigma_T/m_e c^2) * integral of electron pressure along the line of sight.
    lmax = (5 * R500).to_value(u.Mpc)
    def integrand_per_cm(l_Mpc):
        r = np.sqrt(b_Mpc**2 + l_Mpc**2) * u.Mpc
        Pe = (mu / mu_e) * P_thermal(r, M500, z)      # keV cm^-3
        return (pref * Pe).to_value(u.cm**-1)            # cm^-1
    val, _ = quad(integrand_per_cm, -lmax, lmax, limit=100)  # cm^-1 integrated over Mpc
    return val * Mpc_in_cm                                    # dimensionless

b_grid = np.linspace(0.02, 3.0, 40)            # Mpc
y_grid = np.array([y_of_b(b, M500) for b in b_grid])
theta = (b_grid * u.Mpc / cosmo.angular_diameter_distance(z)).to_value(u.dimensionless_unscaled)
theta_arcmin = np.degrees(theta) * 60


plt.figure()
plt.semilogy(theta_arcmin, y_grid)
plt.xlabel(r'$\theta$ [arcmin]'); plt.ylabel('Compton $y$')
plt.title(f'tSZ profile, M500={M500:.0e} Msun, z={z}')
plt.savefig('tsz_y_profile.png', dpi=130)
plt.show()

'''
Y = 2 * np.pi * np.trapezoid(y_grid * theta, theta)
plt.figure()
plt.semilogy(theta_arcmin, Y)
plt.xlabel(r'$\theta$ [arcmin]'); plt.ylabel(r'$Y_{500}$')
plt.title(f'Integrated Compton y-parameter Y profile, M500={M500:.0e} $M_☉$, z={z}')
plt.show()
'''

nu_cst = 150e9       # Hz
x = (h_planck * nu_cst) / (k_B * cmb_temperature_at_z(z))
f_x_cst = x * np.cosh(x/2) / np.sinh(x/2) - 4.0

gradient = np.linspace(0, 1, len(M500_arr))
colors = [(gradient[i], gradient[0], gradient[0]) for i in range(len(M500_arr))]

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

ax.set_xlabel(r'$\theta$ [arcmin]')
ax.set_ylabel(r'$T_{tSZ}$ [$\mu$K.arcmin²]')
for mass in M500_arr:
    b_grid = np.linspace(0.02, 3.0, 40)            # Mpc
    y_grid = np.array([y_of_b(b, mass) for b in b_grid])
    theta = (b_grid * u.Mpc / cosmo.angular_diameter_distance(z)).to_value(u.dimensionless_unscaled)
    theta_arcmin = np.degrees(theta) * 60
    R500_kpc = (R500_of_M(mass, z)*1e3).to_value(u.kpc)
    TtSZ = cmb_temperature_at_z(z)*f_x_cst*y_grid
    yint_SZ = TtSZ*0
    for i in range(len(TtSZ)):
        for j in range(i):
            yint_SZ[i] += ((R500_kpc*10**(np.log10(theta[j])+0.05))**2-(R500_kpc*10**(np.log10(theta[j])-0.05))**2)*np.pi*TtSZ[j]
    ax.plot(theta_arcmin, yint_SZ, label=r'$\log{M_{500}/M_☉}$ = ' f'{(np.log10(mass.value)):.2f}', color=colors[int(np.where(M500_arr==mass)[0][0])])
ax.legend()
plt.suptitle(r'tSZ effect evolution with critical mass $M_{500}$ at redshift $z =$' f'{z}' r' and frequency $\nu =$' f'{nu_cst*1e-9} Ghz')
plt.tight_layout()
plt.savefig('temp_tsz_profile.png', dpi=130)
plt.show()