import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate
from scipy import special


def reduction_factor_gravel(h, k):
    """
    Foot resistance reduction factor of gravel.

    Computing the reduction factor of the gravel layer, when
    determining the foot resistance. This is the simplified
    approach, where mutual resistance between two feet is not
    taken into account.

    Arguments
    ---------
    h : float
        Depth of the gravel layer, m.
    k : reflection factor between gravel layer and soil.

    Returns
    -------
    cs : float
        Reduction factor.
    """
    suma = 0.
    n = 1
    while n < 10_000:
        suma += k**n / np.sqrt(1 + ((2*n*h)/0.08)**2)
        n += 1

    cs = (1/0.96) * (1 + 2*suma)

    return cs


def foot_resistance(mu_rho, sigma_rho, model='plate', **kwargs):
    """
    Computing foot resistance for the touch voltage.

    Arguments
    ---------
    mu_rho : float
        Median value of the soil resistivity (Ohm*m).
    sigma_rho : float
        Standard deviation of the soil resistivity.
    model : str, default='plate'
        Model of the footing for the resistance calculation.
    kwargs : dict
        Additional parameters for the model 'gravel', which include
        'h': thickness of the gravel layer, m and 'g': resistivity.
    
    Returns
    -------
    mu_Rg, sigma_Rg : float, float
        Median value and st. dev. of footing resistance (Ohm).
    """
    if model == 'none':
        # Foot resistance is equal to the soil resistance.
        factor = 1.
    
    elif model == 'plate':
        # Single foot as a circular disc of 200 cm^2.
        factor = 3.

    elif model == 'feet':
        # Parallel connection of two feet, where each foot
        # is modelled as a circular disc.
        factor = 1.5
    
    elif model == 'gravel':
        # Soil is covered by a layer of gravel having (usually)
        # higher resistivity (h: layer depth, g: gravel resistivity).
        h = kwargs['h']
        rho_gravel = kwargs['g']
        rho = mu_rho
        k = (rho - rho_gravel) / (rho + rho_gravel)
        cs = reduction_factor_gravel(h, k)
        factor = 1.5 * cs * (rho_gravel/rho)
    
    else:
        raise NotImplementedError(f'Model {model} is not recognized!')
    
    mu_Rg = factor * mu_rho
    sigma_Rg = factor * sigma_rho

    return mu_Rg, sigma_Rg


def touch_voltage_tolerable_pdf(x, mu_Rk, sigma_Rk, mu_rho, sigma_rho, ti, Pi,
                                kwargs):
    """
    Probability density function of tolerable touch voltage.

    Arguments
    ---------
    x : float
        Voltage value.
    mu_Rk, sigma_Rk : float, float
        Median value and standard deviation of body resistance.
    mu_rho, sigma_rho : float, float
        Median value and standard deviation of soil resistivity.
    ti : array-like
        Time duration of short-circuits.
    Pi : array-like
        Probabilities associated with `ti` values.
    kwargs : dict
        Additional arguments for computing the foot resistance.
    
    Returns
    -------
    pdf_Et : real
        Probability density function value.
    """
    # Foot resistance from soil resistivity.
    mu_Rg, sigma_Rg = foot_resistance(mu_rho, sigma_rho, **kwargs)

    c = 0.116
    d = 0.058
    mu_z = c*mu_Rk + d*mu_Rg
    sigma_z = np.sqrt(c**2 * sigma_Rk**2 + d**2 * sigma_Rg**2)

    n = len(ti)
    pdf_Et = 0.
    for i in range(n):
        konst = Pi[i] / (np.sqrt(2*np.pi) * (sigma_z/np.sqrt(ti[i])))
        value = (x - mu_z/np.sqrt(ti[i])) / (sigma_z/np.sqrt(ti[i]))
        pdf_Et += konst * np.exp(-0.5 * value**2)
           
    return pdf_Et


def touch_voltage_tolerable_cdf(x, mu_Rk, sigma_Rk, mu_rho, sigma_rho, ti, Pi, 
                                kwargs):
    """
    Cumulative probability function of tolerable touch voltage.

    Arguments
    ---------
    x : float
        Voltage value.
    mu_Rk, sigma_Rk : float, float
        Median value and standard deviation of body resistance.
    mu_rho, sigma_rho : float, float
        Median value and standard deviation of soil resistivity.
    ti : array-like
        Time duration of short-circuits.
    Pi : array-like
        Probabilities associated with `ti` values.
    kwargs : dict
        Additional arguments for computing the foot resistance.
    
    Returns
    -------
    cdf_Et : real
        Cumulative probability function value.
    """
    # Foot resistance from soil resistivity.
    mu_Rg, sigma_Rg = foot_resistance(mu_rho, sigma_rho, **kwargs)

    c = 0.116
    d = 0.058
    mu_z = c*mu_Rk + d*mu_Rg
    sigma_z = np.sqrt(c**2 * sigma_Rk**2 + d**2 * sigma_Rg**2)

    n = len(ti)
    cdf_Et = 0.
    for i in range(n):
        value = (x - mu_z/np.sqrt(ti[i])) / (sigma_z/np.sqrt(ti[i]))
        cdf_Et += Pi[i] * 0.5*(1 + special.erf(value))  # correction!
        
    return cdf_Et


def touch_voltage_applied_pdf(x, mu_Va, sigma_Va):
    """
    Probability density function of applied touch voltage.

    Arguments
    ---------
    x : float
        Voltage value.
    mu_Va, sigma_Va : float, float
        Median value and standard deviation of applied touch voltage.
    
    Returns
    -------
    pdf_Va : float
        Probability density function of applied touch voltage.
    """
    konst = 1. / (np.sqrt(2*np.pi) * sigma_Va)
    value = (x - mu_Va)/sigma_Va
    pdf_Va = konst * np.exp(-0.5 * value**2)

    return pdf_Va


def kernel_function(x, *args):
    """
    Compute a kernel function of the main integral,
    as a product of the applied voltage PDF function
    and the tolerable touch voltage CDF function.
    """
    # PDF of the applied touch voltage.
    pdf = touch_voltage_applied_pdf(x, args[0], args[1])
    # CDF of the tolerable touch voltage.
    cdf = touch_voltage_tolerable_cdf(x, *args[2:])
    # Kernel function value.
    value = pdf * cdf

    return value


# Input data (mean values and std. deviations):
# Body resistance (Ohm).
mu_Rk = 1000.
sigma_Rk = 0.4 * mu_Rk

# Soil resistivity (Ohm*m).
mu_rho = np.array([100., 500., 1000.])
sigma_rho = 0.4 * mu_rho

# Grounding grid touch voltage (V).
mu_Va = 56.5
sigma_Va = 24.

# Short-circuit duration.
ti = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # seconds
Pi = [0.2, 0.4, 0.18, 0.1, 0.07, 0.05]  # sum(Pi) = 1
print(f'TEST: SUM(Pi) = {sum(Pi)}')

# Compute risk.
kwargs = {# Foot resistance parameters:
    'model': 'gravel',  # 'none', 'plate', 'feet', 'gravel'
    'h': 0.1,   # thickness of the gravel layer (m)
    'g': 2000.  # gravel resistivity (Ohm*m)
}
risk = []
for mu, sigma in zip(mu_rho, sigma_rho):
    arguments = (# Integral kernel function arguments:
        mu_Va, sigma_Va,  # touch_voltage_applied_pdf
        mu_Rk, sigma_Rk, mu, sigma, ti, Pi,  # touch_voltage_tolerable_cdf
        kwargs  # additional parameters
    )
    res, _ = integrate.quad(kernel_function, a=0., b=np.Inf, args=arguments)
    risk.append(res*100)

print('Risk of touch voltage:')
for rho, r in zip(mu_rho, risk):
    print(f'{rho:.1f} Ohm*m => {r:.3f} %')


# Skip execution of the rest of the code.
sys.exit()

# Test PDF and CDF functions.
x = np.linspace(0, 2000, 1000)
args = (mu_Rk, sigma_Rk, mu_rho[0], sigma_rho[0], ti, Pi,)
# CDF from the integral of PDF function.
y = touch_voltage_tolerable_pdf(x, *args)
y1 = integrate.cumulative_simpson(y, x=x)
# CDF directly from a function.
y2 = touch_voltage_tolerable_cdf(x, *args)

# Graphical comparison.
fig, ax = plt.subplots(figsize=(5,3.5))
ax.plot(x[:-1], y1, label='from integral of pdf')
ax.plot(x, y2, label='from cdf')
ax.legend(loc='best')
ax.set_xlabel('Soil resistivity (Ohm*m)')
ax.set_ylabel('Probability')
ax.grid(which='major', axis='both')
fig.tight_layout()
plt.show()
