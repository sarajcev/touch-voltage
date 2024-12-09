import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate
from scipy import special


def two_layer_soil_equivalent(rho_1, rho_2, d, r):
    """
    Equivalent soil resistivity for a two-layer soil.

    Empirical fit to the solution of elliptic-integral
    potentials (Zaborsky). Actual two-layer soil is 
    approximated by the homogenous soil with an equivalent
    resistivity value.

    Parameters
    ----------
    rho_1 : float
        Resistivity of the upper soil layer (Ohm*m).
    rho_2 : float
        Resistivity of the lower soil layer (Ohm*m).
    d : float
        Depth of the upper soil layer (m).
    r : float
        Equivalent radius of the grounding grid (m).
    
    Returns
    -------
    rho : float
        Equivalent soil resitivity (Ohm*m).
    
    Notes
    -----
    Grounding grid is considered as a disk-like electrode
    buried just below the surface.
    """
    factor = (rho_2*r) / (rho_1*d)
    
    if rho_1 > rho_2:
        C = 1. / (1.4 + (rho_2/rho_1)**0.8)
    else:
        C = 1. / (1.4 + (rho_2/rho_1)**0.8 + (factor)**0.5)
    
    rho = rho_1 * ((1. + C*factor) / (1. + C*(r/d)))
    
    return rho


def reduction_factor_gravel(h, k):
    """
    Foot resistance reduction factor of gravel.

    Computing the reduction factor of the gravel layer, when
    determining the foot resistance. This is the simplified
    approach, where mutual resistance between the two feet 
    is not taken into account.

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
    eps = 1e-12  # tolerance
    suma = 0.
    n = 1
    while n < 1000:
        value = k**n / np.sqrt(1 + ((2*n*h)/0.08)**2)
        suma += value
        if abs(value) < eps:
            break
        n += 1

    cs = (1/0.96) * (1 + 2*suma)

    return cs


def foot_resistance_factor(rho, model='plate', **kwargs):
    """
    Computing foot resistance for the touch voltage.

    Arguments
    ---------
    mu_rho : float
        Median value of the soil resistivity (Ohm*m).
    sigma_rho : float
        Standard deviation of the soil resistivity.
    model : str, default='plate'
        Model of the footing for the resistance calculation. Can be
        one of the following: 'none', 'plate', 'feet', 'gravel'.
    kwargs : dict
        Additional parameters for the model 'gravel', which include
        'h': thickness of the gravel layer (m) and 
        'g': resistivity (Ohm*m).
    
    Returns
    -------
    factor : float
        Factor for calculating the foot resistance.
    """
    if model == 'none':
        # Foot resistance is equal to the soil resistance.
        factor = 1.
    
    elif model == 'plate':
        # Single foot as a circular disc of 200 cm^2.
        factor = 1./(4*0.08)

    elif model == 'feet':
        # Parallel connection of two feet with mutual resistance
        # between them, where each foot is modelled as a circular
        # disc of 200 cm^2, and distance between feet is 10 cm.
        factor = 1./(4*0.08) - 1./(2*np.pi*0.1)
    
    elif model == 'gravel':
        # Soil is covered by a layer of gravel having (usually)
        # higher resistivity (h: layer depth, g: gravel resistivity).
        h = kwargs['h']
        rho_gravel = kwargs['g']
        k = (rho - rho_gravel) / (rho + rho_gravel)
        cs = reduction_factor_gravel(h, k)
        factor = 1.5 * cs * (rho_gravel/rho)
    
    else:
        raise NotImplementedError(f'Model {model} is not recognized!')
    
    return factor


def z_values(*args):
    """Intermediate `z` statistical variable."""
    mu_Rk, sigma_Rk, mu_Rg, sigma_Rg = args

    c = 0.116
    d = 0.058
    mu_z = c*mu_Rk + d*mu_Rg
    sigma_z = np.sqrt(c**2 * sigma_Rk**2 + d**2 * sigma_Rg**2)

    return mu_z, sigma_z


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
    foot_factor = foot_resistance_factor(mu_rho, **kwargs)
    mu_Rg = foot_factor * mu_rho
    sigma_Rg = foot_factor * sigma_rho

    mu_z, sigma_z = z_values(mu_Rk, sigma_Rk, mu_Rg, sigma_Rg)

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
    Cumulative distribution function of tolerable touch voltage.

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
    foot_factor = foot_resistance_factor(mu_rho, **kwargs)
    mu_Rg = foot_factor * mu_rho
    sigma_Rg = foot_factor * sigma_rho

    mu_z, sigma_z = z_values(mu_Rk, sigma_Rk, mu_Rg, sigma_Rg)

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


def exposure_probability(Tf, Te, ff, fe, T):
    """
    Probability of the person's exposure to the dangerous voltage.

    Parameters
    ----------
    Tf : float
        Average duration of one ground fault (seconds).
    Te : float
        Average duration of one exposure of a person (hours).
    ff : int
        Average number of ground faults during the study period.
    fe : int
        Average number of person's exposure during the study period.
    T : float
        Study period (years).
    
    Returns
    -------
    probability : float
        Probability of person's exposure to the dangerous voltage.
    """
    # Convert seconds to years.
    Tf = Tf / 31536000.
    # Convert hours to years.
    Te = Te / 8760.

    q = 1. - (Tf + Te)/T + (Tf**2 + Te**2)/(2*T**2)
    
    eps = 1e-12  # tolerance
    Pe = 0.
    k = 1
    while k < 100:
        k_fact = special.factorial(k)
        value = (ff**k/k_fact) * (1. - q**(fe*k)) * np.exp(-ff)
        Pe += value
        if abs(value) < eps:
            break
        k += 1

    return Pe


# Input data (mean values and std. deviations):
# Body resistance (Ohm).
mu_Rk = 1000.
sigma_Rk = 0.4 * mu_Rk

# Soil resistivity (Ohm*m).
mu_rho = np.array([100., 500., 1000.])
sigma_rho = 0.4 * mu_rho

# Grounding grid touch voltage (V). This input must come 
# either from a numerical analysis or measurements of
# grounding grid under single-phase short circuit.
mu_Va = 56.5
sigma_Va = 24.

# Short-circuit durations & probabilities.
ti = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # seconds
Pi = [0.2, 0.4, 0.18, 0.1, 0.07, 0.05]  # sum(Pi) = 1

# Exposure to dangerous voltages data:
Tf = 1.  # fault duration (seconds)
Te = 1.  # exposure duration (hours)
ff = 1   # av. no of faults during T
fe = 10  # av no of exposures during T
T = 1.   # study period (years)

# Risk of touch voltage.
kwargs = {# Foot resistance parameters:
    'model': 'gravel',  # 'none', 'plate', 'feet', 'gravel'
    'h': 0.1,   # thickness of the gravel layer (m)
    'g': 2000.  # gravel resistivity (Ohm*m)
}
risk_touch_voltage = []
for mu, sigma in zip(mu_rho, sigma_rho):
    arguments = (# Integral kernel function arguments:
        mu_Va, sigma_Va,  # touch_voltage_applied_pdf
        mu_Rk, sigma_Rk, mu, sigma, ti, Pi,  # touch_voltage_tolerable_cdf
        kwargs  # additional parameters
    )
    res, _ = integrate.quad(kernel_function, a=0., b=np.Inf, args=arguments)
    risk_touch_voltage.append(res)

print('Risk of touch voltage:')
for rho, r in zip(mu_rho, risk_touch_voltage):
    print(f'{rho:.1f} Ohm*m => {r*100:.3f} %')

# Exposure probability.
Pe = exposure_probability(Tf, Te, ff, fe, T)
print(f'Exposure probability: {Pe*100:.3f} %')

# Total risk of touch voltage exposure.
risk_total = Pe * np.array(risk_touch_voltage)
print('Total risk of touch voltage exposure:')
for rho, r in zip(mu_rho, risk_total):
    print(f'{rho:.1f} Ohm*m => {r:.4e}')
