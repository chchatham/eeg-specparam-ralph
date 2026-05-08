import numpy as np
import math
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf
from mne_bids import BIDSPath, read_raw_bids
import warnings

def generate_aperiodic_acf(tau, b, k):
    """
    Exact time-domain ACF for the Aperiodic component.
    Assumes exponent chi=2 (Lorentzian / Ornstein-Uhlenbeck process).
    """
    # Prevent division by zero or negative square roots
    k = max(k, 1e-5) 
    
    # R_ap(tau) = (10^b * pi / sqrt(k)) * exp(-2*pi*sqrt(k)*|tau|)
    amplitude = (10**b * np.pi) / np.sqrt(k)
    decay = np.exp(-2 * np.pi * np.sqrt(k) * np.abs(tau))
    return amplitude * decay

def generate_periodic_acf(tau, a, c, w, M=5):
    """
    Exact time-domain ACF for a single FOOOF periodic component.
    Evaluated using a truncated Taylor expansion of the exponentiated Gaussian.
    """
    r_pe = np.zeros_like(tau)
    
    # The M=0 term represents the baseline noise floor delta function
    # Approximated here as 1.0 at tau=0
    tau_zero_idx = np.argmin(np.abs(tau))
    r_pe[tau_zero_idx] = 1.0 
    
    # Sum over M Taylor expansion terms
    for m in range(1, M + 1):
        coef = ((np.log(10) * a)**m) / math.factorial(m)
        width_term = np.sqrt((2 * np.pi * w**2) / m)
        gaussian_decay = np.exp(-(2 * np.pi**2 * w**2 * tau**2) / m)
        cosine_oscillation = np.cos(2 * np.pi * c * tau)
        
        r_pe += coef * width_term * gaussian_decay * cosine_oscillation
        
    return r_pe

def time_domain_specparam_model(tau, b, k, a, c, w):
    """
    The exact full time-domain FOOOF model.
    Convolves the aperiodic and periodic ACFs.
    """
    r_ap = generate_aperiodic_acf(tau, b, k)
    r_pe = generate_periodic_acf(tau, a, c, w, M=5)
    
    # Time resolution for numerical convolution integral
    dtau = np.abs(tau[1] - tau[0])
    
    # Convolve and scale by dtau to approximate the continuous integral
    r_total = np.convolve(r_ap, r_pe, mode='same') * dtau
    return r_total

def fit_time_domain_specparam(time_series, sfreq, max_lag_sec=1.0):
    """
    Computes the empirical ACF of a time series and fits the time-domain model.
    """
    nlags = int(max_lag_sec * sfreq)
    
    # 1. Compute Empirical ACF using statsmodels (returns normalized ACF by default)
    # We multiply by variance to get the unnormalized autocovariance function
    empirical_acf = acf(time_series, nlags=nlags, fft=True) * np.var(time_series)
    
    # Create the lag time vector (tau)
    tau = np.arange(0, nlags + 1) / sfreq
    
    # 2. Define parameter bounds: [b, k, a, c, w]
    # b: offset, k: knee, a: peak amplitude, c: peak center freq, w: peak bandwidth
    lower_bounds = [-5.0, 0.0,  0.0,  1.0, 0.5]
    upper_bounds = [ 5.0, 50.0, 3.0, 40.0, 5.0]
    
    # Initial guesses: b=0, k=5, a=0.5, c=10 (alpha), w=2
    p0 = [0.0, 5.0, 0.5, 10.0, 2.0]
    
    # 3. Fit via Non-Linear Least Squares
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                time_domain_specparam_model, 
                tau, 
                empirical_acf, 
                p0=p0, 
                bounds=(lower_bounds, upper_bounds),
                maxfev=2000
            )
        return popt, tau, empirical_acf
    except RuntimeError:
        print("Optimization failed to converge.")
        return None, tau, empirical_acf

def process_bids_dataset(bids_root, subject, task, datatype='eeg'):
    """
    Loads BIDS data, extracts EEG time-series, and fits the time-domain model.
    """
    bids_path = BIDSPath(subject=subject, task=task, datatype=datatype, root=bids_root)
    
    print(f"Loading data from: {bids_path}")
    raw = read_raw_bids(bids_path, verbose=False)
    raw.load_data()
    
    # Apply standard basic preprocessing if not already done
    raw.pick_types(eeg=True, meg=False)
    raw.filter(l_freq=1.0, h_freq=50.0, verbose=False) 
    
    sfreq = raw.info['sfreq']
    channel_names = raw.ch_names
    data, times = raw.get_data(return_times=True)
    
    results = {}
    
    # Fit the model for the first 5 channels as an example
    for i, ch_name in enumerate(channel_names[:5]):
        print(f"Fitting channel: {ch_name}...")
        
        # Taking a 10-second segment for stability in ACF estimation
        segment = data[i, int(0*sfreq):int(10*sfreq)] 
        
        # Fit model
        popt, tau, emp_acf = fit_time_domain_specparam(segment, sfreq, max_lag_sec=0.5)
        
        if popt is not None:
            b, k, a, c, w = popt
            results[ch_name] = {
                'offset_b': b,
                'knee_k': k,
                'peak_amp_a': a,
                'peak_center_c': c,
                'peak_width_w': w
            }
            print(f"  -> Fit successful: Center Freq = {c:.2f} Hz, Knee = {k:.2f}")
            
    return results

# ==========================================
# Example Usage (assuming you have a BIDS dataset)
# ==========================================
if __name__ == "__main__":
    # Update these variables to point to your actual BIDS directory structure
    BIDS_ROOT = './bids_dataset'
    SUBJECT = '01'
    TASK = 'resting'
    
    # Uncomment to run on your local BIDS data:
    # fitted_parameters = process_bids_dataset(BIDS_ROOT, SUBJECT, TASK)
    # print(fitted_parameters)