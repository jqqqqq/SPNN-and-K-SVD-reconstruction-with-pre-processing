import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv

# Model: y = y0 + A1 * exp(-x / t1) + A2 * exp(-x / t2)
def double_exp(x, y0, A1, t1, A2, t2):
    return y0 + A1 * np.exp(-x / t1) + A2 * np.exp(-x / t2)

def load_columnwise_csv(path):
    """
    Reads CSV with first column = x, second column = y.
    Ignores empty lines and lines starting with '#'.
    """
    xs, ys = [], []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # skip comment lines
            if row[0].strip().startswith('#'):
                continue
            if len(row) < 2:
                continue
            try:
                x = float(row[0].strip())
                y = float(row[1].strip())
            except ValueError:
                # Skip header or non-numeric rows
                continue
            xs.append(x)
            ys.append(y)
    if len(xs) == 0:
        raise ValueError("No valid numeric data found (need first column x, second column y).")
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    return x, y

def fit_double_exp(x, y):
    # Initial guesses
    y0_guess = np.min(y)
    amp_guess = np.max(y) - y0_guess
    A1_guess = amp_guess * 0.6
    A2_guess = amp_guess * 0.4

    xrange = np.max(x) - np.min(x) if np.max(x) > np.min(x) else 1.0
    t1_guess = max(xrange / 5.0, 1e-6)
    t2_guess = max(xrange / 1.5, 1e-6)

    p0 = [y0_guess, A1_guess, t1_guess, A2_guess, t2_guess]
    # Bounds: t1, t2 > 0
    lower_bounds = [-np.inf, -np.inf, 1e-9, -np.inf, 1e-9]
    upper_bounds = [ np.inf,  np.inf,  np.inf,  np.inf,  np.inf]

    popt, pcov = curve_fit(
        double_exp, x, y, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=200000
    )
    return popt, pcov

def print_params_with_errors(popt, pcov):
    # Standard errors are sqrt of diagonal of covariance
    with np.errstate(invalid='ignore'):
        perr = np.sqrt(np.diag(pcov))

    names = ["y0", "A1", "t1", "A2", "t2"]
    # Format to three decimals for errors; values to six significant digits
    print("Fitted parameters (value ± error):")
    for name, val, err in zip(names, popt, perr):
        if np.isfinite(err):
            print(f"{name} = {val:.6g} ± {err:.3f}")
        else:
            print(f"{name} = {val:.6g} ± n/a")

def main(csv_path):
    x, y = load_columnwise_csv(csv_path)
    popt, pcov = fit_double_exp(x, y)

    # Report parameters with errors (three decimals for errors)
    print_params_with_errors(popt, pcov)

    # Compute fitted curve (sorted for plotting)
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    y_fit = double_exp(xs, *popt)

    # Optional: goodness-of-fit (R^2)
    ss_res = np.sum((ys - y_fit) ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    print(f"R^2 = {r2:.4f}")

    # Plot
    import matplotlib
    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, s=22, label='Data', color='tab:blue')
    plt.plot(xs, y_fit, 'r-', label='Double-exp fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Replace with your CSV file path
    main('data_columnwise.csv')
