# list all combinations of ARIMA possible with AR 0 or 1, I 1 and MA 1 or 2
import itertools
def generate_arima_combinations():
    p_values = [0, 1]  # AR terms
    d_values = [1]     # I terms
    q_values = [1, 2]  # MA terms

    combinations = list(itertools.product(p_values, d_values, q_values))
    return combinations

if __name__ == "__main__":
    arima_combinations = generate_arima_combinations()
    for combo in arima_combinations:
        print(f"ARIMA{combo}")