def gregory_pi_approximation(n_terms):
    pi_over_4 = 0
    for n in range(n_terms+1):
        term = ((-1)**n) / (2*n+1)
        pi_over_4 += term
    pi_approx = 4*pi_over_4
    return pi_approx       


approx_pi = gregory_pi_approximation(1000)

print(f"Approximation = {approx_pi} ") 
