import numpy as np
import scipy.stats as stats
import scipy.special as sp
import matplotlib.pyplot as plt
import math

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

def mc_variance(S, K, T, r, sigma, N=50000):
    Z = np.random.randn(N)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.exp(-r * T) * np.maximum(ST - K, 0)
    return np.var(payoffs, ddof=1)

def test_hermite_properties():
    np.random.seed(42)
    N = 1000000
    Z = np.random.randn(N)
    
    H1 = sp.eval_hermitenorm(1, Z)
    H2 = sp.eval_hermitenorm(2, Z)
    H3 = sp.eval_hermitenorm(3, Z)
    
    print("--- 1. Espérance Nulle E[H_n(Z)] = 0 ---")
    print(f"E[H1] = {np.mean(H1):.6f}")
    print(f"E[H2] = {np.mean(H2):.6f}")
    print(f"E[H3] = {np.mean(H3):.6f}\n")
    
    print("--- 2. Orthogonalité E[H_n(Z) * H_m(Z)] = 0 (si n != m) ---")
    print(f"E[H1 * H2] = {np.mean(H1 * H2):.6f}")
    print(f"E[H1 * H3] = {np.mean(H1 * H3):.6f}")
    print(f"E[H2 * H3] = {np.mean(H2 * H3):.6f}\n")
    
    print("--- 3. Variance Var(H_n(Z)) = n! ---")
    print(f"Var(H1) = {np.var(H1):.6f} | Attendu : {math.factorial(1)}")
    print(f"Var(H2) = {np.var(H2):.6f} | Attendu : {math.factorial(2)}")
    print(f"Var(H3) = {np.var(H3):.6f} | Attendu : {math.factorial(3)}\n")

def pricer_monte_carlo_complet():
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    N = 500000 

    Z = np.random.randn(N)
    
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    payoffs_call = np.exp(-r * T) * np.maximum(ST - K, 0)
    prix_mc_naif = np.mean(payoffs_call)
    var_mc_naif = np.var(payoffs_call, ddof=1)
    
    payoffs_put = np.exp(-r * T) * np.maximum(K - ST, 0)
    prix_mc_controle = np.mean(payoffs_put) + S0 - K * np.exp(-r * T)
    var_mc_controle = np.var(payoffs_put, ddof=1)
    
    prix_exact = bs_call(S0, K, T, r, sigma)
    
    erreur_naif = np.abs(prix_mc_naif - prix_exact)
    erreur_controle = np.abs(prix_mc_controle - prix_exact)
    
    print(f"Prix Exact (Black-Scholes) : {prix_exact:.5f}\n")
    
    print("--- Monte Carlo Naïf ---")
    print(f"Prix Estimé  : {prix_mc_naif:.5f}")
    print(f"Erreur Abs.  : {erreur_naif:.5f}")
    print(f"Variance     : {var_mc_naif:.2f}\n")
    
    print("--- Monte Carlo Contrôlé (Parité Call-Put) ---")
    print(f"Prix Estimé  : {prix_mc_controle:.5f}")
    print(f"Erreur Abs.  : {erreur_controle:.5f}")
    print(f"Variance     : {var_mc_controle:.2f}\n")
    
    print(f"Facteur de réduction de la variance : {var_mc_naif / var_mc_controle:.2f}x\n")

def plot_impact_parametres():
    K, r = 100.0, 0.05

    S_range = np.linspace(50, 150, 100)
    C_S = bs_call(S_range, K, 1.0, r, 0.2)

    vol_range = np.linspace(0.01, 0.8, 50)
    C_vol = bs_call(100.0, K, 1.0, r, vol_range)
    V_vol = [mc_variance(100.0, K, 1.0, r, v) for v in vol_range]

    T_range = np.linspace(0.1, 3.0, 50)
    C_T = bs_call(100.0, K, T_range, r, 0.2)
    V_T = [mc_variance(100.0, K, t, r, 0.2) for t in T_range]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(S_range, C_S, color='blue', lw=2)
    axes[0, 0].set_title('Prix Call vs Prix Spot (S0)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(vol_range, C_vol, color='red', lw=2)
    axes[0, 1].set_title('Prix Call vs Volatilité (\u03c3)')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(T_range, C_T, color='green', lw=2)
    axes[0, 2].set_title('Prix Call vs Maturité (T)')
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].axis('off') 

    axes[1, 1].plot(vol_range, V_vol, color='darkred', lw=2)
    axes[1, 1].set_title('Variance Monte Carlo vs Volatilité (\u03c3)')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(T_range, V_T, color='darkgreen', lw=2)
    axes[1, 2].set_title('Variance Monte Carlo vs Maturité (T)')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("====================================================")
    print(" PARTIE 1 : PROPRIÉTÉS DES POLYNÔMES D'HERMITE")
    print("====================================================\n")
    test_hermite_properties()
    
    print("====================================================")
    print(" PARTIE 2 : MONTE CARLO & RÉDUCTION DE VARIANCE")
    print("====================================================\n")
    pricer_monte_carlo_complet()
    
    print("====================================================")
    print(" PARTIE 3 : GÉNÉRATION DES GRAPHIQUES")
    print("====================================================\n")
    print("Génération des graphiques en cours... (fermez la fenêtre pour terminer)")
    plot_impact_parametres()