import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

def test_hermite():
    np.random.seed(42)
    N = 1000000
    Z = np.random.randn(N)
    
    H1 = sp.eval_hermitenorm(1, Z)
    H2 = sp.eval_hermitenorm(2, Z)
    H3 = sp.eval_hermitenorm(3, Z)
    
    print("Hermite Espérances :", np.mean(H1), np.mean(H2), np.mean(H3))
    print("Hermite Orthogonalité :", np.mean(H1 * H2), np.mean(H2 * H3))
    print("Hermite Variances :", np.var(H1), np.var(H2), np.var(H3))

def test_influence_param():
    np.random.seed(42)
    N = 100000
    S0, K, r = 100, 100, 0.05
    
    for T in [0.5, 2.0]:
        for sigma in [0.2, 0.5]:
            Z = np.random.randn(N)
            ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
            payoffs = np.exp(-r * T) * np.maximum(ST - K, 0)
            
            variance = np.var(payoffs, ddof=1)
            erreur = np.sqrt(variance / N)
            print(f"T={T}, vol={sigma} | Var: {variance:.2f}, Err: {erreur:.4f}")

def test_parite():
    np.random.seed(42)
    N = 100000
    S0, K, T, r, sigma = 100, 100, 2.0, 0.05, 0.4
    
    Z = np.random.randn(N)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    call_payoffs = np.exp(-r * T) * np.maximum(ST - K, 0)
    var_call = np.var(call_payoffs, ddof=1)
    
    put_payoffs = np.exp(-r * T) * np.maximum(K - ST, 0)
    var_put = np.var(put_payoffs, ddof=1)
    
    call_via_put = np.mean(put_payoffs) + S0 - K * np.exp(-r * T)
    
    print("Call Direct :", np.mean(call_payoffs), "| Variance :", var_call)
    print("Call Parité :", call_via_put, "| Variance :", var_put)

def test_moneyness_et_volatilite():
    np.random.seed(42)
    N = 50000
    K, T, r = 100.0, 1.0, 0.05
    
    # --- PARTIE 1 : IMPACT DU SPOT (S0) ---
    sigma_fixe = 0.30
    S_range = np.linspace(50, 150, 40)
    var_naif = []
    var_parite = []
    
    for S0 in S_range:
        Z = np.random.randn(N)
        ST = S0 * np.exp((r - 0.5 * sigma_fixe**2) * T + sigma_fixe * np.sqrt(T) * Z)
        
        payoffs_call = np.exp(-r * T) * np.maximum(ST - K, 0)
        var_naif.append(np.var(payoffs_call, ddof=1))
        
        payoffs_put = np.exp(-r * T) * np.maximum(K - ST, 0)
        var_parite.append(np.var(payoffs_put, ddof=1))
        
    # --- PARTIE 2 : IMPACT DE LA VOLATILITÉ SELON MONEYNESS ---
    vol_range = np.linspace(0.05, 0.8, 40)
    var_vol_OTM = []
    var_vol_ATM = []
    var_vol_ITM = []
    
    for vol in vol_range:
        Z = np.random.randn(N)
        
        # OTM (S0 = 80)
        ST_OTM = 80.0 * np.exp((r - 0.5 * vol**2) * T + vol * np.sqrt(T) * Z)
        var_vol_OTM.append(np.var(np.exp(-r * T) * np.maximum(ST_OTM - K, 0), ddof=1))
        
        # ATM (S0 = 100)
        ST_ATM = 100.0 * np.exp((r - 0.5 * vol**2) * T + vol * np.sqrt(T) * Z)
        var_vol_ATM.append(np.var(np.exp(-r * T) * np.maximum(ST_ATM - K, 0), ddof=1))
        
        # ITM (S0 = 120)
        ST_ITM = 120.0 * np.exp((r - 0.5 * vol**2) * T + vol * np.sqrt(T) * Z)
        var_vol_ITM.append(np.var(np.exp(-r * T) * np.maximum(ST_ITM - K, 0), ddof=1))

    # --- TRACÉ DES GRAPHIQUES ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Graphe 1 : Variance vs Spot
    axes[0].plot(S_range, var_naif, label='Call Naïf', color='red', lw=2)
    axes[0].plot(S_range, var_parite, label='Call via Put (Parité)', color='green', lw=2)
    axes[0].axvline(K, color='black', linestyle='--', label='ATM (K=100)')
    axes[0].set_title('Variance vs Prix Spot (S0)')
    axes[0].set_xlabel('Prix Spot (S0)')
    axes[0].set_ylabel('Variance de l\'estimateur')
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)
    
    # Graphe 2 : Variance vs Volatilité
    axes[1].plot(vol_range, var_vol_OTM, label='OTM (S0=80)', color='blue', lw=2)
    axes[1].plot(vol_range, var_vol_ATM, label='ATM (S0=100)', color='green', lw=2)
    axes[1].plot(vol_range, var_vol_ITM, label='ITM (S0=120)', color='red', lw=2)
    axes[1].set_title('Variance du Call vs Volatilité (\u03c3)')
    axes[1].set_xlabel('Volatilité (\u03c3)')
    axes[1].set_ylabel('Variance de l\'estimateur')
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_hermite()
    print("-" * 30)
    test_influence_param()
    print("-" * 30)
    test_parite()
    print("-" * 30)
    test_moneyness_et_volatilite()