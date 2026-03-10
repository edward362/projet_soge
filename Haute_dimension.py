import numpy as np

def pricer_basket_multidim():
    np.random.seed(42)
    N = 100000
    T, r, K = 1.0, 0.05, 100.0
    
    S0_A, vol_A = 100.0, 0.20
    S0_B, vol_B = 100.0, 0.30
    rho_2D = 0.60
    
    Z1 = np.random.randn(N)
    Z2 = np.random.randn(N)
    
    X1 = Z1
    X2 = rho_2D * Z1 + np.sqrt(1 - rho_2D**2) * Z2
    
    ST_A = S0_A * np.exp((r - 0.5 * vol_A**2) * T + vol_A * np.sqrt(T) * X1)
    ST_B = S0_B * np.exp((r - 0.5 * vol_B**2) * T + vol_B * np.sqrt(T) * X2)
    
    panier_2D = 0.5 * ST_A + 0.5 * ST_B
    payoffs_2D = np.exp(-r * T) * np.maximum(panier_2D - K, 0)
    
    print(f"Prix Basket 2D : {np.mean(payoffs_2D):.4f}")
    print(f"Var Basket 2D  : {np.var(payoffs_2D, ddof=1):.2f}\n")

    S0_actifs = np.array([100.0, 100.0, 100.0])
    vols = np.array([0.20, 0.30, 0.25])
    poids = np.array([0.333, 0.333, 0.334])
    
    Sigma = np.array([
        [ 1.0,  0.6,  0.4],
        [ 0.6,  1.0, -0.2],
        [ 0.4, -0.2,  1.0]
    ])
    
    L = np.linalg.cholesky(Sigma)
    Z_matrice = np.random.randn(3, N)
    X_matrice = L @ Z_matrice
    
    panier_3D = np.zeros(N)
    
    for i in range(3):
        ST_i = S0_actifs[i] * np.exp((r - 0.5 * vols[i]**2) * T + vols[i] * np.sqrt(T) * X_matrice[i])
        panier_3D += poids[i] * ST_i
        
    payoffs_3D = np.exp(-r * T) * np.maximum(panier_3D - K, 0)
    
    print(f"Prix Basket 3D : {np.mean(payoffs_3D):.4f}")
    print(f"Var Basket 3D  : {np.var(payoffs_3D, ddof=1):.2f}")

if __name__ == "__main__":
    pricer_basket_multidim()