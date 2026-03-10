import numpy as np
import scipy.special as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12})

def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

def eval_variance_reduction():
    np.random.seed(42)
    N = 100000
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    k_max = 8
    
    prix_exact_BS = black_scholes_call(S0, K, T, r, sigma)
    
    Z = np.random.randn(N)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.exp(-r * T) * np.maximum(ST - K, 0)
    
    var_totale = np.var(payoffs, ddof=1)
    
    liste_k = []
    liste_lambda = []
    liste_var_residuelle = [var_totale]
    liste_r2_marginal = []
    liste_r2_cumule = [0.0]
    
    payoffs_residuels = payoffs.copy()
    
    for k in range(1, k_max + 1):
        H_k = sp.eval_hermitenorm(k, Z)
        
        cov = np.cov(payoffs_residuels, H_k)[0, 1]
        var_Hk = math.factorial(k)
        lambda_k = cov / var_Hk
        
        payoffs_residuels -= lambda_k * H_k
        var_actuelle = np.var(payoffs_residuels, ddof=1)
        
        r2_marginal = (liste_var_residuelle[-1] - var_actuelle) / var_totale
        r2_cumule = 1.0 - (var_actuelle / var_totale)
        
        liste_k.append(k)
        liste_lambda.append(abs(lambda_k))
        liste_var_residuelle.append(var_actuelle)
        liste_r2_marginal.append(r2_marginal)
        liste_r2_cumule.append(r2_cumule)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot([0] + liste_k, liste_var_residuelle, marker='o', color='black')
    axes[0].set_title("Variance Résiduelle")
    axes[0].set_xlabel("Ordre k")
    axes[0].set_ylabel("Variance")
    
    axes[1].bar(liste_k, liste_lambda, color='steelblue')
    axes[1].plot(liste_k, liste_lambda, marker='x', color='black', lw=1)
    axes[1].set_title("Magnitude des Projections ($|\lambda_k|$)")
    axes[1].set_xlabel("Ordre k")
    axes[1].set_yscale("log")
    
    axes[2].plot(liste_k, liste_r2_cumule[1:], marker='s', color='darkgreen', label="R² Cumulé")
    axes[2].bar(liste_k, liste_r2_marginal, color='lightgreen', alpha=0.6, label="R² Marginal")
    axes[2].set_title("Variance Expliquée ($R^2$)")
    axes[2].set_xlabel("Ordre k")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    prix_naif = np.mean(payoffs)
    prix_controle = np.mean(payoffs_residuels)
    se_naif = np.sqrt(var_totale / N)
    se_controle = np.sqrt(liste_var_residuelle[-1] / N)
    
    print(f"Prix Théorique Exact (BS) : {prix_exact_BS:.4f} €")
    print("-" * 45)
    print(f"Prix Naïf Estimé          : {prix_naif:.4f} €")
    print(f"Intervalle de Conf. (95%) : [{prix_naif - 1.96*se_naif:.4f} €, {prix_naif + 1.96*se_naif:.4f} €]")
    print(f"Variance Naïve            : {var_totale:.4f}")
    print("-" * 45)
    print(f"Prix Contrôlé Estimé      : {prix_controle:.4f} €")
    print(f"Intervalle de Conf. (95%) : [{prix_controle - 1.96*se_controle:.4f} €, {prix_controle + 1.96*se_controle:.4f} €]")
    print(f"Variance Finale (k={k_max})   : {liste_var_residuelle[-1]:.4f}")
    print("-" * 45)
    print("ÉVOLUTION DE LA RÉDUCTION DE VARIANCE :")
    print("k\t|Lambda|\tVar Res.\tR² Cumulé")
    for i in range(k_max):
        print(f"{liste_k[i]}\t{liste_lambda[i]:.4f}\t\t{liste_var_residuelle[i+1]:.4f}\t\t{liste_r2_cumule[i+1]:.2%}")

if __name__ == "__main__":
    eval_variance_reduction()