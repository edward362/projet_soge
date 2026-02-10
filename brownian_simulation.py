import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def verification_brownien_complete():
    # --- Paramètres de simulation ---
    T = 1.0         # Temps total
    N = 1000        # Nombre de pas
    dt = T / N      # Pas de temps
    M_sim = 5000    # Nombre de simulations pour les stats

    t_grid = np.linspace(0, T, N+1)

    print(f"--- Simulation de {M_sim} trajectoires Browniennes sur [0, {T}] ---")

    # 1. Génération des incréments Gaussiens centrés réduits N(0,1)
    Z = np.random.randn(M_sim, N)

    # 2. Mise à l'échelle pour obtenir la variance dt
    dW = np.sqrt(dt) * Z

    # 3. Construction des trajectoires (Somme cumulée)
    # On ajoute 0 au départ car W_0 = 0
    W = np.concatenate([np.zeros((M_sim, 1)), np.cumsum(dW, axis=1)], axis=1)

    # --- Visualisation et Tests ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.3)

    # Graphique 1 : Trajectoires
    ax0 = axes[0, 0]
    ax0.plot(t_grid, W[:10, :].T, lw=1, alpha=0.7)
    ax0.set_title("1. Échantillon de trajectoires $W_t$")
    ax0.set_xlabel("Temps $t$")
    ax0.set_ylabel("$W_t$")
    ax0.grid(True, alpha=0.3)

    # Graphique 2 : Test de la Variance (Diffusion)
    # On vérifie que la variance empirique croit linéairement avec le temps (pente = 1)
    ax1 = axes[0, 1]
    var_empirique = np.var(W, axis=0)
    
    ax1.plot(t_grid, var_empirique, 'b-', lw=2, label='Variance Empirique')
    ax1.plot(t_grid, t_grid, 'r--', lw=2, label='Théorie: $Var(W_t)=t$')
    ax1.set_title("2. Vérification de la Variance (Diffusion)")
    ax1.set_xlabel("Temps $t$")
    ax1.legend()
    ax1.grid(True)

    # Graphique 3 : Test de Normalité des incréments
    # On vérifie que les pas suivent bien une loi N(0, dt)
    ax2 = axes[1, 0]
    all_increments = dW.flatten()
    
    ax2.hist(all_increments, bins=100, density=True, color='skyblue', alpha=0.7, label='Hist. Incréments')
    
    # Superposition de la densité théorique
    x_vals = np.linspace(min(all_increments), max(all_increments), 100)
    pdf_theo = stats.norm.pdf(x_vals, loc=0, scale=np.sqrt(dt))
    ax2.plot(x_vals, pdf_theo, 'r-', lw=2, label='Densité $\mathcal{N}(0, dt)$')
    ax2.set_title(f"3. Normalité des incréments (Pas dt={dt:.4f})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Test statistique de Shapiro-Wilk (sur un échantillon réduit pour la performance)
    stat, p_val = stats.shapiro(all_increments[:5000])
    print(f"Test Shapiro-Wilk (Normalité) : p-value = {p_val:.4f}")

    # Graphique 4 : Test d'Indépendance (Autocorrélation)
    # On vérifie l'absence de mémoire du processus (Bruit blanc)
    ax3 = axes[1, 1]
    sample_increments = dW[0, :]
    
    ax3.acorr(sample_increments, maxlags=20, lw=2, color='green')
    ax3.set_title("4. Autocorrélation des incréments")
    ax3.set_xlabel("Lag")
    ax3.grid(True, alpha=0.3)

    plt.show()

if __name__ == "__main__":
    verification_brownien_complete()