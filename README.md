# 📈 Hermite-Transformed Brownian Motions for Monte Carlo Pricing

## 🎯 Project Objective

The purpose of this project is to study the advantage of a classical Brownian motion generator compared to a version transformed using **Hermite polynomials**.

The main goal is to evaluate whether this transformation improves Monte Carlo pricing, in particular:
- the **statistical quality** of simulations,
- and the **speed of convergence** of the Monte Carlo estimator.

We compare two sampling methods:
- **sampling1**: classical Brownian motion simulation  
- **sampling2**: Hermite-transformed Brownian motion simulation  

---

## 🧠 Background

In Monte Carlo pricing, Brownian motions are usually simulated using generators based on **uniform random variables**.

Other pseudo-random generators exist (Sobol, Halton, etc.), but in this project we deliberately stay in the **classical Monte Carlo framework** and investigate instead:

> the use of **Hermite polynomials** to orthogonalize simulated Brownian motions.

The objective is to determine whether this transformation leads to:
- better statistical properties,
- faster convergence,
- and more efficient option pricing.

---

## 📦 Project Content

### 1️⃣ Statistical Study

- Generate high-dimensional Brownian motions (dimension \( N \)) using:
  - **sampling1** (classical),
  - **sampling2** (Hermite-transformed).
- Compare both methods from a **statistical point of view**:
  - correlation structure,
  - variance reduction,
  - distributional properties.

---

### 2️⃣ Basket Option Pricing – Effect of Dimensionality

- Price a basket option (e.g. European call) using:
  - classical sampling,
  - Hermite-transformed sampling.
- Compare:
  - Monte Carlo convergence speed,
  - estimator variance,
  - stability with respect to dimension.

---

### 3️⃣ Low-Dimensional Case – Heston Model (Bivariate)

- Study the joint dynamics of:
  - asset price \( S_t \),
  - volatility \( \sigma_t \).
- Perform vanilla option pricing under the Heston model using:
  - classical Brownian sampling,
  - Hermite-transformed Brownian sampling.
- Compare:
  - pricing accuracy,
  - convergence behavior,
  - statistical properties.

---

### 4️⃣ Bonus – Corridor Option on Libor

- Simulate the underlying using:
  - a **Hermite-transformed Brownian bridge**,
  - a **classical time-discretized Brownian motion** on \([T_1, T_2]\).
- Compare:
  - the statistical behavior of both simulations,
  - the impact on vanilla option pricing.

---

## 🔧 Practical Remarks

- No external dataset is required.
- All parameters (strike, interest rate, volatility, maturity, etc.) are free to choose.
- Scientific references for the project are provided separately.

---

## 📚 References

You may refer to the articles provided for the theoretical background and project description.

---
