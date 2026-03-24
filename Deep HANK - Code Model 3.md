# Chat History
[‎Gemini - direct access to Google AI](https://gemini.google.com/share/57e6aecf1e51): Some references at the end about Permutations invariance in finite agent approach.


# Deep HANK: Code Implementation Writeup

This document bridges the theoretical model in [Overleaf/main.tex](Overleaf/main.tex) (PDF: [Model_simplified_WIP_ver2 1](assets/Model_simplified_WIP_ver2%201.pdf) and a finite-population deep-learning implementation following [08-SolvingKS_Master](assets/08-SolvingKS_Master.pdf), with calibration and tricks from [PS3 - Deep Learning W-Residual Solver.ipynb](sample%20code/PS3%20-%20Deep%20Learning%20W-Residual%20Solver.ipynb), [Seung/nn_finag_Seung.ipynb](sample%20code/Seung/nn_finag_Seung.ipynb), and [GLMP_EMINN.pdf](References/GLMP_EMINN.pdf). Notation and equations are aligned with the LaTeX source.

Refrences:
[GLMP_EMINN](assets/GLMP_EMINN.pdf)

> [!PDF|] [GLMP_EMINN, p.25](ECO%20529%202025%20Jonathan%20Payne/Deep%20HANK/Deep%20HANK%20-%20Code%20Model%203/assets/GLMP_EMINN.pdf#page=25&selection=78,61,80,32)
> > It is therefore advisable to start with a pre-specified sampling distribution in early training and switch to a dynamic sampling scheme later on
> 

## Meeting agenda
1. Can't pindown wage
2. Have to pindown wage that depends on distribution.
3. alternative
	1. Calvo for endo labor supply
		1. might not have neat HJBE. complicated linearization process. so might be better for rotemberg.
4. labor income is taxed. 


---

# Derivation Sketch ([2026-02-27](2026-02-27))
[0227-Exogenous Labor](assets/Deep_HANK.pdf)
## Master equation

In the fully recursive formulation, the household value function $V(a,n,z,\pi,g)$ depends on individual states $x=(a,n)$ and aggregate states $m=(z,\pi,g)$. 

Define $\mathcal{L}$ operator
$$
\begin{align}
 \mathcal{L}V(x;m)= &- \rho V(a,n,z,\pi,g)+ u\big( (u')^{-1}[ \partial_a V(a,n,z,\pi,g) ] \big)
+ \psi(a) \cdot I_{\{ a \leq \underline{a} \}}
 \\
+ &  \partial_a V(a,n,z,\pi,g)\, s(a,n,z,\pi,g)
 \\
+ &  \lambda(n) \big( V(a,\check{n},z,\pi,g) - V(a,n,z,\pi,g) \big)
 \\
+ &  \mu_z(z)\, \partial_z V + \tfrac{1}{2} \sigma_z^2 \partial_{zz} V
 \\
+ &  \mu_\pi(z,\pi,g)\, \partial_\pi V + \tfrac{1}{2} \sigma_z^2 \pi^2 \partial_{\pi\pi} V
 \\
- &  \sigma_z^2 \pi\, \partial_{\pi z} V
 \\
+ &  \sum_{j=1}^2 \int_{-\infty}^{\infty} D_g V(a',n_j;z,\pi,g)\, \mu_g(a',n_j; z,\pi,g)\, da'
 \\
= & -\rho V+ u(c^{*}(x,m))+\psi(a)I\{ a\leq a_{lb} \} \\
+ & \mathcal{L}_{x}V \\
+ & \mathcal{L}_{z,\pi}V \\
+ & \mathcal{L}_{g}V
\end{align}
$$

Here $(a',n_j) \mapsto D_g V(a',n_j; z,\pi,g)$ is the Fréchet derivative of $V$ with respect to the distribution $g$ evaluated at $(a',n_j)$.

The state variables evolve as

- **Individual wealth drift (saving flow)**  
  $s(a,n,z,\pi,g) = r(z,\pi,g)\,a + w(z,\pi,g)\,n - (u')^{-1}[\partial_a V(a,n,z,\pi,g)] + \underbrace{ (Y-w(z,\pi,g)N^{*}) }_{ \text{rebate adjustment cost} }$
- **Real interest rate**:  
  $r(z,\pi,g) = r^* + (\phi_\pi - 1)\pi$.
- **Real wage** (See closure from below)

- **TFP drift (Ornstein–Uhlenbeck)**  
  $\mu_z(z) = \eta(\bar{z} - z)$.  
  **TFP volatility:** $\sigma_z$ (coefficient of $dW^0$ in $dz$).

- **Inflation drift (NKPC)**  
  $\mu_\pi(z,\pi,g) = r^* \pi + (\phi_\pi - 1) \pi^2 + \tfrac{\epsilon}{\psi}\big( \tfrac{\epsilon-1}{\epsilon} - \tfrac{w(z,\pi,g)}{e^z} \big) - \big( \mu_z(z) - \tfrac{1}{2}\sigma_z^2 \big)\pi$.  
  **Inflation volatility:** $-\sigma_z \pi$ (coefficient of $dW^0$ in $d\pi$; same $dW^0$ as in $dz$).

- **Distribution drift (KFE)**  
  $\mu_g(a,n; z,\pi,g) = - \partial_a\big( s(a,n,z,\pi,g)\, g(a,n) \big) + \lambda(\check{n})\, g(a,\check{n}) - \lambda(n)\, g(a,n)$.

- **Borrowing penalty**  
soft penalty $\psi(a) = -\tfrac{\kappa}{2}(a - a_{lb})^2$ on $[\underline{a}, a_{lb}]$, we have $\partial_a \psi(a) = -\kappa(a - a_{lb})$ and the indicator is $I_{\{a \leq a_{lb}\}}$


## $W:=V_{a}$ representation
To solve our model in a smoother space, we employ a transformation from $V$ to $V_{a}\equiv W$. Applying [[Envelope Theorem]] to our operator we have
$$
\begin{align}
0=\mathcal{L}W= & -\rho W+u'(c^{*}) \underbrace{ \partial \frac{c^{*}}{ \partial a} }_{ =0,\text{by Envelope} }+\psi'(a)I\{ a\leq a_{lb} \} \\
+ & W_{a}(x,m)s(x,m)+W(x,m)r+W(x,m) \partial \frac{c^{*}}{ \partial a} \\
+ &  \lambda(n) \big( W(a,\check{n},z,\pi,g) - W(a,n,z,\pi,g) \big)
 \\
+ &  \mu_z(z)\, \partial_z W + \tfrac{1}{2} \sigma_z^2 \partial_{zz} W
 \\
+ &  \mu_\pi(z,\pi,g)\, \partial_\pi W + \tfrac{1}{2} \sigma_z^2 \pi^2 \partial_{\pi\pi} W
 \\
- &  \sigma_z^2 \pi\, \partial_{\pi z} W
 \\
+ &  \sum_{j=1}^2 \int_{-\infty}^{\infty} D_g W(a',n_j;z,\pi,g)\, \mu_g(a',n_j; z,\pi,g)\, da' \\
= & (r-\rho)W(x,m)+\psi'(a)I\{ a\leq a_{lb} \} \\
+ & \underbrace{ W_{a}(x,m)s(c^{*};x,m)+\lambda (n)(W(a,\check{n};m)-W(x,m)) }_{ \mathcal{L}_{x}W } \\
+ & \underbrace{ \mu_{z}(z)W_{z}+\frac{1}{2}\sigma ^{2}_{z}W_{z z} +\mu_{\pi}(m)W_{\pi}+\frac{1}{2}\sigma ^{2}_{z}\pi ^{2}W_{\pi \pi}-\sigma ^{2}_{z}\pi W_{\pi z} }_{ \mathcal{L}_{\pi,z}W } \\
+ &  \underbrace{ \sum_{j=1}^2 \int_{-\infty}^{\infty} D_g W(a',n_j;z,\pi,g)\, \mu_g(a',n_j; z,\pi,g)\, da'  }_{ \mathcal{L}_{g}W }\\
\end{align}
$$

## Approximation Method 1 - Finite Population Approximation
We approximate infinite-dimensional $g$ using a finite panel
$$
\hat{\varphi}=\{(a_j,n_j)\}_{j=1}^{N}.
$$
In code, each training sample contains one focal agent $i=0$ and $(N-1)$ other agents.

Neural approximation target is
$$
\hat{W}(a^i,n^i,z,\pi,\hat{\varphi}^{-i}),
$$
where $\hat{\varphi}^{-i}$ is the panel excluding agent $i$.

### Bond market clearing enforcement
- [ ] Code implementation
1. draw $a^{i}$

### Order shuffling problem
Permutation data? Permutation invariant?

### Price-taking behavior (explicit)
The model is closed with prices that exclude own state:
$$
r^i = r(z,\pi,\hat{\varphi}^{-i}),\qquad
w^i = w(z,\pi,\hat{\varphi}^{-i}),\qquad
\Pi^i = \Pi(z,\pi,w^i,\hat{\varphi}^{-i}).
$$
This means when evaluating agent $i$'s residual, $a^i,n^i$ do not directly move prices.

### MRS wage closure
We use MRS closure with a parametric labor-disutility derivative $v'(N)=\chi N^\nu$:
$$
w(z,\pi,\hat{\varphi}^{-i})
=\frac{\chi\,[N^*(\hat{\varphi}^{-i})]^{\nu}}
{\mathbb{E}_{\hat{\varphi}^{-i}}[u'(c_j)]},
\qquad
N^*(\hat{\varphi}^{-i})=\mathbb{E}_{\hat{\varphi}^{-i}}[n_j].
$$
with $u'(c)=c^{-\gamma}$.

### Residual in finite-agent form
With $c^i=(\hat W^i)^{-1/\gamma}$ and
$$
s^i=r^i a^i+w^i n^i-c^i+\Pi^i,
$$
the focal residual is
$$
\mathcal{R}^i=(r^i-\rho)\hat W^i+\psi'(a^i)I_{\{a^i\le a_{lb}\}}
+\mathcal{L}_x^i\hat W+\mathcal{L}_{z,\pi}^i\hat W+\mathcal{L}_g^i\hat W.
$$
where
$$
\mathcal{L}_x^i\hat W
=\hat W_{a^i}s^i+\lambda(n^i)\big(\hat W(a^i,\check n^i,\cdot)-\hat W(a^i,n^i,\cdot)\big),
$$
$$
\mathcal{L}_{z,\pi}^i\hat W
=\mu_z\hat W_z+\tfrac12\sigma_z^2\hat W_{zz}
+\mu_\pi\hat W_\pi+\tfrac12\sigma_z^2\pi^2\hat W_{\pi\pi}
-\sigma_z^2\pi \hat W_{z\pi},
$$
$$
\mathcal{L}_g^i\hat W
=\sum_{j\ne i}\Big[\hat W_{a^j}s^j+\lambda(n^j)\big(\hat W|_{n^j\to\check n^j}-\hat W\big)\Big].
$$

### Closure count (unknowns vs equations)
At each sampled state, unknowns are:
1. $\hat W^i$ from the neural approximation.
2. $w^i$ from MRS closure.
3. Policies $c^j$ implied by $\hat W^j$ for panel agents.

Equations are:
1. Master-equation residual $\mathcal{R}^i=0$ (enforced in loss).
2. Wage closure equation above.
3. Policy inversion $c^j=(\hat W^j)^{-1/\gamma}$.

So the finite-agent training problem is closed pointwise in sampled states.

## Finite Agent Approximation Sampling Strategy
We keep adaptive interval sampling in own assets $a^i$:
1. Partition $[a_{\min},a_{\max}]$ into intervals.
2. Sample $z,\pi$ uniformly on configured ranges.
3. Draw panel states $(a_j,n_j)$ for others and oversample near borrowing region.
4. Reweight intervals over training by relative residual scores.

## Computing Itô terms

### Baseline Auto-diff
In the baseline model, aggregate Itô terms are computed explicitly with full auto-diff:
$\hat W_z,\hat W_{zz},\hat W_\pi,\hat W_{\pi\pi},\hat W_{z\pi}$.
The required $-\sigma_z^2\pi\hat W_{z\pi}$ cross term naturally appears in the residual via the manual combination of these components. This approach is highly expressive but requires many distinct backward passes per batch over all $N$ dimensions.

### Duarte Itô Trick (`HANKMRSItoResidual`)
To vastly accelerate computation and remove the independent asset gradients, the codebase employs an adaptation of the one-dimensional auxiliary function trick from Duarte et al. (2021). Since $z$ and $\pi$ are driven by the *same* Brownian shock $dW^0$, their diffusion parameters map to a single scalar variable $\epsilon$:
$\Sigma_z = \sigma_z$ and $\Sigma_\pi = -\sigma_z \pi$.

We define an auxiliary function $F(\epsilon)$ evaluated near zero:
$$F(\epsilon) = \hat{W}\left(a + \frac{1}{2}\epsilon^2 s, \ z + \frac{\epsilon}{\sqrt{2}}\Sigma_z + \frac{1}{2}\epsilon^2 \mu_z, \ \pi + \frac{\epsilon}{\sqrt{2}}\Sigma_\pi + \frac{1}{2}\epsilon^2\mu_\pi\right)$$

Taking its analytical second derivative $F''(0)$ exactly generates the full collection of continuous drifts and covariance multi-directional diffusion components:
$$ F''(0) = \underbrace{ \sum_i s_i \partial_{a_i}\hat{W} + \mu_z \partial_z \hat{W} + \mu_\pi \partial_\pi \hat{W} }_{\text{Full continuous drift terms}} + \underbrace{ \frac{1}{2} \sigma_z^2 \partial_{zz} \hat{W} + \frac{1}{2} (-\sigma_z \pi)^2\partial_{\pi\pi} \hat{W}  - \sigma_z^2 \pi \partial_{z\pi} \hat{W} }_{\text{Covariance multi-directional diffusion terms}} $$

## Training Optimization Potential & Feature Roadmap
As observed, running the master equation directly scales poorly for large $N$ (currently $O(N^2)$ due to nested evaluations excluding individual agents sequentially). We can stage a sequence of algorithmic optimizations to massively accelerate training.

### 1. $O(N)$ Vectorized State Swapping & Price Computation
**Current Bottleneck:** The focal residual requires swapping every agent $j$ into the focal position $0$ to compute its drift $s_j$. To compute the price $w^j$, it internally swaps *every other agent* $k \neq j$ into the focal position. This creates an $O(N^2)$ sequence of sequential model evaluations.
**Optimization:** 
- Tile the batch tensor to size $B \times N$. In slice $k$, swap agent $k$ into position $0$.
- Evaluate all $\hat{W}^k$ simultaneously with a single massive batched NN forward pass. 
- Use the identity $\sum_{k \neq j} f(k) = \sum_{all} f(k) - f(j)$. Since we evaluated all $N$ agents, we can calculate the focal-excluding price for *any* agent $j$ by taking the global sum of marginal utilities and subtracting agent $j$'s contribution. 
- The Poisson jump states $\hat{W}_{n_j \to \check{n}_j}$ can similarly be evaluated in parallel across all agents by flipping the focal employment bit on the $B \times N$ batched tensor.
- **Estimated impact:** Reduces NN evaluations from $\sim 600$ per batch to exactly $2$ large batched evaluations, providing an estimated $10-20\times$ speedup per step.

### 2. DeepSets for Permutation Invariance
**Current Bottleneck:** The input to the MLP is a flat concatenated vector of $(a_0, \dots, a_{N-1}, n_0, \dots, n_{N-1})$. The network must waste capacity learning that the exact ordering of agents $1$ through $N-1$ does not matter.
**Optimization:**
- Refactor the MLP into a DeepSet architecture.
- Pass focal $(a_0, n_0)$ directly, compute embeddings $h(a_j, n_j)$ for others, pool them via a permutation-invariant operation like sum or mean to form $H$, and decode via $\Phi(a_0, n_0, z, \pi, H)$.
- **Estimated impact:** Faster learning curves, zero permutation-overfitting, and the ability to train on one $N$ and test on a larger $N$.

### 3. JIT Compilation / `torch.compile`
**Current Bottleneck:** Complex Python-level tensor manipulation over batches causes considerable dispatch overhead, especially during the Itô trick's auto-diff passes.
**Optimization:**
- Wrap the focal PDE residual trace in `torch.compile` (available in newer PyTorch versions) or use `jax.jit` if porting to JAX to fuse the automatic differentiation graphs.

[Deep HANK - Code Model 3 - Archive](Deep%20HANK%20-%20Code%20Model%203%20-%20Archive)
