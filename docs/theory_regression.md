# Section IV-C: Machine Learning Regressors & Localization Theory

**Author:** Member 4 — Regression and Validation Lead  
**Scope:** Phase 1 Coordinate Localization (`LONGITUDE`, `LATITUDE` in metres)

---

## 1. Mathematical Formulation of Wi-Fi Fingerprinting

Indoor Wi-Fi localization operates by mapping an observed vector of Received Signal Strength Indicators (RSSI) $\mathbf{s} \in \mathbb{R}^D$ to spatial Cartesian coordinates $\mathbf{y} = [x, y]^T \in \mathbb{R}^2$ (representing `LONGITUDE` and `LATITUDE` in metres).

### $k$-Nearest Neighbors ($k$-NN) Spatial Fingerprinting
The fundamental baseline in indoor positioning (RADAR, Bahl & Padmanabhan, 2000) posits that locations close in physical space exhibit similar signal propagation patterns. Given a query scan $\mathbf{s}^*$ and a reference training dataset $\mathcal{D} = \{(\mathbf{s}_i, \mathbf{y}_i)\}_{i=1}^N$:

1. Compute distance in feature space:
   $$d_i = \|\mathbf{s}^* - \mathbf{s}_i\|_p = \left( \sum_{j=1}^D |s^*_j - s_{ij}|^p \right)^{1/p}$$
   where $p=2$ corresponds to Euclidean distance and $p=1$ to Manhattan distance.

2. Identify the $k$ nearest training samples $\mathcal{N}_k(\mathbf{s}^*)$ with smallest $d_i$.

3. Estimate coordinates using inverse-distance weighting:
   $$\hat{\mathbf{y}} = \sum_{i \in \mathcal{N}_k(\mathbf{s}^*)} w_i \mathbf{y}_i, \quad \text{where } w_i = \frac{1 / (d_i + \epsilon)}{\sum_{j \in \mathcal{N}_k(\mathbf{s}^*)} 1 / (d_j + \epsilon)}$$
   For uniform weighting, $w_i = \frac{1}{k}$.

---

## 2. Structural Challenges in the RSSI Feature Space

Fingerprint regression is non-trivial due to the physical realities of radio frequency (RF) propagation in indoor environments:

1. **Log-Distance Path Loss & Non-Linearity:**  
   Signal attenuation follows:
   $$P_r(d) = P_r(d_0) - 10 n \log_{10}\left(\frac{d}{d_0}\right) + X_\sigma$$
   Because received power decays logarithmically with distance $d$, a $\Delta \text{RSSI} = 2\text{ dBm}$ near the transmitter represents centimetres of movement, whereas the same $\Delta \text{RSSI}$ at $-85\text{ dBm}$ corresponds to tens of metres. Linear Euclidean distance in raw RSSI fails to reflect this scale variance.
2. **Multipath, Shadowing, and Dynamic Obstacles:**  
   Walls, reinforced concrete floors, metal furniture, and human traffic induce fast and slow fading ($X_\sigma \sim \mathcal{N}(0, \sigma^2)$ with $\sigma \in [4, 12]\text{ dB}$), causing identical physical coordinates to emit fluctuating signal vectors over time.
3. **High Dimensionality and Extreme Sparsity:**  
   With hundreds of installed Wireless Access Points (WAPs), typical mobile devices detect only 5–25 APs per scan ($<5\%$ density). Unobserved APs (encoded as $+100$ in raw UJI, mapped to $-110\text{ dBm}$ / $0.0$) dominate the vector, requiring robust distance metrics that do not conflate non-detection with active zero-power signals.
4. **Client Device Heterogeneity:**  
   Varying receiver antenna gains and operating systems introduce systematic offsets across `PHONEID` and `USERID`.

---

## 3. Loss Functions: MSE vs. Huber Loss

In spatial coordinate regression, the loss function dictates how the model balances central accuracy against catastrophic spatial outliers:

### Mean Squared Error (MSE / $L_2$)
$$L_{\text{MSE}}(\mathbf{y}, \hat{\mathbf{y}}) = \frac{1}{2} \|\mathbf{y} - \hat{\mathbf{y}}\|_2^2 = \frac{1}{2} \left[ (x - \hat{x})^2 + (y - \hat{y})^2 \right]$$
* **Behavior:** The gradient $\nabla_{\hat{\mathbf{y}}} L_{\text{MSE}} = -(\mathbf{y} - \hat{\mathbf{y}})$ scales linearly with residual error.
* **Limitation:** In the presence of noisy Wi-Fi scans or cross-building anomalies, squaring large errors excessively penalizes outliers, causing tree ensembles to over-smooth coordinates toward the global spatial mean.

### Huber Loss (Smooth $L_1$)
$$L_{\delta}(r) = \begin{cases} \frac{1}{2} r^2 & \text{for } |r| \le \delta \\ \delta (|r| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$
* **Advantage:** Huber loss transitions from quadratic behavior for small errors to linear penalization for residuals exceeding threshold $\delta$. This bounds the influence of anomalous Wi-Fi bursts and preserves sharp spatial localization near partition boundaries.

---

## 4. Bias–Variance Trade-Off in Hierarchical Routing

We systematically compare three spatial routing paradigms:

```
[Wi-Fi Scan] 
     │
     ├──> Global Regressor ───────────────> (x, y) [Campus-wide single model]
     │
     ├──> Building Route (B) ─────────────> (x, y) [Building-specific model]
     │
     └──> Building (B) + Floor (F) Route ─> (x, y) [Floor-slab model + fallbacks]
```

1. **Global Regressors:**
   * **High Structural Bias, Low Variance:** A single model must capture coordinates spanning multiple non-contiguous buildings across hundreds of metres. While trained on all 19,300 samples, it struggles to learn sharp spatial boundaries separating discrete architectural structures.
2. **Per-Building Regressors:**
   * **Balanced Bias and Variance:** Restricting training to scans within a specific building footprint eliminates cross-campus interference and aligns coordinates with local building geometry.
3. **Per-Building-and-Floor Regressors:**
   * **Minimum Structural Bias, Higher Variance Risk:** Specializing models to individual floor slabs isolates 2D coordinates to single plane footprints (~$80\text{m} \times 60\text{m}$). However, smaller sample sizes per floor slab ($N \approx 700\text{--}2,700$) can increase variance if over-parameterized. A hierarchical fallback mechanism ensures robustness when test scans encounter rare or unrepresented floor combinations.

### Oracle Routing vs. End-to-End Routing
* **Oracle Routing Diagnostic:** Routes scans using true `BUILDINGID` and `FLOOR`. This measures pure coordinate regression capability independent of upstream classification errors.
* **End-to-End Routing:** Routes scans using Member 3's out-of-fold predicted building and floor. Because building classification achieves $>99.4\%$ accuracy, building-level routing suffers negligible degradation. For floor-level routing, upstream floor misclassifications ($~75\%$ accuracy) test the model's resilience to wrong-floor spatial projections.
