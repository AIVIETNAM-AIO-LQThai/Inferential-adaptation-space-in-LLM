# Inferential adaptation space in LLM

## Set up

`git clone https://github.com/AIVIETNAM-AIO-LQThai/Inferential-adaptation-space-in-LLM.git`

`cd Inferential-adaptation-space-in-LLM`

`python -m pip install -r requirements.txt`

`hf download sbordt/OLMo-2-1B-1x --local-dir checkpoints/olmo2-wd01`

`hf download sbordt/OLMo-2-1B-1x-WD1 --local-dir checkpoints/olmo2-wd1`

## Research motivation

Pretraining determines more than the knowledge stored by a large language model. It may also determine how that knowledge can be modified during subsequent learning. In [_Weight Decay Improves Language Model Plasticity_](https://arxiv.org/html/2602.11137v2), Han, Bordt, Zhang and Kakade (2026) report that increasing pretraining weight decay (WD) can improve downstream performance after fine-tuning, even when the resulting base model has worse pretraining validation loss. They investigate representation separability, attention regularization and pretraining overfitting as possible explanations.

We investigate a complementary explanation from the perspective of **optimization and functional geometry**: weight decay may establish a distribution of parameter magnitudes that changes how different components of a pretrained model respond to subsequent optimization. The resulting changes in the model's predictions may involve not only a different learning speed, but also a different direction of adaptation.

The distinction is important. A pretrained model's current predictive performance, the function it represents, and its capacity to adapt are not equivalent properties. Two models with similar pretraining losses need not compute the same function. Conversely, two parameterizations of the same function can exhibit different learning dynamics under the same optimizer.

## Research questions

| Research question                                                                                         | Investigation                                              |
| --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **RQ1**. What changes in parameter space as a consequence of pretraining weight decay?                    | [Checkpoint analysis](huggingface.co/collections/th135/pt) |
| **RQ2**. How do these differences influence the magnitude and direction of subsequent functional updates? | Mathematical formulation and controlled interventions      |
| **RQ3**. Can these properties predict downstream plasticity and retention?                                | Fine-tuning and evaluation                                 |

## Investigation A — Weight decay and inherited adaptation geometry

**Notebook:** [`weight_decay_adaptation_geometry.ipynb`](notebooks/weight_decay_adaptation_geometry.ipynb)

**Status:** Experimental pipeline implemented; full checkpoint experiments and empirical conclusions pending.

The objective is to determine whether the parameter norm profile inherited from pretraining influences downstream learning through the relative optimization responses of model components. We separate the existence of this mechanism from its usefulness:

1. Establish whether a native plasticity gap exists at the investigated task and training horizon.
2. Determine whether inherited parameter magnitudes can be isolated through function-preserving transformations and corresponding optimizer interventions.
3. Investigate whether the resulting functional updates can be reproduced by global learning-rate adjustment or require finer component-wise scaling.
4. Determine whether the non-global changes improve held-out learning or the adaptation–retention trade-off.

### 1. Background and mathematical motivation

#### 1.1 Weight decay changes parameter magnitude profiles

Figures 29–30 of Han et al. report changes in parameter magnitudes and relative norm contributions across the architecture. In particular, input embeddings are reported to be excluded from weight decay. An increase in a component's share of the model norm does not necessarily imply an increase in its absolute norm: other components may simply contract more strongly.

Let $W_i^{(\lambda)}$ be parameter block $i$ of a model pretrained with weight decay $\lambda$. Define

$$
r_i^{(\lambda)} = \|W_i^{(\lambda)}\|_F, \qquad U_i^{(\lambda)} = \frac{W_i^{(\lambda)}}{r_i^{(\lambda)}}.
$$

For a recipient checkpoint $s$ and a donor checkpoint $h$, the norm ratio is

$$
c_i=\frac{r_i^{(h)}}{r_i^{(s)}}.
$$

By definition, $r_i^{(h)}=c_i r_i^{(s)}$. However, this does not imply $W_i^{(h)}\approx c_iW_i^{(s)}$: the normalized directions, spectral structures, and represented functions of independently pretrained checkpoints can differ.

A single positive scalar contraction preserves the normalized singular-value distribution and rank of a matrix. Any change in normalized spectral structure over training must therefore involve the accumulation and interaction of optimizer updates or other training differences. We defer extensive weight-spectrum and effective-attention analyses to a separate investigation because they do not directly identify the norm-mediated mechanism.

#### 1.2 Why magnitudes influence subsequent learning

AdamW updates a parameter block according to

$$
W_{t+1}=(1 - \eta_t \lambda)W_t - \eta_t A_t,
\qquad A_t=\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon_A}.
$$

The decay term contracts the weights, while the adaptive term moves them according to the gradient history. Their interaction determines the final parameter magnitude. [Kosson, Messmer and Jaggi (2024)](https://proceedings.mlr.press/v235/kosson24a.html) provide a theoretical connection between weight decay, weight norms and angular updates in normalized networks.

Consider a parameter block as a vector $w=ru$, where $r=\|w\|$ and $\|u\|=1$. The first-order change in its normalized direction is

$$
\Delta u\approx\frac{1}{r}(I-uu^\top)\Delta w.
$$

Substituting the AdamW update gives

$$
\Delta u\approx-\frac{\eta}{r}(I-uu^\top)A.
$$

The explicit WD term vanishes to first order because $(I-uu^\top)w=0$. Nevertheless, weight decay can influence future angular updates by changing the inherited radius $r$. It can also influence finite-step angular movement through its interaction with the adaptive update.

For example, let $w_L=(10,0)$ and $w_H=(2,0)$. Both point in the same direction. Given the same tangential update $(0,1)$, their angular displacements are $5.71^\circ$ and $26.57^\circ$, respectively. For a function depending only on the normalized direction of $w$, these updates produce different functional changes despite having identical absolute parameter displacements.

Under idealized Adam for an exactly scale-invariant parameter, $A(cw)\approx A(w)$, so $\eta/r$ acts as a norm-dependent angular learning-rate coefficient. It is not the actual angular displacement: the latter also depends on the tangential component of $A$. Independently pretrained checkpoints need not share that component.

#### 1.3 Why a larger global learning rate may be insufficient

If high WD simply reduced every relevant parameter magnitude by the same positive factor, increasing the fine-tuning LR of a low-WD model might reproduce the corresponding angular learning dynamics. However, the required compensation may vary across components.

Under the idealized Adam scaling relationship, the candidate multiplier for a recipient block is

$$
a_i = \frac{1}{c_i} = \frac{r_i^{(s)}}{r_i^{(h)}}.
$$

For example, consider four hypothetical scale-invariant blocks:

| Component         | Recipient norm | Donor norm | Candidate LR multiplier |
| ----------------- | -------------: | ---------: | ----------------------: |
| Query projection  |            100 |         25 |                    4.00 |
| Key projection    |            100 |         50 |                    2.00 |
| Value projection  |            100 |         80 |                    1.25 |
| MLP up-projection |            100 |         40 |                    2.50 |

No single scalar matches these four coefficients simultaneously. Moreover, even uniform scaling among the eligible blocks could accelerate those blocks relative to ineligible parameters, such as the gate projection and normalization gains.

Appendix E.2 of Han et al. reports a sweep of pretraining WD, fine-tuning WD, fine-tuning LR and batch size, with learning rates $10^{-5}$, $3\times10^{-5}$ and $6\times10^{-5}$. The reported high-WD advantage motivates investigating alternatives to global acceleration, but a finite grid does not rule out a better global LR outside or between its tested values. We therefore compare norm-derived compensation predictions against the grid and independently calibrate learning rates in our local experiments.

**Working hypothesis:** pretraining WD may alter the relative optimization geometry of model components. The resulting heterogeneous functional contributions may change the direction of adaptation, not just its speed. Whether the resulting changes are beneficial is an independent empirical question.

### 2. Isolating the inherited norm profile

To separate the norm-mediated mechanism from differences in the learned function, we intervene within a single recipient checkpoint. For each candidate scale-invariant block, construct the physical norm transplant

$$
\widetilde W_i^{(s)}=c_i W_i^{(s)}.
$$

The transplant preserves the recipient's normalized weight direction while assigning it the donor's norm. We do not copy the donor's weights or representations. The primary candidate families in OLMo-2 are the query, key, value and output projections and the MLP up/down projections. Their approximate scale invariance depends on the placement of RMSNorm; finite normalization epsilon and numerical precision can break it. Noneligible parameters receive $c_i=1$.

For exact scale invariance, differentiating $L(cw)=L(w)$ gives $g(cw)=g(w)/c$. With fresh Adam moments,

$$
A(g/c)=\frac{g}{|g|+c\epsilon_A}.
$$

This explains why LR-only emulation is accurate only when optimizer epsilon is negligible for relevant gradient coordinates. For a fixed positive rescaling, the exact AdamW transport laws under matched histories and no incompatible clipping are:

| Intervention           | Weights   | LR         | Adam epsilon     | Fine-tuning WD              |
| ---------------------- | --------- | ---------- | ---------------- | --------------------------- |
| Baseline               | $W_i$     | $\eta$     | $\epsilon_A$     | $\lambda_{\mathrm{FT}}$     |
| Physical transplant    | $c_i W_i$ | $\eta$     | $\epsilon_A$     | $\lambda_{\mathrm{FT}}$     |
| Exact emulation        | $W_i$     | $\eta/c_i$ | $c_i\epsilon_A$  | $c_i\lambda_{\mathrm{FT}}$  |
| Compensated transplant | $c_iW_i$  | $c_i\eta$  | $\epsilon_A/c_i$ | $\lambda_{\mathrm{FT}}/c_i$ |
| LR-only emulation      | $W_i$     | $\eta/c_i$ | $\epsilon_A$     | $\lambda_{\mathrm{FT}}$     |

We expect the physical transplant and exact emulation to follow equivalent functional trajectories under the stated assumptions. The compensated transplant should reproduce the baseline. These are validation experiments, not presumed empirical findings. Initial predictive KL, gradient transformation errors, transported parameter errors and trajectory differences determine whether the gauge interpretation is defensible. If it fails, an LR-only intervention remains a valid optimizer experiment but cannot be described as a validated function-preserving transplant.

### 3. From parameter-space updates to functional adaptation geometry

#### 3.1 Component-wise functional contributions

Let $z = f_\theta(X)$ denote the logits on fixed evaluation contexts. For a sufficiently small parameter update $\Delta\theta$, the first-order functional change is

$$
\Delta z \approx J_\theta(X) \Delta \theta = \sum_{i=1}^m J_i\Delta W_i.
$$

Define $d_i=J_i\Delta W_i$ and collect the contributions as columns of $D=[d_1,\ldots,d_m]$. The baseline functional direction is $D\mathbf 1$. If we scale each block's next optimizer update by $a_i$, the first-order direction becomes $Da$.

Heterogeneous coefficients are not sufficient to guarantee a different functional direction: the individual block contributions may be collinear or their changes may cancel. To test this, we must measure how those contributions interact in output space.

#### 3.2 Fisher geometry

Raw logit distance can be misleading because adding the same scalar to every vocabulary logit leaves the predictive distribution unchanged. For categorical probabilities $p=\operatorname{softmax}(z)$, define the logit Fisher matrix

$$
F_z=\operatorname{diag}(p) - pp^\top.
$$

For a small logit perturbation $d$, predictive KL has the local approximation

$$
D_{\mathrm{KL}}(p(z) \Vert p(z+d)) \approx \frac{1}{2} d^\top F_zd.
$$

Average the Fisher inner products over fixed, valid teacher-forced token contexts, and write the corresponding block contribution Gram matrix as

$$
G=D^\top FD.
$$

The notebook computes these inner products as probability-weighted covariances without constructing a dense vocabulary Fisher matrix or full parameter Jacobian. Directional derivatives are approximated using central finite differences with convergence and aggregate reconstruction diagnostics. Centered-logit geometry and actual finite-step KL provide sensitivity checks.

The resulting Gram matrix describes the selected optimizer's realized block contributions on sampled contexts, not every possible direction in the model's functional tangent space. For exact emulation that changes the direction within a parameter block, an augmented contribution basis is used rather than assuming scalar LR reweighting is exact.

#### 3.3 The limits of global compensation

A positive global LR multiplier can reproduce the local LR-only profile if $Da$ and $D\mathbf1$ are positively collinear in Fisher-weighted output space. For a nonzero baseline Fisher direction, the best nonnegative multiplier is

$$
k^*=\max\left(0,\frac{\mathbf1^\top Ga}{\mathbf1^\top G\mathbf1}\right).
$$

The irreducible relative functional discrepancy is

$$
E_F=\sqrt{\frac{(a-k^*\mathbf{1})^\top G(a-k^*\mathbf{1})}{a^\top Ga}}.
$$

To investigate which combinations of component-wise changes produce this discrepancy, define the residual Gram matrix

$$
Q=G-\frac{G\mathbf{1} \space \mathbf{1}^\top G}{\mathbf{1}^\top G\mathbf{1}}.
$$

For the unconstrained projection, or when its multiplier is nonnegative, let $a=\bar a\mathbf{1} + \delta$. Since $Q\mathbf{1}=0$,

$$
E_F^2=\frac{\delta^\top Q\delta}{a^\top Ga}.
$$

If the unconstrained multiplier is negative, the best nonnegative fit is zero and the relative residual is one. This identity connects heterogeneity in the multiplier profile to the functional sensitivity of the model: a large variation in norms need not produce a large directional change if the variation lies in directions to which $Q$ is insensitive.

Additional diagnostics include the cancellation ratio $\kappa = \mathbf{1}^\top G \mathbf{1} / \operatorname{tr}(G)$, the local angular sensitivity $\chi = \sqrt{\lambda_{\max}(Q)/(\mathbf{1}^\top G \mathbf{1})}$, and the eigenmode contributions $\lambda_j(v_j^\top\delta)^2$ for eigenpairs of $Q$. All depend on the chosen block partition, task and optimizer state.

#### 3.4 From global to blockwise compensation

We compare global, eligible-only, two-group, family-level and measured blockwise profiles. In the fitted hierarchy, let $B$ assign parameter blocks to groups of shared multipliers. We solve

$$
\beta^* = \arg\min_{\beta \geq 0}(a - B\beta)^\top G(a - B\beta).
$$

The nested hierarchy necessarily improves the in-sample approximation as the allowed profile becomes more flexible. Its scientific value comes from determining whether the simpler profiles reproduce the measured profile on independent gradient batches and contexts. The blockwise target has zero self-reconstruction error by definition.

Shuffled profiles preserve the distribution of multipliers while disrupting their assignment across the architecture. We use both shuffling across all eligible blocks and shuffling within projection families. They form conditional reference distributions, not formal permutation tests without exchangeability assumptions.

### 4. Useful adaptation: does changing the direction help?

A functionally distinct direction is not necessarily a better one. On held-out downstream contexts, let $g_D=\nabla_z L_D$, and define

$$
b_i = -g_D^\top d_{D,i}.
$$

Then the predicted first-order loss reduction for profile $a$ is $U_D(a) = b^\top a$. The non-global residual utility is $b^\top(a - k^*\mathbf1)$, which measures the difference relative to the best geometric global projection. This is not necessarily an advantage at matched functional movement.

Define the downstream and retention Grams $G_D=D_D^\top F_DD_D$ and $G_P = D_P^\top F_P D_P$, computed from the same training update evaluated on different context distributions. The local efficiency measures are

$$
\mathcal E_D(a) = \frac{b^\top a}{\sqrt{a^\top G_Da}},
\qquad
\mathcal E_P(a) = \frac{b^\top a}{\sqrt{a^\top G_Pa}}.
$$

These quantities measure local downstream improvement per unit of downstream predictive movement or retention-domain disruption. They are undefined for zero denominators and should not be interpreted as long-horizon plasticity metrics. A larger score is useful only under a meaningful, matched comparison.

We test the predictions using finite parameter steps, held-out loss and realized predictive KL over a shared movement range. We also measure the actual change in retention cross-entropy, because predictive drift does not necessarily imply harmful forgetting.

### 5. Experimental design

#### Checkpoints

All five checkpoints are drawn from the [released pretrained collection](https://huggingface.co/collections/th135/pt), using the 20-token-per-parameter regime. The WD labels follow the study and the current notebook mapping; exact run provenance is recorded separately.

| Pretraining WD | Hugging Face repository                                              | Role                                 |
| -------------: | -------------------------------------------------------------------- | ------------------------------------ |
|            0.0 | [OLMo-2-1B-1x-WD0](https://huggingface.co/sbordt/OLMo-2-1B-1x-WD0)   | No-decay static reference            |
|            0.1 | [OLMo-2-1B-1x](https://huggingface.co/sbordt/OLMo-2-1B-1x)           | L: standard baseline                 |
|            0.3 | [OLMo-2-1B-1x-WD03](https://huggingface.co/sbordt/OLMo-2-1B-1x-WD03) | Intermediate static reference        |
|            0.6 | [OLMo-2-1B-1x-WD06](https://huggingface.co/sbordt/OLMo-2-1B-1x-WD06) | M: reported pretraining-loss optimum |
|            1.0 | [OLMo-2-1B-1x-WD1](https://huggingface.co/sbordt/OLMo-2-1B-1x-WD1)   | H: reported CoT-performance optimum  |

L–H is the high-contrast mechanism comparison. M–H investigates the reported trade-off between pretraining loss and CoT performance. Both comparisons are bidirectional: L receives H's profile and vice versa; M receives H's profile and vice versa. The physical and optimizer interventions preserve the recipient as the starting point, rather than treating independently trained checkpoints as functionally equivalent.

#### Datasets and protocol

| Role                             | Dataset                                                                                      |
| -------------------------------- | -------------------------------------------------------------------------------------------- |
| Mathematical adaptation          | [SimpleScaling full59K](https://huggingface.co/datasets/simplescaling/data_ablation_full59K) |
| Reading-comprehension adaptation | [RACE](https://huggingface.co/datasets/ehovy/race)                                           |
| General-text retention proxy     | [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext)                          |
| Auxiliary generation diagnostic  | [GSM8K](https://huggingface.co/datasets/openai/gsm8k), numeric exact match                   |

The local protocol uses separate gradient-generation, calibration and held-out evaluation partitions. Intervention arms share training batches, tokenization, loss masks and evaluation contexts. The default configuration uses 500 optimization steps, effective batch size 16, maximum sequence length 512, three trajectory seeds and eight diagnostic repeats. The native LR calibration grid is $\{10^{-5},3\times10^{-5},6\times10^{-5},10^{-4}\}$. The primary clean optimizer experiment uses zero fine-tuning decay and no clipping; nonzero decay and clipping are included in the optimizer identity sensitivity tests.

This is a controlled local study with a constant-LR protocol, not an exact reproduction of the original paper's SFT recipe, filtered data, or benchmark scores. WikiText-103 is a proxy, not the original pretraining distribution.

#### Native adaptation anchor

Before interpreting the functional mechanism as an explanation for plasticity, measure native learning curves for L, M and H. Define

$$
P_s(t) = L_{D,s}(0) - L_{D,s}(t), \qquad
\Delta_{H,s}(t) = P_H(t) - P_s(t).
$$

Track both loss improvement and absolute downstream loss, alongside retention changes and the auxiliary GSM8K diagnostic. Compare shared and separately calibrated global LRs. The native contrast establishes the relevance of the mechanism at the selected horizon and calibrates the practical margin for trajectory utility. If the gap is absent or inconclusive, local geometric results remain valid as a separate investigation, but cannot be claimed to explain a native advantage at that horizon.

### 6. Hypotheses and decision criteria

| Question              | Null or reference model                                                               | Evidence                                              |
| --------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| **Native anchor**     | No practically meaningful native short-horizon advantage                              | Paired validation-loss trajectories                   |
| **H1a-local**         | Global scaling reproduces the local functional direction                              | Fisher residual and numerical derivative uncertainty  |
| **H1a-performance**   | Tuned global LR suffices for downstream learning                                      | Calibrated LR curves and short learning trajectories  |
| **H1b-selective**     | Eligible/noneligible group scaling suffices                                           | Held-out two-group approximation residual             |
| **H1b-heterogeneous** | A simpler architectural family profile suffices                                       | Held-out compensation hierarchy                       |
| **H1b-specificity**   | The inherited arrangement is no more useful than alternative assignments              | Shuffled-profile references and matched-step controls |
| **H1b-utility**       | No advantage at matched downstream movement                                           | Local utility and finite-step loss reduction          |
| **H1b-retention**     | No advantage at matched retention disruption                                          | Held-out loss and predictive KL trade-off             |
| **H1b-trajectory**    | No practically meaningful advantage over calibrated controls across the local horizon | Paired short trajectories and stateful geometry       |

We distinguish meaningful differences, approximate equivalence and inconclusive outcomes. Numerical validation tolerances are separate from scientific effect-size margins. Parameter blocks, layers and individual token positions are not independent pretraining replicates. Uncertainty is estimated across paired diagnostic repeats or fine-tuning seeds; the default three trajectory seeds give only descriptive conditional uncertainty, not robust population-level inference.

### 7. Unified notebook structure

| Section                             | Purpose                                                          | Main measurements and visualizations                                                                          |
| ----------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **0a. Provenance and setup**        | Establish comparable inputs and protocol                         | Checkpoint and data manifests, model configuration inventory                                                  |
| **0b. Native adaptation anchor**    | Establish the phenomenon at the investigated horizon             | Native LR calibration, loss and retention trajectories, gap estimates                                         |
| **I. Static parameter geometry**    | Define donor–recipient intervention profiles                     | Frobenius norms, weight RMS, squared norm shares, log-ratio heatmaps, WD dose–response, effective LM head     |
| **II. Historical norm dynamics**    | Provide limited theoretical context                              | Integrated decay exposure and initialization attenuation; no equilibrium identification from endpoint weights |
| **III. Published LR evidence**      | Audit global-compensation coverage                               | Existing LR grid versus candidate multipliers; optional verified sweep re-evaluation                          |
| **IV. Symmetry validation**         | Establish function-preserving intervention validity              | Predictive KL, logit and residual-stream errors, RMSNorm epsilon diagnostics                                  |
| **V. Inherited optimizer dynamics** | Validate the scale and optimizer transformation laws             | Gradient scaling, Adam epsilon ECDFs, angular amplification, optimizer-identity tests                         |
| **VI. Fisher functional geometry**  | **Core:** determine how the profile changes functional direction | Fine-grained Fisher/centered-logit Grams, $Q$, eigenspectra, cancellation, derivative convergence             |
| **VII. Compensation hierarchy**     | **Core:** identify how much profile structure is required        | Held-out global, selective, two-group and family residuals; shuffled reference distributions                  |
| **VIII. Utility and retention**     | **Core:** determine whether functional changes help              | Residual utility, movement-matched efficiencies, actual KL and held-out loss trade-offs                       |
| **IX. Short trajectories**          | Test persistence and bidirectional effects                       | Calibrated profile interventions, stateful Adam geometry, 500-step adaptation/retention curves                |
| **X. Evidence synthesis**           | Decide which hypotheses survive                                  | Conditional uncertainty intervals, evidence ledger and research decisions                                     |

Section II is intentionally limited. For AdamW, the exact squared-norm recurrence contains the cross term $-2\eta_t (1-\eta_t\lambda) \langle W_t,A_t \rangle_F$, which generally cannot be neglected merely because a block is scale-invariant. Final checkpoints do not identify its historical value. Decay exposure measures attenuation of the initialization term, not its share of the final norm. The optional published-sweep re-evaluation in Section III is disabled until the repository-to-hyperparameter mapping is verified.

### 8. Experimental outputs

All outputs are written under `results/investigation_a/`. Principal artifacts include:

| Output                                                                                     | Content                                                               |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| `experiment_config.json`, `checkpoint_manifest.json`, `data_manifest.json`                 | Settings, resolved model revisions and dataset partition IDs          |
| `native_lr_calibration.csv`, `native_adaptation_anchor.csv`, `anchor_decisions.csv`        | Native learning curves and relevance decisions                        |
| `parameter_geometry.csv`, `norm_emulation_profiles.csv`                                    | Parameter inventory statistics and donor–recipient profiles           |
| `symmetry_validation.csv`, `optimizer_identity_tests.csv`                                  | Function preservation and optimizer transport validation              |
| `finite_difference_validation.csv`, `grams/`, `gram_manifest.csv`                          | Numerical derivative diagnostics and per-repeat Gram NPZ artifacts    |
| `functional_geometry.csv`, `residual_eigenmodes.csv`                                       | Fisher amplification, alignment, residuals and mode contributions     |
| `compensation_hierarchy.csv`, `hierarchy_coefficients.json`, `profile_randomization.csv`   | Calibration fits and held-out compensation results                    |
| `local_linear_utility.csv`, `local_adaptation_retention.csv`, `finite_matched_utility.csv` | Local and finite-step downstream utility at comparable budgets        |
| `short_trajectory_analysis.csv`, `trajectory_geometry.csv`                                 | Learning trajectories and evolving optimizer-state-dependent geometry |
| `evidence_ledger.csv`, `investigation_a_summary.md`                                        | Final conditional hypothesis decisions                                |

PNG figures are saved alongside these tables. Keep the manifests, calibration settings and raw per-run results when reporting findings. Do not commit large checkpoint weights or temporary derivative artifacts.

### 9. Scope and interpretation

Investigation A studies the inherited norm-mediated mechanism on fixed recipient models. Pretraining WD may also change which function is learned, through the effective learning-rate schedule, representation development, attention structure and other historical effects. These routes can coexist. Similar checkpoint losses or embedding similarities do not establish identical functions, shared initialization or shared training sample order.

A successful local intervention establishes a conditional effect on the particular recipient checkpoint, not population-level causal mediation of the WD effect across independently pretrained models. A large functional residual without utility benefit is evidence of altered local geometry, not improved plasticity. An LR-only benefit without validated transplant equivalence is evidence for an optimizer heuristic rather than a clean gauge intervention. First-step results cannot establish endpoint behaviour; the short trajectories test persistence but full fine-tuning remains necessary to connect the mechanism to the original study's longer-horizon outcomes.

The next investigation should be determined by the measured evidence: if a simple global or two-group profile is sufficient, prioritize that explanation; if the measured profile is beneficial beyond these controls and the identities are validated, investigate its longer-horizon effects; if the local native advantage is absent, reconsider the task and protocol before claiming an explanation of the published phenomenon.

## References

- Han, T., Bordt, S., Zhang, H., & Kakade, S. (2026). [Weight Decay Improves Language Model Plasticity](https://arxiv.org/abs/2602.11137). [Training and evaluation repository](https://github.com/th789/wd-plasticity).
- Kosson, A., Messmer, B., & Jaggi, M. (2024). [Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks](https://proceedings.mlr.press/v235/kosson24a.html).
- Wang, X. & Aitchison, L. [How to set AdamW's weight decay as you scale model and dataset size](https://arxiv.org/abs/2405.13698).
- Martens, J. (2020). [New Insights and Perspectives on the Natural Gradient Method](https://jmlr.org/papers/v21/17-678.html).
- [Effective Learning Rate Governs Loss Dynamics in Language Model Pretraining](https://arxiv.org/abs/2608.24814). Relevant to the historical effective-LR route; similar pretraining loss trajectories do not establish identical learned functions.
