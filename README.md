# Experiment under "noise_init"

## <u>1. Model and metric selection:</u>
**For the model：**
- **FLUX.2 - Klein - 4B (4-step)** is more up-to-dated，is used more often, put in prioriity；
- SDXL-Turbo is distilled， but can still adopt for comparison；

There is a difference between distilled/non-distilled models on the probability distribution learned. (have to notice)
May bring significant difference to the generated result.
Distilled be less stable on the generated result for different noise variance.

**Measurements：**
- **Diversity：** Dreamsim, Lpips, Vendi；

- **Quality：** CLIP, HPSv3 (HPv2 uses less computation)；*(MANIQA if progress goes well)*

##
## <u>2. Noise Design:</u>
### Pink noise baseline:

$z \in (H\times W\times D)$ white noise  
$\hat z \in (H\times W\times D)\ u,v\ domain$

In some channel ∈ D

$\hat z_\alpha(u,v) = \hat z(u,v)\cdot\frac{1}{(1+f_{u,v})^\alpha} \qquad f_{u,v}=\sqrt{u^2+v^2}$ $\qquad$ Radial distance from position $(u,v)$ to the centre within the fourier plain;

$z_\alpha=\mathrm{normalise}\big(\mathrm{FFT2D}^{-1}(\hat z_\alpha(u,v))\big)$ 

##
### Ours:

**Increase α:** for pink, The higher the freq, the stronger the attenuation; (i.e. Reduce mid/high freq)  
**what we want:** Restore its suppressed mid/high frequency power，since low-freq for diversity，high-freq：quality.

##
#### Design A: Same-phase PSD floor

$\hat z_{\alpha,\gamma}(u,v) = \hat z(u,v)\sqrt{(1-\gamma)H_\alpha(u,v)^2+\gamma} \quad H_\alpha(u,v)=\frac{1}{(1+f_{u,v})^\alpha}$

//given γ, the higher the freq, the higher it is lifted. $H_α$ is the response after $\alpha$ filtering

$z_{\alpha,\gamma}=\mathrm{normalise}\big(\mathrm{FFT2D}^{-1}(\hat z_{\alpha,\gamma}(u,v))\big)$

##
#### Design B: Independent-white replenishment

$\hat z_\alpha(u,v) = \hat z(u,v)\cdot\frac{1}{(1+f_{u,v})^\alpha} \qquad f_{u,v}=\sqrt{u^2+v^2}$

$z_\alpha=\mathrm{normalise}\big(\mathrm{FFT2D}^{-1}(\hat z_\alpha(u,v))\big)$

$z'_\alpha = \sqrt{1-\gamma}\, z_\alpha + \sqrt{\gamma}\, \eta, \quad \eta \sim N(0, I),\quad \eta \perp z$

//The mixed-in noise is independently sampled white noise η，not same z.

##
## <u>3. Process:</u>
- Prepare the generation result on the baseline noise：the **quality** and **diversity score** on white noise and simple pink noise of different α.
  **-> obtain Quality-Diversity curve for baselines.**
- Measure same metrics for diversity and quality on "ours" method.
  **-> Quality-Diversity curve for "ours".**
- Compare the Quality-Diversity curves obtained，examine improvement on quality under same diversity level.
  (Or the other way around; Can trial on different models)



##
## <u>4. Other possible consideration:</u>
- ODE/SDE Sampling; May also use SDE for flow model；
