# Update:
- Had a meeting with Terry for discussion;
  Some adjustment on the current experiment plan according to urgency.

- Experiment in progress.


##
# Experiment in Progress:

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
white floor idea；
May be different for the noise optained, and result generated:
- Same-phase PSD floor：
  - Operating in the FFT space：
    $$\hat{z}_{\alpha,\gamma}^{\mathrm{same}}(f) = \hat{\epsilon}(f) \sqrt{(1-\gamma)H_\alpha(f)^2+\gamma}
$$
    
- Independent-white replenishment
  Can directly combine in noise space
  (same effect as $\hat{z}_{\alpha,\gamma}^{\mathrm{ind}}(f)
=\sqrt{1-\gamma}\,H_\alpha(f)\hat{\epsilon}(f)
+\sqrt{\gamma}\,\hat{\eta}(f)$.)

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
