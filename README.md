# Intrinsic_Data_Structures_via_Singular_Metrics_in_VAEs
Code for the numerical experiments of the paper "Manifold Learning for Inverse Problems: Estimation of Dataset Intrinsic Dimension via Singular Metrics ".

This repository contains the code for the numerical experiments of the paper "Estimation of Intrinsic Data Structures via Singular Metrics in VAEs: Application to Inverse Problems" (preprint available at https://arxiv.org/abs/2507.07291).

Please cite the paper, if you use the code.

```
@article{ManifoldVAEs,
  author  = {Paola Causin and Alessio Marta},
  title   = {Estimation of Intrinsic Data Structures via Singular Metrics in VAEs: Application to Inverse Problems},
  eprint = {https://arxiv.org/abs/2507.07291}
}
```

The code for the mixture of VAEs is an adaption of https://github.com/johertrich/Manifold_Mixture_VAEs to non-invertible VAEs.

- The scripts `train_whitney_img_cnn.py` and `train_whitney_sin_cnn.py` contains the code to train the VAEs performing the Whitney embedding of the images and sinograms datasets (see fig. 4 of the paper)
- The scripts `dimension_estimate_img.py` and `dimension_estimate_sin.py` contains the code to estimate the intrinsic dimension of the manifold learned by the Whiteny embedding VAEs.
- The scripts `comparison_dimension_estimate_imgs.py` and `comparison_dimension_estimate_sins.py` contains the code to compare our dimension estimate with other three ID estimation methods.
- The scripts `generate_charts_img.py` and `generate_charts_sin.py` contains the code to generate the local charts for the learned manifolds.
- The script `pruning_img_net.py` contains the code to prune the VAE performing the Whitney embedding of the images.
