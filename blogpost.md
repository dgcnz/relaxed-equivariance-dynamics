# Effect of equivariance on training dynamics

*Group Equivariant Convolutional Networks* (G-CNN) have gained significant traction in recent years owing to their ability to generalize the property of CNNs being equivariant to translations in convolutional layers. With equivariance, the network is able to exploit groups of symmetries and a direct consequence of this is that it generally needs less data to perform well. However, incorporating such knowledge into the network may not always be advantageous, especially when the data itself does not exhibit full equivariance. To address this issue, the concept of *relaxed equivariance* was introduced, offering a means to adaptively learn the degree of equivariance imposed on the network, enabling it to operate on a level between full equivariance and no equivariance.

Interestingly, for rotational symmetries on fully equivariant data, [Wang et al. (2023)](#References) found that a fully equivariant network exhibits poorer performance compared to a relaxed equivariant network. This is a surprising result since usually, fully equivariant models perform the best in cases where the data is also fully equivariant and is precisely what they were designed for. One plausible rationale for why it is different in this case is that the training dynamics benefit from the relaxation of the equivariance constraint.  To address this proposition, we use the framework described in [Park & Kim (2022)](#References) for measuring convexity and flatness using the Hessian spectra.

Since relaxed equivariance adaptively learns the amount of equivariance from the data it is important to also understand this process to investigate the training dynamics.
To do this, we investigate how the equivariance possessed by the data at hand influences the training dynamics. Importantly, [Gruver et al. (2022)](#References) shows that more equivariance imposed on a network does not necessarily imply more equivariance learned. As such, we will use more advanced methods to discover the true equivariance that the model expresses. 

Inspired by the aforementioned observations, the purpose of this blog post is to investigate the following question. **How does the equivariance imposed on a network affect its training dynamics?** With support from these observations we hypothesize that the less equivariance is imposed the better the training dynamics will be. To answer our research question, we identify the following subquestions:

1.  How effective is regularization for imposing equivariance on a network?
2.  How does the amount of equivariance imposed on the network affect the amount of equivariance learned?
3.  How does equivariance imposed on a network influence generalization?
4.  How does equivariance imposed on a network influence the convexity of the loss landscape?   


We answer these questions by:
- Reproducing results to establish common ground
- Performing experiments to investigate learned equivariance
- Analyzing trained models to investigate their training dynamics






<!--Furthermore, to either substantiate some of the aforementioned observations or the effectiveness of approximate equivariance networks, we validate the following claims:

- Posed in [1], an approximate equivariant network outperforms an equivariant network on a fully equivariant dataset (isotropic flow).
- Posed in [5], an approximate equivariant network outperforms an equivariant network on non-equivariant smoke dataset. 
-->
## Reproduction 
First, we introduce some theory needed for the reproduction. After that, we will go over the exact experiments we reproduced and our results.
### Regular G-CNN

Consider the segmentation task depicted in the picture below.

![Alt text](https://analyticsindiamag.com/wp-content/uploads/2020/07/u-net-segmentation-e1542978983391.png)

Naturally, applying segmentation on a rotated 2D image should give the same segmented image as applying such rotation after segmentation. Mathematically, for a neural network $NN$ to be equivariant w.r.t. the group $(G,\cdot)$ comprised of 2D rotations, which is a set equipped with a closed and associative binary
operation, containing inverse elements and an identity, the following property needs to be satisfied: 

$$
NN(gx) = gNN(x)
$$

for all images $x \in \mathbb{R}^2$ and rotations $g \in G$.

To build such a network, it is sufficient that each of its layers is equivariant in the same sense. Recall that a CNN achieves equivariance to translations by sharing weights in kernels that are translated across the input in each of its convolution layers. A G-CNN extends this concept of weight sharing to achieve equivariance w.r.t an arbitrary (finite) group $G$. 

#### Lifting convolution

Consider any 2D image as an input signal $f^0: \mathbb{R}^2 \rightarrow \mathbb{R}^c$, where $c$ is the number of channels. When passing it through a G-CNN, from the outset, it undergoes the lifting convolution with kernel $k : \mathbb{R}^2 \rightarrow \mathbb{R}^{n \times c}$ on $x \in \mathbb{R}^2$ and $g \in G$:

$$(k*\_{lifting} f^0)(x,g) = \int_{y \in \mathbb{R}^2}k(g^{-1}(y-x))f^0(y)$$

Suppose $f^1: \mathbb{R}^2 \times G \rightarrow \mathbb{R}^n$ is the output signal thereof, which is fed to the next layer.

#### $G$-equivariant convolution

Now, $f^1$ undergoes $G$-equivariant convolution with a kernel $\psi: \mathbb{R}^2 \times G \rightarrow \mathbb{R}^{m \times n}$ on $x \in \mathbb{R}^2$ and $g \in G$:

$$(\psi *\_{G} f^1)(x, g) = \int_{h \in G}\int_{y \in \mathbb{R}^2}\psi(g^{-1}(y-x), g^{-1}h)f^1(y, h)$$

For [Relaxed Equivariant Networks](#Relaxed-Equivariant-Networks), we define $H := \mathbb{R}^2 \rtimes G$, which means $$(\psi *\_{G} f^1)(h') := (\psi *\_{G} f^1)(x, g)$$ for $h' = (x, g) \in H$.
This gives the output signal $f^2: \mathbb{R}^2 \times G \rightarrow \mathbb{R}^m$. This way of convolving is repeated for all subsequent layers until the final aggregation layer, e.g. linear layer, if there is one.

Note that for the group convolution to be practically feasible, $G$ has to be **finite** and relatively small in size (roughly up to a hundred elements) and $\mathbb{R}^2$ becomes $\mathbb{Z}^2$.
However, if one is interested in equivariance w.r.t. an infinite group, e.g. all 2D rotations, the best they can do is to pick $G$ as a finite subset of those rotations. In this case, it is also unclear to what extent such a  network is **truly** rotationally equivariant.

### Steerable G-CNN

First, consider the representation of rotations $\rho_{in}: G \rightarrow \mathbb{R}^{out \times in}$ parameterized by the infinite group $G$, and analogously $\rho_{out}$.
To address the aforementioned equivariance problem w.r.t. the group of 2D rotations, $G$-steerable convolution modifies $G$-equivariant convolution with the following three changes:

- The input signal becomes $f: \mathbb{R}^2 \rightarrow \mathbb{R}^{in}$.
- The kernel $\psi: \mathbb{R}^2 \rightarrow \mathbb{R}^{out \times in}$ used must satisfy the following constraint for all $g \in G$: $$\psi(gx) = \rho_{out}(g) \psi(x) \rho_{in}(g^{-1})$$
- Standard convolution only over $\mathbb{R}^2$ and not $G$ is performed.

To secure kernel $\psi$ has the mentioned property, we precompute a set of non-learnable basis kernels $(\psi_l)_{l=1}^L$ which do have it, and define all other kernels as weighted combinations of the basis kernels, using learnable weights with the same shape as the kernels.

Therefore, the convolution is of the form:

$$
(\psi*\_{\mathbb{Z}^2}f) (x) = \sum_{y \in \mathbb{Z}^2} \sum_{l=1}^L (w_l ⊙ \psi_l(y))f(x+y)
$$

Whenever both $\rho_{in}$ and $\rho_{out}$ can be decomposed into smaller building blocks called **irreducible representations**, equivariance w.r.t. infinite group $G$ is achieved (see Appendix A.1 of [[15]](#References)).

<!--
ALTERNATIVE:

We will focus on rotation equivariance to introduce steerable G-CNNs.
Let $f: \mathbb{Z}^2 \to \mathbb{R}^{c_{in}}$ be an imput feature map, $H$ a group of 2D rotations, $\rho_{in}$ and $\rho_{out}$ representations of the group $H$, i.e. homomorphisms from $H$ to $\mathbb{R}^{c_{in} \times c_{in} }$ and  $\mathbb{R}^{c_{out} \times c_{out} }$ respectively.

We can convolve $f$ with a kernel $\psi: \mathbb{Z}^2 \to \mathbb{R}^{c_{out} \times c_{in}}$ using the standard 2D convolution operation $f*_{\mathbb{Z}^2}\psi$. 
This operation is $H$  -->

<!--
Furthermore, we may efficiently re-use part of the results for convolution with one transformation for all other transformations by observing the following 2 facts:

1. $\rho_n$ may be seen as $n^n$ real functions over $G$.
2. By Peter-Weyl theorem, any function defined on $G$ (parameters of rotation), may be expressed as linear combination of the real functions that the irreduciblie representations are made of, where the (Fourier) coefficients may be calculated through Fourier transform [[20]](#References). Since, the Fourier transform involves an integral over the infinite group $G$, in practice, it is often estimated using a finite number of randomly sampled group elements.

This means that if the number of irreducible representations are finite, we have that: $$\rho(g) \psi(x) := \psi(\rho(g^{-1})x) = \sum_{q=1}^Q w_q(g) \psi_{q}(x)$$
This implies that convolution of a function $f$ with a rotated steerable kernel is the linear combination of convolution with the rotated steerable basis kernels: $$(\rho(g) \psi * f)(x) = \sum_{q=1}^Q w_q(g) ( \psi_q * f)(x)$$ 
-->

<!-- With this, we see that $\psi$ is steerable in the sense that the output of the convolution is being steered with the orientation of the kernel by the weights. -->

<!-- The reason that with this construction, equivariance w.r.t. a infinite group is achievable, is because those irreducible representations form a basis for some of the functions defined over $G$. Therefore, picking the weights appropriately facilitates the desired equivariance when the basis of irreducible representations is finite.  -->

### Relaxed Equivariant Networks

The desirability of equivariance in a network depends on the amount of equivariance possessed by the data of interest. To this end, relaxed equivariant networks are built on top of G-CNNs using a modified (relaxed) kernel consisting of a linear combination of standard G-CNN kernels. Define $H := \mathbb{R}^2 \rtimes G$, then, for relaxed G-equivariant group convolution is defined for $g \in H$ as:

$$
(\psi *^R_{G} f)(g) = \sum_{h \in G}\psi(g,h)f(h) = \sum_{h \in G}\sum_{l=1}^L w_l(h) \psi_l(g^{-1}h)f(h)
$$

$G$-equivariance of the group convolution arises from kernel $\psi$'s dependence on the composite variable $g^{-1}h$, rather than on both variables $g$ and $h$ separately. This property is broken in relaxed kernels, leading to a loss of equivariance.

The equivariance error increases with the number of kernels $L$ and the variability of $w_l(h)$ over $h \in G$, allowing us to control the amount of equivariance imposed on the model by adding the term:

$$ 
\alpha \sum_{l=1}^L\sum_{g,h \in G}|w_l(g)-w_l(h)|
$$

to the loss function. A higher value of the hyperparameter $\alpha$ enforces a higher degree of equivariance, by forcing the weights to be nearly constant functions.

Therefore, using relaxed group convolutions allows the network to relax strict symmetry constraints, offering greater flexibility at the cost of reduced equivariance.

Relaxed steerable G-CNNs are defined using a similar idea, again we let the weights depend on the variable of integration:

$$(\psi *\_{\mathbb{Z}^2}^R f) (x) = \sum_{y \in \mathbb{Z}^2} \sum_{l=1}^L (w_l(y) ⊙ \psi_l(y))f(x+y)$$

which leads to a loss of equivariance. Not unlike the previous case, the closer the weights are to constant functions the more equivariant the model is, and thus we can impose equivariance by adding the following term to the loss function:

$$
\alpha(\|\frac{\partial w(m,n)}{\partial m}\|+ \|\frac{\partial w(m,n)}{\partial n}\|)
$$

Here the partial derivatives are discrete and just represent the difference of neighbouring weight values over spacial locations.

<!--Naturally, we would expect approximately equivariant networks to achieve better results than fully equivariant models on datasets which are themselves not perfectly equivariant.
[1] supports this intuition showing that an AENN yielded better results than a fully equivariant model on super-resolution tasks for partially-equivariant channel flow data.
Interestingly, the AENN prevailed even in the fully-equivariant isotropic flow dataset, which could potentially be explained by AENN weights enhancing optimization.-->

## Reproduction Methodology

We perform two reproduction studies in this blogpost.

Our first objective is to reproduce the experiment demonstrating that a relaxed equivariant model can outperform a fully equivariant model on fully equivariant data. Specifically, we reproduce the Super-resolution of 3D Turbulence experiment (Experiment 5.5) in the paper "Discovering Symmetry Breaking in Physical Systems with Relaxed Group Convolution" [(Wang et al., 2023)](#References). This reproduction provides grounds for expecting a relaxed equivariant model to outperform a model that is properly equivariant to the symmetries of the data.

In this study, the authors of [[1]](#References) focus on understanding the asymmetries in physics data using relaxed equivariant neural networks. They employ various relaxed group convolution architectures to identify symmetry-breaking factors in different physical systems.

<!-- This reproduction allows us to investigate the effects of approximate equivariance on the training dynamics in our extensions. We will do so by looking at the differences between a non-equivariant model, an approximate equivariant one and a fully equivariant one. -->
<!-- to ensure relaxed group convolutions to perform well as shown in Wang 2024 [[1]](#References), -->

Additionally, we reproduce results from [Wang et al. (2022)](#References) that introduced relaxed group convolutions. In this paper, relaxed group convolutions are compared to other methods on the 2D smoke simulation data. We intend to do the same, focusing on the experiments that fit our first objective the closest, namely the ones involving rotational symmetries.
<!--
Note: The following sentence is NOT true, we have a different dataset for reproduction and for analysis.
We will again evaluate the training dynamics of our methods on this smoke simulation dataset. -->


### Super Resolution

In Experiment 5.5 of [Wang et al. (2023)](#References), the authors evaluate the performance of one network architecture with three variations 1) convolutional blocks, 2) group equivariant blocks, and 3) relaxed group equivariant blocks. All networks are tasked with upscaling 3D channel isotropic turbulence data.

The data consists of liquid flowing in 3D space and is produced by a high-resolution state-of-the-art simulation hosted by the John Hopkins University [(Li et al., 2008)](#References). Importantly, this dataset is forced to be isotropic, i.e. fully equivariant to rotations, by design. 

For the experiment, a subset of 50 timesteps are taken, each downsampled from $1024^3$ to $64^3$ and processed into a task suitable for learning. The model is given an input of 3 consecutive timesteps, which are all downsampled again to $16^3$, and is tasked to predict and to upsample to the subsequent timestep, which is still of size $64^3$. Furthermore, we have published our data processing scripts and production-ready dataloader on HuggingFace [[9]](#References).

The architecture of the models is entirely taken over from [[1]](#References), with the following additions that are not specified in the paper. The first layer of the relaxed and the regular GCNNs are lifting layers. To preserve spatial size, in every non-Upconv convolution, $1$ is used as the stride, padding and dilation, to preserve spatial size. On the other hand, to double the spatial dimension in the Upconv layers, $2$ is used as the stride, and $1$ as the padding, dilation and output padding. For all layers, separable group convolutions are used. For the Upconv layers this meant that upsampling was only done on the spatial elements, while for a non-Upconv convolution, this was done on the group elements instead. Additionally, batch norm and ReLu where used after every convolution. Finally, for training the Adam optimizer was used with a learning rate of $0.01$.

#### Expectation Based on Equivariance

- For isotropic turbulence, networks with full equivariance should either outperform or be on par with those with relaxed equivariance, as the data fully adheres to isotropic symmetry. However, as shown in [[1]](#References), the opposite happens.
<!--
#### Results and Observations

Their results indicate that incorporating both equivariance and relaxed equivariance enhances prediction performance. Interestingly, the relaxed group convolution outperformed even the fully equivariant network on isotropic turbulence data, which contradicts initial expectations.

This unexpected outcome may be attributed to the enhanced optimization process facilitated by the relaxed weights, and serves as the main motivating factor for our extension.
 -->

### Smoke Plume

For this experiment in [[5]](#References), a specialized 2D smoke simulation was generated using Phiflow ([[6]](#References)). In these simulations, smoke flows into the scene from a position towards the direction of the buoyant force. For instance, in everyday life, the buoyant force opposes gravity and thus smoke floats upwards. Additionally, every inflow position has a slightly different buoyant force, varying in either strength or angle of dispersion. Furthermore, for a given inflow position, the simulation is ran multiple times. Each time, its buoyant force is modified by a scalar factor to increase the difference between the buoyant forces. In total, this results in $4$ different inflow positions, each of which is simulated $10$ times for $311$ timesteps.

<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/S1RILeLE0.png" alt="Figure 2" style="max-width: 100%;">
  <p>Figure 2: Example of a Smoke Plume sequence.</p>
</div>

Using this dataset, the task is to predict the upcoming frames based on the previous ones. Evaluation on this task is done on the following settings:

- Domain: the model is tested on inflow locations it was not trained on.
- Future: the model is tested on timesteps that are further in the simulation than what it was trained on.

#### Expectations Based on Equivariance

- Since, the dataset is partially equivariant to rotation, the partial equivariant model is expected to perform the best.

## Reproduction Results
<!--We aim to reproduce the experiment of [1] using a 64x64 synthetic smoke dataset which has rotational symmetries. Specifically the data contains 40 simulations varied by inflow positions and buoyant forces, which exhibit perfect C4 rotational symmetry. However, buoyancy factors change with inflow locations, disrupting this symmetry.-->
Both of our reproduction studies **corroborate** the conclusion drawn from the results in the original papers.

### Super Resolution

We compare our results with those of [Wang et al. (2023)](#References) for the CNN (SuperResCNN), regular group equivariant network (GCNNOhT3), and relaxed regular group equivariant network (RGCNNOhT3). The reconstruction mean absolute error (MAE) is presented in the table below.

<div style="display: flex;">
  <table border="0">
    <tr>
      <th colspan="3">Results from original paper (MAE 1e-1)</th>
    </tr>
    <tr>
      <td>cnn</td>
      <td>gcnn</td>
      <td>rgcnn</td>
    </tr>
    <tr>
      <td>1.22 (0.04)</td>
      <td>1.12(0.02)</td>
      <td><b>1.00(0.01)</b></td>
    </tr>
      <caption style="caption-side: bottom"> MAEs on the super resolution dataset in the original paper. </caption>
  </table>

  <table border="0" > <!-- Adding space between the tables -->
    <tr>
      <th colspan="3">Reproduction Results (MAE 1e-1)</th>
    </tr>
    <tr>
      <td>cnn</td>
      <td>gcnn</td>
      <td>rgcnn</td>
    </tr>
    <tr>
      <td>0.992(0.03)</td>
      <td>0.928(0.04)</td>
      <td><b>0.915(0.04)</b></td>
    </tr>
      <caption style="caption-side: bottom">Our MAEs on the super resolution dataset. </caption>
  </table>
</div>

We see that although all our results are a little better than the original ones, the trend of relaxed equivariant model **outperforming** the fully equivariant model on fully equivariant data remains.

Additionally, we investigate the parameter efficiency of the networks below.
<!-- Although we made sure to have comparable number of parameters, we can go a step further and normalize the above metrics by each model's parameter counts to analyze parameter-efficiency. -->

<div style="display: flex;">
  <table border="0">
    <tr>
      <th colspan="3">Number of learnable parameters</th>
    </tr>
    <tr>
      <td>cnn</td>
      <td>gcnn</td>
      <td>rgcnn</td>
    </tr>
    <tr>
      <td>132795</td>
      <td>123801</td>
      <td>130175</td>
    </tr>
      <caption style="caption-side: bottom"> </caption>
  </table>

  <table border="0" > <!-- Adding space between the tables -->
    <tr>
      <th colspan="3">Parameter Efficiency (MAE per 1e6 parameters)</th>
    </tr>
    <tr>
      <td>cnn</td>
      <td>gcnn</td>
      <td>rgcnn</td>
    </tr>
    <tr>
      <td>0.747 (0.03)</td>
      <td>0.750 (0.03)</td>
      <td><b>0.703 (0.03)</b></td>
    </tr>
  </table>
</div>

We observe that the relaxed equivariant network is more parameter-efficient than the fully equivariant network.

### Smoke plume

We compare our results with those in [Wang et al. (2022)](#References) for the relaxed regular and steerable GCNNs. The reconstruction RMSE for both methods is shown in the table below. 

<div style="display: flex;">
  <table border="0">
    <tr>
      <th colspan="3">Results from original paper</th>
    </tr>
    <tr>
      <td></td>
      <td>rgroup</td>
      <td>rsteer</td>
    </tr>
    <tr>
      <td>Domain</td>
      <td>0.73 (0.02)</td>
      <td><b>0.67 (0.01)</b></td>
    </tr>
    <tr>
      <td>Future</td>
      <td>0.82 (0.01)</td>
        <td><b>0.80 (0.00)</b></td>
    </tr>
      <caption style="caption-side: bottom">Root Mean Squared Errors found on the smoke plume dataset for both tasks by the original authors </caption>
  </table>

  <table border="0" > <!-- Adding space between the tables -->
    <tr>
      <th colspan="3">Reproduction Results</th>
    </tr>
    <tr>
      <td></td>
      <td>rgroup</td>
      <td>rsteer</td>
    </tr>
    <tr>
      <td>Domain</td>
      <td>0.90 (0.04)</td>
        <td><b>0.67 (0.00)</b></td>
    </tr>
    <tr>
      <td>Future</td>
      <td>0.88 (0.03)</td>
        <td><b>0.82 (0.00)</b></td>
    </tr>
      <caption style="caption-side: bottom">Root Mean Squared Errors found on the smoke plume dataset for both tasks by the current paper </caption>
  </table>
</div>

Again, we see that although there are some discrepancies in values, the same trend with comparable performance is observed for the relaxed steerable GCNN. On the other hand, our results for the relaxed regular GCNN are somewhat different from those reported in the original paper. One potential reason for this is that the original paper did not provide the hyperparameters used to obtain their results, and we did not perform a grid search over the provided grid of parameter values. Another reason might be that the early stopping metric we used is different.

### Reproduction Efforts

To maximize reproducibility and future usability, we provide config files for all the experiments, models, datasets, trainers, etc. using Hydra and PyTorch Lightning (more information on the README). This means that all the models are wrapped in Lightning Modules and all datasets (SmokePlume, JHTDB) are uploaded to HuggingFace and have a corresponding Lightning DataModule. We reuse and upgrade the data generation scripts for the SmokePlume datasets from [Wang et al. (2022)](#References) and implement a configurable data generation and HuggingFace-compatible data loading script from scratch for the JHTDB Dataset. Furthermore, we integrate our code with Weights and Biases and publish all the relevant runs and plots on publicly accessible reports ([[11], [12], [13]](#References)). Finally, for the relaxed regular group convolutional networks, we implement all components on our fork of the `gconv` library ([[14]](#References)).


To summarize the missing/added reproduction code:
- For [Wang et al. (2023)](#References):
  - All models (rgcnn, gcnn, cnn), where we implement 3d relaxed separable convolutions, octahedral group convolutions, 3d equivariant transposed, convolutions, 3d group upsampling, and we made educated guesses on which activations, normalizations to use and where to place them (along skip and upsampling residual connections).
  - The JHTDB dataset, where we implement all the subsampling, preprocessing and loading logic of the 3d turbulence velocity fields.
- For [Wang et al. (2022)](#References), we added the missing weight constraint and hyperparameters for rgroup.

All the experimentation code can be found at: https://github.com/dgcnz/dl2.

## Extension and Analysis
As the results of the reproduction match those in their respective papers, we are free to conduct several analyses using approximate equivariance. For these experiments, we introduce a dataset that is very similar to the 2D smoke dataset seen in [Reproduction](#Reproduction). Additionally, we analyze trained models to learn about their training dynamics. The techniques used for this are explained in [Theory for Analysis](#Theory-for-Analysis). With these results, we answer the research questions posed in [Introduction](#Introduction) to ultimately shed light on the role of imposed equivariance equivariant models on its training dynamics.



### Extension Theory
In this section, we introduce the necessary definitions of measuring quantities of interest for our additional experiments.
#### Measuring the Amount of Equivariance Learned by a Network
##### Equivariance Error
It is natural to measure the amount of equivariance a network $f$ has as the expected difference between the output of the transformed data and the transformed output of the original data.

$$\mathbb{E}_{x \sim D}\left[\frac{1}{|G|}\sum_{g \in G}\|f(gx)-gf(x)\|\right]$$

We can estimate this expectation by computing the average for a series of batches from our test set. However, this approach has downsides, which we can tackle using the Lie derivative. 

##### Lie Derivative

<!--
<span style="color:red;">Add Lie group def and say that G' usually has that structure?</span>
-->


<!-- However, this approach is problematic as it only measures the amount of equivariance w.r.t. the finite group $G$. Instead, [2] proposed the use of (Lie) derivatives to evaluate the robustness of the network to infinitesimal transformations. For the notion of derivative to be defined, however, we need to assume the group to have a differential structure (Lie group). Since the space consisting of $G$ may be too peculiar to work in, we smoothly parameterize the representations of these transformations in the tangent space at the transformation that does nothing (identity element).  -->

In practice, even though we are imposing $G$-equivariance on a network, what we would like to achieve is $G'$-equivariance for an infinite (Lie) group $G'$ which contains $G$. The previous approach is problematic as it only measures the amount of acquired equivariance w.r.t. the finite group $G$, neglecting all other transformations, and thus doesn't give us the full picture of the network's equivariance.
 
 [Gruver et al. (2022)](#References) proposed the use of Lie derivatives, which focus on the equivariance of the network towards very small transformations in $G'$, and give us a way to measure full $G'$-equivariance. The intuitive idea is the following: Imagine a smooth path $p(t)$ traversing the group $G'$ that starts at the identity element $e_{G'}$ (i.e. transformation that does nothing) of the group. This means that at every time-point $t \geq 0$, $p(t)$ is an element of $G'$ (some transformation), and $p(0) = e_{G'}$. Then, we can define the function:
  $$\Phi_{p(t)}f(x) := p(t)^{-1}f(p(t)x)
  $$
  This function makes some transformation $p(t)$ on the data, applies $f$ to the transformed data, and finally applies the inverse transformation $p(t)^{-1}$ to the output. Notice that if $f$ is $G'$-equivariant this value is constantly equal to $f(x)$, and that $\Phi_{p(0)}f(x) = f(x)$ always. The Lie derivative of $f$ along the path $p(t)$ is the derivative 
  $$L_pf(x) := \frac{d\Phi_{p(t)}f(x)}{dt} = \lim_{t \to 0+} \frac{\Phi_{p(t)}f(x) - f(x)}{t}
  $$
at time $t=0$. One might note that this only measures the local equivariance around the identity element of the group. Luckily, it is shown in [Gruver et al. (2022)](#References) that $L_{p}f(x) = 0$ for all $x$ and $d$ specific paths, where $d$ corresponds to the dimensionality of $G'$, is equivalent to $f$ being $G'$- equivariant.


Specifically, if $G'$ is the set of all 2D rotations, then $d=1$ and there is only one relevant path $p$. We can interpret $\|L_pf(x)\|$ as the rotation-equivariance error of $f$ at point $x$. And thus define the rotation-equivariance error of $f$ on a dataset $D$ as 

$$
\frac{1}{|D|}\sum_{x \in D} \|L_pf(x)\|
$$

Having small Lie derivatives (in norm) therefore implies that $f$ is highly rotation-equivariant.



<!-- 
### Equivariance error (EE)

 Another alternative for measuring equivariance relies on the variant of approximate equivariance network we consider. Recall that what broke equivariance therein are the weights used in the linear combination of kernels that constituted the modified kernel. Therefore, a proxy for the amount of equivariance is naturally the difference between the individual kernels used over all possible transformations.
$$\frac{1}{L|G|}\sum_{l=1}^L\sum_{g \in G} |w_l(g)-w_l(e)|$$ 

-->

### Training Dynamics Evaluation

To assess the training dynamics of a network, we are interested in the final performance and the generalizability of the learned parameters, which are quantified by the final RMSE, and the sharpness of the loss landscape near the final weight-point [(Zhao et al., 2023)](#References). 

#### Sharpness

To measure the sharpness of the loss landscape after training, we consider changes in the loss averaged over random directions. Let $D$ denote a set of vectors randomly drawn from the unit sphere, and $T$ a set of displacements, i.e. real numbers. Then, the sharpness of the loss $\mathcal{L}$ at a point $w$ is: 

$$ \phi(w,D,T) = \frac{1}{|D||T|} \sum_{t \in T} \sum_{d \in D} |\mathcal{L}(w+dt)-\mathcal{L}(w)| 
$$

This definition is an adaptation from the one in [Zhao et al. (2023)](#References) which does not normalize by $\mathcal{L}(w)$ inside the sum. 
The perturbation of the weights in $\mathcal{L}(w + dt)$ can be understood as follows: first, flatten all the weight matrices into a single vector. Then, add a random unit vector, multiplied by a magnitude t, of the same shape to this vector. Finally, reshape the resulting vector back into the original set of weight matrices. A sharper loss landscape around the model's final weights usually implies a greater generalization gap.

#### Hessian Eigenvalue 

Finally, the Hessian eigenvalue spectrum ([Park & Kim, 2022](#References)) sheds light on both the efficiency and efficacy of neural network training. Negative Hessian eigenvalues indicate a non-convex loss landscape, which can disturb the optimization process, whereas very large eigenvalues indicate training instability, sharp minima and consequently poor generalization.



## Extension Methodology

The purpose of our extensions is twofold.
First, we examine the impact of equivariance imposed and data equivariance on the amount of equivariance learned, answering subquestions 1 and 2 posed in [Introduction](#Introduction). We do this by computing the [equivariance error](#Equivariance-Error) and [Lie derivative](#Lie-Derivative). We plot these measures for varying levels of imposed equivariance and data equivariance.  

Second, we examine how equivariance imposed on a network influences the convexity of the loss landscape and generalization, answering subquestions 3 and 4 posed in [Introduction](#Introduction). We can strongly impose equivariance on a network through architecture design, and we can weakly impose equivariance on a network through a regularization term in the loss of the relaxed models. We train multiple models with different levels of imposed equivariance on two fully equivariant datasets, namely the super resolution and 2D Smoke Plume with varying levels of Equivariance, both of which were introduced in [Reproduction Methodology](#Reproduction-Methodology). Note however that the 2D Smoke Plume we use for our additional study is modified, with the changes described in [Smoke Plume with Varying Equivariance](#Smoke-Plume-with-Varying-Equivariance). For these models, we examine the convexity of its loss landscape and its generalizability with the measures defined in [Extension Theory](#Extension-Theory).
<!-- For these models, we evaluate generalization by looking at the training and validation loss curves of different models and computing sharpness for different training epochs. We examine convexity of the loss landscape by computing hessian spectra for the models at different training epochs. -->
<!-- However, as we will see the weak method has minimal effects and so the strong method is used.   -->



### Equivariance Imposed and Learned
#### Smoke Plume with Varying Equivariance
For this experiment, we use a synthetic $64 \times 64$ 2D smoke simulation dataset generated by PhiFlow [(Holl et al., 2020)](#References). In contrast to the one used in [Reproduction](#Reproduction), this dataset has a fixed inflow position. Additionally, the buoyant force may only be pointed upwards, downwards, to the left, or right. To vary the dataset's equivariance, the strength of this force is different in each of the $4$ directions. The strength in the upward direction remains the same across all simulations, while it increases progressively in the right, downward, and left directions, with the left direction being the strongest. The larger the difference between the strength of the forces, the less equivariant the data becomes. On the other hand, when the forces are the same in all directions, the data is fully equivariant to the group comprised of $90$ degrees rotations (henceforth, we refer to this group as the C4 group), as the simulation develops in the same way after rotation. 

To quantify different levels of data equivariance, we use the following metric proposed in [Wang et al. (2022)](#References):
First, we rotate the right, down, and left directions back to the upward position. Then, we compare these rotated directions against the original upward direction. The MAE of these comparisons is considered as the equivariance error the dataset possesses.
<!-- Since for this experiment, we work with different levels of data equivariance, we need to quantify it. In this case, that can easily be achieved with a method inherent to the dataset itself, introduced in Wang 2022. Namely, by rotating the right, down and left directions back to the up position, we can then compare them against the upward direction. We then simply take the mean absolute error for these comparisons and add them up. -->

In total, we experiment with $10$ different settings of the equivariance level. Each of these includes $4$ simulations of $60$ timesteps.

We use $2$ models for our analysis on this dataset. The first model is the Rsteer model from [Wang et al. (2022)](#References), introduced in [Reproduction Methodology](#Reproduction-Methodology). These models possess relaxed equivariance with respect to the C4 group. We set the hidden dimension to 64, we use a dept of $5$ layers and set $\alpha=0$. The second model is the E2CNN model from [Weiler et al. (2019)](#References), which is fully equivariant to the C4 group. We set the hidden dimension to $86$ and use a dept of $5$ layers. To facilitate a fair comparison, both models have $600$ thousand parameters. 

Using these models means we have the following $2$ ways of imposing equivariance:

Strongly, by using the E2CNN model architecture which is strictly equivariant and has a very similar architecture to the rsteer model. Or weakly by adding a regularization term to the loss in the rsteer model; namely, the parameter $\alpha$ introduced in the [Relaxed Equivariant Networks](#Relaxed-Equivariant-Networks).

### Training Dynamics

We investigate this on two different datasets.

#### Fully equivariant Smoke Plume
We use the same [Smoke Plume dataset](#Smoke-Plume-with-Varying-Equivariance). We analyze the model checkpoints corresponding to the third and best epochs during training, where best means highest validation RMSE. As we will see, the regularization using $\alpha$ has minimal impact. Therefore, we use the Rsteer model with $\alpha=0$ and the aforementioned E2CNN model. Both models are trained using the Adam optimizer with weight decay $0.0004$ and learning rate $0.001$.

#### Super Resolution
We first investigate the model's generalizability by looking at the training and loss curves we obtained by running the reproduction experiments in [Reproduction](#Reproduction) for both the relaxed and the non-relaxed models. Additionally, we evaluate the sharpness metric for both models and a CNN.

For this, we again use the third epoch and the epoch with the best loss.
Although we wanted to also compute the Hessian spectra, this was unfortunately not possible because the second derivative of the 3D grid sampler used in both equivariant networks is not yet implemented in PyTorch.


## Extension Results - Measuring Learned Equivariance

<table>
  <tr>
    <td>
      <img src="https://hackmd.io/_uploads/H1tTMORQR.png" alt="Figure 3" style="max-width: 100%;">
      <p align="center">Figure 3: Impact of equivariance imposed on Model's Equivariance Error, for rsteer and E2CNN models</p>
    </td>
    <td>
      <img src="https://hackmd.io/_uploads/rytpfdRmA.png" alt="Figure 4" style="max-width: 100%;">
      <p align="center">Figure 4: Impact of equivariance imposed on Model's Lie Derivative, for rsteer and E2CNN models</p>
    </td>
  </tr>
</table>

Figure 3 shows the Equivariance Error of different model specifications. The Equivariant Net is the E2CNN model, it is positioned  to the right of the x-axis because we can think of a fully equivariant net as a relaxed equivariant net where alpha is set to infinity. 
<!-- Note that this is purely conceptual, setting alpha to a very high value would not have this effect in practice.  -->

For rsteer, we observe that the Data Equivariance has a large effect on how equivariant the model learns to be. This shows that the relaxed architecture can adapt well to the equivariance of the data, which matches the findings in [Wang et al. (2022)](#References). However, we see that the hyperparameter $\alpha$ has barely any effect on the Equivariance Error of the model, as alpha is increased the amount of Equivariance Error should decrease but it does not. We thus find that regularization is not very effective for imposing equivariance on a network for the values we choose. For E2CNN, the Equivariant Net, we see that the Equivariance Error is near zero for all levels of data equivariance. This is as expected because the equivariance to C4 rotation is built into the model, thus the Equivariance Error can only come from artifacts created when rotating the input or the output feature maps. 

Figure 4 shows the Lie derivative for different model specifications. A lower Lie Derivative means the model is more equivariant to the complete rotation group. For Rsteer we see similar results to Figure 3. However, for E2CNN, we do not see a zero Lie derivative because the architecture only guarantees equivariance w.r.t. the C4 group.  

Interestingly, rsteer exhibits a lower Lie derivative than E2CNN when trained on fully equivariant data. This could be due to rsteer's greater flexibility, allowing it to learn equivariance w.r.t. a broader group of rotations beyond C4. In contrast, E2CNN achieves perfect C4 equivariance but struggles to generalize to all rotations.


## Results: Training Dynamics 
### Smoke Plume with full Equivariance
<!--
In this section we will give further support to the statement that relaxed equivariant networks can outperform  fully equivariant networks on perfectly equivariant datasets, and analyze the training dynamics as a potential explanation of this phenomenon.
-->


First, we examine the training, validation and test RMSE for the E2CNN and Rsteer models on the fully equivariant Smoke Plume dataset. 
<table>
  <tr>
    <td>
      <img src="https://hackmd.io/_uploads/ByqLXIUEA.png" alt="Figure 5" style="max-width: 100%;">
      <p align="center">Figure 5: Train RMSE curve for rsteer and E2CNN models</p>
    </td>
    <td>
      <img src="https://hackmd.io/_uploads/rJ58Q8LVC.png" alt="Figure 6" style="max-width: 100%;">
      <p align="center">Figure 6: Validation RMSE curve for rsteer and E2CNN models</p>
    </td>
  </tr>
</table>


<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/HyqIXLIEA.png" alt="Figure 7" style="max-width: 100%;">
  <p>Figure 7: Test RMSE for best models</p>
</div>



Figures 5 and 6 show the train and validation RMSE curves, respectively. We see that on the training data, rsteer and E2CNN have similar performance. However, on the validation data, the curve for rsteer lies strictly below the one for E2CNN. Therefore, the relaxed steerable GCNN, i.e. rsteer, seems to generalize better. This again might be attributed to its flexibility compared to the vanilla steerable GCNN. 

Figure 7 shows the test set RMSE for the two models averaged over five seeds. We find that the relaxed equivariant model performs better, even though the data is fully C4 equivariant, reaffirming the observation we validated on the Isotropic Flow dataset.

To obtain insight into why the relaxed equivariant models outperform the fully equivariant ones on these datasets, we inspect the hessian spectra and the sharpness of the loss landscape of these models.

<table>
  <tr>
    <td>
      <img src="https://hackmd.io/_uploads/rJJQKAyNC.png" alt="Epoch 3" style="max-width: 100%;">
      <p align="center">Figure 8: Hessian spectra at an early epoch for rsteer and E2CNN models</p>
    </td>
    <td>
      <img src="https://hackmd.io/_uploads/S1jQF0JN0.png" alt="Epoch best" style="max-width: 100%;">
      <p align="center">Figure 9: Hessian spectra at the best epoch for rsteer and E2CNN models</p>
    </td>
  </tr>
</table>

Figures 8 and 9 show hessian spectra for the same early and best checkpoints of E2CNN and rsteer used in the previous analysis. With regard to the flatness of the loss landscape, these plots allow us to make a similar conclusion. We see that for both checkpoints E2CNN has much larger eigenvalues than rsteer, which can lead to training instability, less flat minima, and consequently poor generalization for E2CNN.

To evaluate the convexity of the loss landscape, we focus on the negative eigenvalues in the Hessian Spectra. We see that for both models, neither spectra shows any negative eigenvalues. This suggests that for both the fully equivariant E2CNN and the relaxed rsteer models, the points it has traversed in the loss landscape, exhibit "convex" loss landscapes. Thus, in this case, the convexity of the loss landscapes does not seem to play a large role in the performance.  


<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/S14l5XZ4A.png" alt="Figure 10" style="max-width: 100%;">
  <p>Figure 10: Sharpness at early and best epochs for rsteer and E2CNN models</p>
</div>

Next, we examine checkpoints for the two models trained on the Smoke Plume Dataset with $0$ equivariance error. We specifically look at a checkpoint during early training, epoch three, and the best model checkpoint. We see that the rsteer model has a significantly lower sharpness of the loss landscape for the best weights compared to E2CNN, which indicates a lower generalization gap, and thus more effective learning. We similarly observe lower sharpness for rsteer with early model weights, this matches with the lower validation RMSE curve for rsteer we saw earlier. Although a flatter loss landscape is not necessarily advantageous for faster convergence during training, it is interesting that the rsteer model exhibits a flatter loss landscape than E2CNN, rather than simply converging to a flatter minimum.

### Super Resolution

Similarly, we also analyze the training dynamics of the superresolution models on the isotropic JHTDB dataset as a potential explanation for the superiority of the relaxed equivariant model over the fully equivariant one.


First, we examine the training and validation MAE curves for the Relaxed Equivariant (RGCNN), Fully Equivariant (GCNN), and non-equivariant (CNN) models (run on 6 different seeds).

<table>
  <tr>
    <td>
      <img src="https://hackmd.io/_uploads/HJrVE8IEA.png" alt="Figure 8" style="max-width: 100%;">
      <p align="center">Figure 11: Training MAE curve for RGCNN, GCNN and CNN models</p>
    </td>
    <td>
      <img src="https://hackmd.io/_uploads/HJrV48UN0.png" alt="Figure 9" style="max-width: 100%;">
      <p align="center">Figure 12: Validation MAE curve for RGCNN, GCNN and CNN models</p>
    </td>
  </tr>
</table>



Here, we observe that early in the training (around epoch $3$), RGCNN starts outperforming the other two models and keeps this lead until its saturation at around $0.1$ MAE. For this reason, we take a checkpoint for each model on epoch $3$ (early) and on its best epoch (Best), to examine the corresponding sharpness value.  

<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/SkpgREzV0.png" alt="Figure 10" style="max-width: 100%;">
  <p>Figure 13: Sharpness of the loss landscape on the super resolution dataset. Ran over 6 seeds, error bars represent the standard deviation. For early, the third epoch was chosen, while for best the epoch with the best validation loss was chosen.</p>
    
</div>

In any case, as seen in Figure 13, the sharpness value of the loss landscape was the lowest for the relaxed model in both the early and best checkpoints. This again indicates that the relaxed steerable GCNN has better generalisability during its training and at its convergence, matching our previous findings in our extensions on the SmokePlume dataset and the reproduction study on the Super Resolution dataset.


## Concluding Remarks

We reproduced two experiments: (1) On the Smoke Plume dataset in [Wang et al. (2022)](#References) and (2) Super Resolution dataset in [Wang et al. (2023)](#References). Our reproduction results align with the findings of the authors of the original papers, reaffirming the effectiveness of relaxed equivariant models and demonstrating that they are able to outperform fully equivariant models even on perfectly equivariant datasets. We extend our findings from the reproduction in (2) to the fully equivariant smokeplume dataset and find that the same conclusion can be made there.

We furthermore investigated the authors' speculation that this superior performance could be due to relaxed models having enhanced training dynamics. Our experiments empirically support this hypothesis, showing that relaxed models exhibit lower validation error, a flatter loss landscape around the final weights, and smaller Hessian eigenvalues, all of which are indicators of improved training dynamics and better generalization.

Finally, we demonstrated that the amount of equivariance in the training data predominantly influences the amount of equivariance learned by relaxed equivariant models. Datasets with higher degrees of equivariance yield models with higher degrees of internalized equivariance. Conversely, adding regularization terms to the loss function has negligible effects on the amount of learned equivariance. 

Our results suggest that replacing fully equivariant networks with relaxed equivariant networks could be advantageous in all application domains where some level of model equivariance is desired, including those where full equivariance is beneficial. For future research, we should investigate different versions of the relaxed model to find out which hyperparameters, like the number of filter banks, correlate with sharpness. Additionally, the method should be applied to different types of data to see if the same observations can be made there.


<!--
Our finding that relaxed models outperform fully equivariant models can be related to previous literature examining the impact of inductive bias on training dynamics. Specifically, [Park & Kim, 2022](#References) highlights that the weak inductive bias of Vision Transformers (ViTs) results in a flat and non-convex loss landscape, compared to the sharp and convex loss landscape of CNNs. They introduced additional spatial locality to ViTs to limit non-convexity while still maintaining a flatter landscape. Similarly, for the Isotropic Flow dataset, we found that the relaxed model outperformed both the CNN and GCNN, attributable to a flatter loss landscape.
-->

<!--
Additionally, we demonstrated that the amount of equivariance in the data predominantly influences the amount of equivariance learned by relaxed equivariant models. This supports the view that relaxed equivariant models learn the appropriate level of equivariance to match the equivariance level of the training dataset. Furthermore, we have shown that on highly equivariant datasets relaxing the model's equivariance constraint can lead to even more learned equivariance, as the relaxed models have the necessary flexibility to learn equivariance with respect to groups that are larger than the one imposed by their architecture. 
 -->


## Author Contributions
- Nesta: Reproduction of [Wang et al. (2022)](#References), including porting models to lighting and creating configuration. Implementation of experiment scripts using Wandb API. Implementation of Equivariance Error, parts of Hessian Spectra and Sharpness metric. Writing of the analysis in the results section for the experiments using the Smoke Plume Dataset.
- Sebastian: Research of Lie derivatives, Hessians, Sharpness and Writing.
- Jiapeng: Research and implementation of Lie derivatives and Sharpness, Research on Hessians, Writing.
- Thijs: Research on the octahedral group, Implementation of Super-Resolution models and 3D separable group upsampling on gconv. Reproduction code from [Wang et al. (2023)](#References). Writing.
- Diego: Integration with Hydra, Integration with W&B, Implementation of Hessian Spectra, Reproduction code for [Wang et al. (2023)](#References),  Implementation of the JHTDB dataloader, Implementation of octahedral relaxed separable, lifting and regular group convolutions on gconv library, SLURM setup, hyperparameter search.

## References

[1] Wang, R., Walters, R., & Smidt, T. E. (2023). Relaxed Octahedral Group Convolution for Learning Symmetry Breaking in 3D Physical Systems. arXiv preprint arXiv:2310.02299.

[2] Gruver, N., Finzi, M., Goldblum, M., & Wilson, A. G. (2022). The lie derivative for measuring learned equivariance. arXiv preprint arXiv:2210.02984.

[3] Park, N., & Kim, S. (2022). How do vision transformers work?. arXiv preprint arXiv:2202.06709.

[4] Zhao, B., Gower, R. M., Walters, R., & Yu, R. (2023). Improving Convergence and Generalization Using Parameter Symmetries. arXiv preprint arXiv:2305.13404.

[5] Wang, R., Walters, R., & Yu, R. (2022, June). Approximately equivariant networks for imperfectly symmetric dynamics. In International Conference on Machine Learning (pp. 23078-23091). PMLR.

[6] Holl, P., Koltun, V., Um, K., & Thuerey, N. (2020). phiflow: A differentiable pde solving framework for deep learning via physical simulations. In NeurIPS workshop (Vol. 2).

[7]  Y. Li, E. Perlman, M. Wan, Y. Yang, C. Meneveau, R. Burns, S. Chen, A. Szalay & G. Eyink. "A public turbulence database cluster and applications to study Lagrangian evolution of velocity increments in turbulence". Journal of Turbulence 9, No. 31, 2008.

[8] E. Perlman, R. Burns, Y. Li, and C. Meneveau. "Data Exploration of Turbulence Simulations using a Database Cluster". Supercomputing SC07, ACM, IEEE, 2007.

[9] Super-resolution of Velocity Fields in Three-dimensional Fluid Dynamics: https://huggingface.co/datasets/dl2-g32/jhtdb

[10] Weiler, M. and Cesa, G. General E(2)-equivariant steerable CNNs. In Advances in Neural Information Processing Systems (NeurIPS), pp. 14334–14345, 2019b.

[11] Turbulence SuperResolution Replication W&B Report: https://api.wandb.ai/links/uva-dl2/hxj68bs1

[12] Equivariance and Training Stability W&B Report: https://api.wandb.ai/links/uva-dl2/yu9a85jn

[13] Rotation SmokePlume Replication W&B Report: https://api.wandb.ai/links/uva-dl2/hjsmj1u7

[14] `gconv` library for regular group convnets: https://github.com/dgcnz/gconv

[15] Bekkers, E. J., Vadgama, S., Hesselink, R. D., van der Linden, P. A., & Romero, D. W. (2023). Fast, Expressive SE $(n)$ Equivariant Networks through Weight-Sharing in Position-Orientation Space. arXiv preprint arXiv:2310.02970.
