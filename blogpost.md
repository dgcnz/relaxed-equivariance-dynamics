# Effect of equivariance on training dynamics

*Group Equivariant Convolutional Networks* (G-CNN) have gained significant traction in recent years owing to their ability to generalize the property of CNNs being equivariant to translations in convolutional layers. With equivariance, the network is able to exploit groups of symmetries and a direct consequence of this is that it generally needs less data to perform well. However, incorporating such inductive knowledge into the network may not always be advantageous, especially when the data itself does not exhibit full equivariance. To address this issue, the concept of *relaxed equivariance* was introduced, offering a means to adaptively learn the degree of equivariance imposed on the network, enabling it to operate on a level between full equivariance and no equivariance.

Interestingly, for rotational symmetries on fully equivariant data, [[1]](#References) found that a fully equivariant network exhibits poorer performance compared to a relaxed equivariant network. This is a surprising result, since usually fully equivariant models perform the best in cases were the data is also fully equivariant and is precisely what they were designed for. One plausible rationale for why it is different in this case is that the training dynamics benefit from relaxation of the equivariance constraint.  This proposition gains inspiration from [[3]](#References), where they find that there is an optimal amount of inductive bias between strong inductive bias of CNNs and weak inductive bias of ViTs where the training dynamics are best. In that paper, analysis was done using the eigenvalues of the Hessian throughout training and we intend to incorporate this method into our analysis as well.

Since approximate equivariance adaptively learns the amount of equivariance from the data it is important to also understand this process to investigate the training dynamics.
To do this we will investigate how the equivariance possessed by the data at hand influences the training dynamics. Importantly, [[2]](#References) shows that more equivariance imposed on a network does not necessarily imply more equivariance learned. As such we will use more advanced methods to discover the true equivariance that the model expresses. 

Inspired by the aforementioned observations, the purpose of this blog post is to investigate the following question. **How does the equivariance imposed on a network affect its training dynamics?** With support from these observations we also hypothesize that the less equivariance is imposed the better the training dynamics will be. For answering our research question we identify the following subquestions:

1.  How effective is regularization for imposing equivariance on a network?
2.  How does the amount of equivariance imposed on the network affect the amount of equivariance learned?
3.  How does equivariance imposed on a network influence generalization?
4.  How does equivariance imposed on a network influence the convexity of the loss landscape?   


We will answer these questions by:
- Reproducing results to establish common ground
- Performing experiments to investigate learned equivariance
- Analyzing trained models to investigate their training dynamics






<!--Furthermore, to either substantiate some of the aforementioned observations or the effectiveness of approximate equivariance networks, we validate the following claims:

- Posed in [1], an approximate equivariant network outperforms an equivariant network on a fully equivariant dataset (isotropic flow).
- Posed in [5], an approximate equivariant network outperforms an equivariant network on non-equivariant smoke dataset. 
-->
## Reproduction 
First, we introduce some theory needed for the reproduction. After that we will go over the exact experiments we reproduced and our results.
### Regular G-CNN

Consider the segmentation task depicted in the picture below.

![Alt text](https://analyticsindiamag.com/wp-content/uploads/2020/07/u-net-segmentation-e1542978983391.png)

Naturally, applying segmentation on a rotated or reflected image should give the same segmented image as applying such transformation after the segmentation. Mathematically, it means the neural network $f$ should satisfy: 

$$
f(gx) = gf(x)
$$

for all datapoints $x$ and transformations (in this case rotations and reflections) $g$.
A network that satisfies this property is considered to be equivariant w.r.t. the group of transformations comprised of 2D rotations and reflections.

To build such a network, it is sufficient that each of its layers is equivariant in the same sense. Recall in a CNN, its building block, the convolution layer, achieves equivariance to translations by means of weight sharing using kernels that are shifted along the image.
<!--
<span style="color:red;">Insert picture of convolution layer here</span>
-->

Formally, a group $(G,\cdot)$ is a set equipped with a closed and associative binary
operation, containing inverse elements and an identity. Adapting the idea of weight sharing to arbitrary groups of transformations leads to $G$-equivariant group convolution, defined between a kernel $\psi: G \rightarrow \mathbb{R}^{n\times m}$ and a function $f: G \rightarrow \mathbb{R}^m$ on element $g \in G$ as:
$$
    (\psi *_{G} f)(g) = \int_{h \in G}\psi(g^{-1}h)f(h)
$$
When $G$ is the group of translations, this reduces to the regular convolution operation, which makes G-CNNs a generalization of CNNs, convolving over arbitrary groups.
<!--
LIFTING CONVOLUTION
The function $f$ represents a hidden layer of the G-CNN, and it's important to notice that its domain is the group $G$. The first layer of the neural network is usually an image, i.e. a function defined on $\mathbb{R}^2$...
-->

For the group convolution to be practically feasible, $G$ has to be finite and relatively small in size (roughly up to a hundred elements). 
However, if one is interested in equivariance w.r.t. an infinite group, e.g. all 2D rotations, $G$ would have to be a finite subset of those rotations, and the integral reduces to a summation. In this case, it is unclear to what extent such a network is truly rotationally equivariant.

### Steerable G-CNN
To address the aforementioned equivariance problem w.r.t. the group of 2D rotations $\rho: G \rightarrow \mathbb{R}^{2\times 2}$ parameterized by the infinite group $G$, $G$-steerable convolution is proposed based on the following observation:

Rotation of a **steerable** kernel $\psi: \mathbb{R}^2 \rightarrow \mathbb{R}$ by representation over G (the rotation operator) $\rho: G \rightarrow \mathbb{R}^{2\times 2}$ may be expressed as a linear combination of $Q$ basis kernels: $$\rho_g \psi(x) := \psi(\rho_{-g}x) = \sum_{q=1}^Q w_q(g) \psi_{q}(x)$$
This implies that convolution of a function $f$ with a rotated steerable kernel is the linear combination of convolution with the rotated steerable basis kernels: $$(\rho_g \psi *_{G} f)(x) = \sum_{q=1}^Q w_q(g) (\rho_g \psi_q *_{G} f)(x)$$ With this, we see that $\psi$ is steerable in the sense that the output of the convolution is being steered with the orientation of the kernel. We spare the readers from the technicalities of calculating the weights, which is done via something called **irreducible representations of $\rho$**.

The reason that with this construction, equivariance w.r.t. a infinite group is achievable is because those irreducible representations form a basis for some of the functions defined over $G$. Therefore, picking the weights appropriately facilitates  the desired equivariance when the basis of irreducible representations are finite. 


### Relaxed Equivariant Networks

The desirability of equivariance in a network depends on the amount of equivariance possessed by the data of interest. To this end, relaxed equivariant networks are built on top of G-CNNs using a modified (relaxed) kernel consisting of a linear combination of standard G-CNN kernels.

<!-- $$ (\psi \hat{*}_{G} f)(g) = \sum_{h \in G}\psi(g,h)f(h) = \sum_{h \in G}\sum_{l=1}^L w_l(h) \psi_l(g^{-1}h)f(h) $$ -->
$$ 
    (\psi *_G f)(g) = \sum_{h \in G}\psi(g,h)f(h) = \sum_{h \in G}\sum_{l=1}^L w_l(h) \psi_l(g^{-1}h)f(h) 
$$

$G$-equivariance of the group convolution arises from kernel $\psi$'s dependence on the composite variable $g^{-1}h$, rather than on both variables $g$ and $h$ separately. This property is broken in relaxed kernels, which leads to a loss of equivariance.

The equivariance error increases with the number of kernels $L$ and the variability of $w_l(h)$ over $h \in G$, allowing us to control the amount of equivariance imposed on the model by adding the term: 

$$ \alpha \sum_{l=1}^L\sum_{g,h \in G}|w_l(g)-w_l(h)|
$$

to the loss function. A higher value of the hyperparameter $\alpha$ enforces a higher degree of equivariance, by forcing the weights to be nearly constant functions.

Therefore, using relaxed group convolutions allows the network to relax strict symmetry constraints, offering greater flexibility at the cost of reduced equivariance.


<!--Naturally, we would expect approximately equivariant networks to achieve better results than fully equivariant models on datasets which are themselves not perfectly equivariant.
[1] supports this intuition showing that an AENN yielded better results than a fully equivariant model on super-resolution tasks for partially-equivariant channel flow data.
Interestingly, the AENN prevailed even in the fully-equivariant isotropic flow dataset, which could potentially be explained by AENN weights enhancing optimization.-->

## Reproduction Objective and Method

Our first objective is to reproduce the surprising experiment where a relaxed equivariant model outperforms an equivariant one on fully equivariant data. Namely, we intend to reproduce the Super-resolution of 3D Turbulence experiment (Experiment 5.5) as presented in the paper "Discovering Symmetry Breaking in Physical Systems with Relaxed Group Convolution" [[1]](#References).

In this study, the authors focus on understanding the asymmetries in physics data using relaxed equivariant neural networks. They employ various relaxed group convolution architectures to identify symmetry-breaking factors in different physical systems. 

This part of the reproduction allows us to investigate the effects of approximate equivariance on the training dynamics later on in the extension chapter. We will do so by looking at the differences between a non-equivariant model, an approximate equivariant one and a fully equivariant one.

Secondly, to ensure relaxed group convolutions perform as shown in Wang 2024 [[1]](#References), we reproduce results from the paper that introduced them, Wang 2022 [[5]](#References). In this paper relaxed group convolutions are compared to other methods on 2D smoke simulation data. We intend to do the same, focusing on the experiments that fit our first objective the closest, namely the ones involving rotational symmetries. 
<!--
Note: The following sentence is NOT true, we have a different dataset for reproduction and for analysis.
We will again evaluate the training dynamics of our methods on this smoke simulation dataset. -->


### Super Resolution

In Experiment 5.5 of Wang 2024, the authors evaluate the performance of regular convolutional layers, group equivariant layers, and relaxed group equivariant layers integrated into a neural network. This network is tasked with upscaling 3D channel isotropic turbulence data.

The data consists of liquid flowing in 3D space and is produced by a high resolution state of the art simulation hosted by the John Hopkins University [[7][8]](#References). Importantly, this simulation is forced to be isotropic (equivariant to rotation) by design. 

For the purposes of the experiment a subset of 50 timesteps was taken and downsampled from 1024^3 to 64^3 and processed into a task suitable for learning. The model is given an input of 3 consecutive timesteps, which are downsampled again to 16^3, and is tasked to predict and to upsample to the following timestep (which is still 64^3). We publish our data processing scripts and production-ready dataloader on HuggingFace[[9]](#References).

The architecture of the models is exactly as in described in Wang 2024, with the following additions that were not specified in the paper. The first layer of the relaxed and the non-relaxed group convolution networks were lifting layers. Every convolution (that was not an upconvolution) used stride 1, padding 1 and dilation 1 to preserve spatial size. In the Upconv layers (Transposed Convolutions), stride 2, padding 1, dilation 1 and output padding 1 were used, ensuring the spatial dimensions doubled in each of these layers. Separable group convolutions were used, for the Upconv layers this meant that upsampling was only done on the spatial elements, a non-upconv convolution was done on the group elements. Additionally, batch norm and ReLu where used after every convolution. Furthermore, for training the Adam optimizer was used with a learning rate of 0.01. 



#### Expectations Based on Equivariance


- For isotropic turbulence, networks with full equivariance should either outperform or be on par with those with relaxed equivariance, as the data fully adheres to isotropic symmetry. However, as Wang et. al 2024 notes, the opposite happens. 
<!--
#### Results and Observations

Their results indicate that incorporating both equivariance and relaxed equivariance enhances prediction performance. Interestingly, the relaxed group convolution outperformed even the fully equivariant network on isotropic turbulence data, which contradicts initial expectations.

This unexpected outcome may be attributed to the enhanced optimization process facilitated by the relaxed weights, and serves as the main motivating factor for our extension.
 -->
 
### Smoke Plume
For this experiment a specialized 2D smoke simulation was generated using Phiflow. In these simulations, smoke is added from an inflow point and flows in the direction of the buoyant force (ex. in everyday life the buoyant force opposes gravity and thus smoke floats up). Across simulations, the inflow positions vary. Additionally, every inflow position has a slightly different buoyant force, varying in either strength or angle. Finally, for a given inflow position the simulation is ran multiple times where each time its buoyant force is modified by a scalar factor to increase the difference between buoyant forces. In total, this results in 4 inflow positions each ran 10 times for 311 timesteps.

The construction of this dataset is such that it is not fully equivariant to rotations but also not completely non-equivariant. On a small scale, smoke will flow the same way regardless of rotation as it is mostly influenced by the smoke particles around it. However, on a larger scale the influence of the specific angle of the buoyant force on movement is larger. This due to the fact that some directions will have stronger buoyant forces than others. Of note is that since all the tested models are equivariant with respect to translation as they are based on CNN's. Meaning that the exact location of the inflow position is not generally important.  

Using this dataset, the task is to predict the following frames based on the previous ones. Evaluation on this task is done in two settings, domain and future. In the domain task the model is tested on inflow locations it was not trained on. On the other hand, in the future task the model is tested on timesteps that are further in the simulation than what it was trained on.
#### Expectations Based on Equivariance
- Since the dataset is partially equivariant to rotation, the partial equivariant model is expected to perform the best.
## Reproduction Results
<!--We aim to reproduce the experiment of [1] using a 64x64 synthetic smoke dataset which has rotational symmetries. Specifically the data contains 40 simulations varied by inflow positions and buoyant forces, which exhibit perfect C4 rotational symmetry. However, buoyancy factors change with inflow locations, disrupting this symmetry.-->

### Super Resolution 

We compare our results with those of (Wang et. al, 2024)[1] for the CNN (SuperResCNN), regular group equivariant network (GCNNOhT3) and relaxed regular group equivariant network (RGCNNOhT3). The reconstruction mean absolute error is found in the table below.

<div style="display: flex;">
  <table border="0">
    <tr>
      <th colspan="3">Results from [1]</th>
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
      <caption style="caption-side: bottom">Mean absolute error (MAE), Scale: 1e-1 </caption>
  </table>

  <table border="0" > <!-- Adding space between the tables -->
    <tr>
      <th colspan="3">Reproduction Results</th>
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
      <caption style="caption-side: bottom">Mean absolute error (MAE), Scale: 1e-1 </caption>
  </table>
</div>

We see that all the models perform a bit better, but the overall intriguing trend remains: That relaxed equivariant model performs better than the fully equivariant model on equivariant data.

Although we made sure to have comparable number of parameters, we can go a step further and normalize the above metrics by each model's parameter counts to analyze parameter-efficiency.

<div style="display: flex;">
  <table border="0">
    <tr>
      <th colspan="3">Number of parameters</th>
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
  </table>

  <table border="0" > <!-- Adding space between the tables -->
    <tr>
      <th colspan="3">MAE per 1e6 parameters</th>
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

Here we observe, that the relaxed equivariant network is more parameter-efficient than the fully equivariant network.

### Smoke plume

We compare our results with those of the original paper for the rsteer and rgroup models, which are the ones the paper introduces. The reconstruction RMSE for both methods is found in the table below. 

<div style="display: flex;">
  <table border="0">
    <tr>
      <th colspan="3">Results from [5]</th>
    </tr>
    <tr>
      <td></td>
      <td>rgroup</td>
      <td>rsteer</td>
    </tr>
    <tr>
      <td>Domain</td>
      <td>0.73(0.02)</td>
      <td>0.67(0.01)</td>
    </tr>
    <tr>
      <td>Future</td>
      <td>0.82(0.01)</td>
      <td>0.80(0.00)</td>
    </tr>
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
      <td>0.90(0.04)</td>
      <td>0.67(0.00)</td>
    </tr>
    <tr>
      <td>Future</td>
      <td>0.88(0.03)</td>
      <td>0.82(0.00)</td>
    </tr>
  </table>
</div>

We see that the rgroup performs quite a bit worse than what was presented in [[1]](#References), with especially the in domain performance being 0.2 lower. One potential reason for this is that the original paper does not provide the hyperparameters used to obtain their results, and we did not perform a grid search over the provided grid.

We see that the rsteer model performs similarly to what was seen in [[1]](#References), the results on in domain data are slightly better while on future data the performance is a bit worse. Possibly due to a difference in the early stopping metric used. Overall we can conclude that the models presented in the original paper work as expected.  

### Reproduction Efforts

To maximize reproducibility and future usability, we provide config files for all the experiments, models, datasets, trainers, etc. using Hydra and PyTorch Lightning (more information on the README). This means that all the models are wrapped on Lightning Modules and all datasets (SmokePlume, JHTDB) are uploaded to HuggingFace and have a corresponding Lightning DataModule. We reuse and upgrade the data generation scripts for the SmokePlume datasets from [[5]](#References) and implement a configurable data generation and HuggingFace-compatible data loading script from scratch for the JHTDB Dataset. Furthermore, we integrate our code with Weights and Biases and publish all the relevant runs and plots on publicly accessible reports [[11], [12], [13]](#References). Finally, for the relaxed regular group convolutional networks, we implement all components on our fork of the `gconv` library [[14]](#References).


To summarize briefly the missing/added reproduction code:
- For (Wang et. al, 2022)[5] we add the missing weight constraint and hyperparameters for rgroup.
- For (Wang et. al, 2024)[1]:
  - All models (rgcnn, gcnn, cnn), where we implement 3d relaxed separable convolutions, octahedral group convolutions, 3d equivariant transposed, convolutions, and we made educated guesses on which activations, normalizations to use and where to place them (along skip and upsampling residual connections).
  - The JHTDB dataset, where we implement all the subsampling, preprocessing and loading logic of the 3d turbulence velocity fields.


# Extension and Analysis
As the results of the reproduction match those in their respective papers, we are free to conduct several analyses using approximate equivariance. For these experiments we introduce a dataset that is very similar to the 2D smoke dataset seen in the Reproduction. Additionally, we analyze trained models to learn about their training dynamics. The techniques used for this will be explained in following theory section. The results from these analyses we will be able to answer our research questions and in doing so allow us to discover the, perhaps crucial, role of training dynamics in the performance of equivariant models.



## Theory for Analysis 
### Measuring the Amount of Equivariance Learned by a Network
It is natural to measure the amount of equivariance a network $f$ has as the expected difference between the output of the transformed data and the transformed output of the original data.

$$ \mathbb{E}_{x \sim D}\left[\frac{1}{|G|}\sum_{g \in G}\|f(gx)-gf(x)\|\right]$$

We can estimate this expectation by computing the average for a series of batches from our test set, [[1]](#References) uses this and call . However this approach has downsides, which we can tackle using the Lie derivative. 

### Lie Derivative
<!--
<span style="color:red;">Add Lie group def and say that G' usually has that structure?</span>
-->


<!-- However, this approach is problematic as it only measures the amount of equivariance w.r.t. the finite group $G$. Instead, [2] proposed the use of (Lie) derivatives to evaluate the robustness of the network to infinitesimal transformations. For the notion of derivative to be defined, however, we need to assume the group to have a differential structure (Lie group). Since the space consisting of $G$ may be too peculiar to work in, we smoothly parameterize the representations of these transformations in the tangent space at the transformation that does nothing (identity element).  -->

In practice, even though we are imposing $G$-equivariance on a network, what we would like to achieve is $G'$-equivariance for an infinite (Lie) group $G'$ which contains $G$. The previous approach is problematic as it only measures the amount of acquired equivariance w.r.t. the finite group $G$, neglecting all other transformations, and thus doesn't give us the full picture of the network's equivariance.
 
 [[2]](#References) proposed the use of Lie derivatives, which focus on the equivariance of the network towards very small transformations in $G'$, and give us a way to measure $G'$-equivariance of the network. The intuitive idea is the following: Imagine a smooth path $p(t)$ traversing the group $G'$ that starts at the identity element (i.e. transformation that does nothing) of the group, $e_{G'}$. This means that at every time-point $t \geq 0$, $p(t)$ is an element of $G'$ (some transformation), and $p(0) = e_{G'}$. Then, we can define the function:
  $$\Phi_{p(t)}f(x) := p(t)^{-1}f(p(t)x)
  $$
  This function makes some transformation $p(t)$ on the data, applies $f$ to the transformed data, and finally applies the inverse transformation $p(t)^{-1}$ to the output. Notice that if $f$ is $G'$-equivariant this value is constantly equal to $f(x)$, and that $\Phi_{p(0)}f(x) = f(x)$. The Lie derivative of $f$ along the path $p(t)$ is the derivative 
  $$L_pf(x) := d\Phi_{p(t)}f(x)/dt = \lim_{t \to 0+} \frac{\Phi_{p(t)}f(x) - f(x)}{t}
  $$
at time $t=0$. One might note that this only measures the local equivariance around the identity element of the group. Luckily, it is shown in [[2]](#References) that $L_{p}f(x) = 0$ for all $x$ and $d$ specific paths, where $d$ corresponds to the dimensionality of $G'$, is equivalent to $f$ being $G'$- equivariant.

<!-- 
### Equivariance error (EE)

 Another alternative for measuring equivariance relies on the variant of approximate equivariance network we consider. Recall that what broke equivariance therein are the weights used in the linear combination of kernels that constituted the modified kernel. Therefore, a proxy for the amount of equivariance is naturally the difference between the individual kernels used over all possible transformations.
$$\frac{1}{L|G|}\sum_{l=1}^L\sum_{g \in G} |w_l(g)-w_l(e)|$$ 

-->

### Training Dynamics Evaluation

To assess the training dynamics of a network, we quantify both the efficiency and efficacy of learning. 

To estimate the efficiency of learning, we analyze:
* convergence rate as measured by number of epochs trained until convergence
* learning curve shape


For efficacy of learning, we are interested in the final performance and the generalizability of the learned parameters, which are quantified by the final RMSE, and sharpness of the loss landscape near the final weight-point [[4]](#References). 

#### Sharpness

To measure the sharpness of the loss landscape after training, we consider changes in the loss averaged over random directions. Let $D$ denote a set of vectors randomly drawn from the unit sphere, and $T$ a set of displacements, i.e. real numbers. Then, the sharpness of the loss $\mathcal{L}$ at a point $w$ is: 

$$ \phi(w,D,T) = \frac{1}{|D||T|} \sum_{t \in T} \sum_{d \in D} |\mathcal{L}(w+dt)-\mathcal{L}(w)| 
$$

This definition is an adaptation from the one in [[4]](#References) which does not normalize by $\mathcal{L}(w)$ inside the sum. 
The perturbation of the weights in $\mathcal{L}(w + dt)$ can be understood as follows: first, flatten all the weight matrices into a single vector. Then, add a unit vector of the same shape to this vector. Finally, reshape the resulting vector back into the original set of weight matrices. A sharper loss landscape around the model's final weights, usually implies a greater generalization gap.

#### Hessian Eigenvalue 

Finally, the Hessian eigenvalue spectrum [[3]](#References) sheds light on both the efficiency and efficacy of neural network training. Negative Hessian eigenvalues de-convexify the loss landscape disturbing the optimization process, whereas very large eigenvalues lead to training instability, sharp minima and consequently poor generalization.



## Extension Objectives and Method

Our experiments are split up into two parts. First we examine the impact of equivariance imposed and data equivariance on the amount of equivariance learned, answering subquestions 1 and 2. We do this by computing the Equivariance Error and Lie Derivative as described in the previous section. We plot these measures for varying levels of imposed equivariance and data equivariance.  

Second we examine how equivariance imposed on a network influences convexity of the loss landscape and generalization, answering subquestions 3 and 4. We can strongly impose equivariance on a network through architecture design, and we can weakly impose equivariance on a network through a regularization term in the loss of the relaxed models. However, as we will see the weak method has minimal effects and so the strong method is used.  We train multiple models with different levels of imposed equivariance on two fully equivariant datasets, namely Isotropic Flow and 2D Smoke Plume with Varying Equivariance which is introduced below. For these models, we evaluate generalization by looking at the training and validation loss curves of different models and computing sharpness for different training epochs. We examine convexity of the loss landscape by computing hessian spectra for the models at different training epochs.


### Imposing Equivariance
#### Smoke Plume with Varying Equivariance

For this experiment, we use a synthetic $64 \times 64$ 2D smoke simulation dataset generated by PhiFlow [[5]](#References) similar to the one used in the reproduction with varying amounts of equivariance. In contrast to the other one, this dataset has a fixed inflow position. For this dataset, the buoyant force is pointed either up, left or right (i.e. the C4 group). To vary the equivariance then, the strength of the force is different in each direction. The larger the difference between the strength of the forces the less equivariant the data becomes. Of course, when the forces are exactly the same the data is fully equivariant to the C4 group as the simulation should play roughly the same in every direction.

In total there are 10 different settings of the equivariance level. Each of these include 4 simulations of 60 timesteps.

We use two models for our analysis on this dataset. The first model is the Rsteer model from [[4]] as introduced in [Reproduction Background](#Reproduction-Background) this models posses relaxed equivariance with respect to the C-4 rotation group. The second model is the E2CNN model from [[10]](#References), which is fully equivariant to the C-4 rotation group. Both models have 600 thousand parameters for a fair comparison. 

Using these models means we have two ways of imposing equivariance. Strongly, by using the E2CNN model architecture which is strictly equivariant and has a very similar architecture to the rsteer model. Or weakly by adding a regularization term to the loss in the relaxed rsteer model. Namely $\alpha$ as introduced in the [Relaxed Equivariant Networks](#Relaxed-Equivariant-Networks) section.

### Training Dynamics
#### Smoke Plume with full Equivariance
For this experiment we use the [Smoke Plume dataset](#Smoke-Plume-with-Varying-Equivariance) introduced in the section above using its fully equivariant setting. We analyze the model checkpoints corresponding to the third and best epochs during training. Where best means highest validation RMSE. As we will see, the regularization using alpha has minimal impact. Therefore, we use the Rsteer model with $\alpha$=0 and the E2CNN model introduced above. Both models are trained using Adam with weight decay 0.0004 and learning rate 0.001.


#### Super Resolution
We the third epoch using sharpness







## Results: Measuring Learned Equivariance

<table>
  <tr>
    <td>
      <img src="https://hackmd.io/_uploads/H1tTMORQR.png" alt="Figure 1" style="max-width: 100%;">
      <p align="center">Figure 1: Impact of equivariance imposed on Model Equivariance Error </p>
    </td>
    <td>
      <img src="https://hackmd.io/_uploads/rytpfdRmA.png" alt="Figure 2" style="max-width: 100%;">
      <p align="center">Figure 2: Impact of equivariance imposed on Lie Derivative of the Model</p>
    </td>
  </tr>
</table>

Figure 1 shows the Equivariance Error of different model specifications. The Equivariant Net is the E2CNN model, it is positioned  to the right of the x-axis because we can think of a fully equivariant net as a relaxed equivariant net where alpha is set to infinity. Note that this is purely conceptual, setting alpha to a very high value would not have this effect in practice. 

For Rsteer we observe that the Data Equivariance has a large effect on how equivariant the model learns to be. This shows the relaxed architecture can adapt well to the equivariance of the data, which matches the findings in [[5]](#References). However we see that the hyperparameter $\alpha$ has barely any effect on the Equivariance Error of the model, as alpha is increased the amount of Equivariance Error should decrease but it does not. We thus find that regularization is not very effective for imposing equivariance on a network. For E2CNN, the Equivariant Net, we see that the Equivariance Error is near zero for all levels of data equivariance. This is expected because the equivariance to C-4 rotation is build into the model, thus the Equivariance Error can only come from artifacts created when rotating the input or the output feature maps. 

Figure 2 shows the Lie derivative for different model specifications. A lower Lie Derivative means the model is more equivariant to the complete rotation group. For Rsteer we see similar results to figure 1. However for E2CNN we do not see a zero lie derivative, this is because the architecture ensures equivariance to just the C-4 group.  

Interestingly, Rsteer exhibits a lower Lie derivative than E2CNN when trained on fully equivariant data. This could be due to Rsteer's greater flexibility, allowing it to learn equivariance with respect to a broader group of rotations beyond C-4. In contrast, E2CNN achieves perfect C-4 equivariance but struggles to generalize to other rotations due to its architectural constraints, even when the training data is fully equivariant with respect to all rotations.


## Results: Training Dynamics 
### Smoke Plume with full Equivariance
In this section we will give further support to the statement that relaxed equivariant networks can outperform  fully equivariant networks on perfectly equivariant datasets, and analyze the training dynamics as a potential explanation of this phenomenon.

First we examine the training, validation and test RMSE for the E2CNN and Rsteer models on fully equivariant data. 
<table>
  <tr>
    <td>
      <img src="https://hackmd.io/_uploads/ryueMnyER.png" alt="Figure 3" style="max-width: 100%;">
      <p align="center">Figure 3: Train RMSE curve</p>
    </td>
    <td>
      <img src="https://hackmd.io/_uploads/Hkulf2kE0.png" alt="Figure 4" style="max-width: 100%;">
      <p align="center">Figure 4: Validation RMSE curve</p>
    </td>
  </tr>
</table>


<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/rkIw-fbEA.png" alt="Figure 5" style="max-width: 100%;">
  <p>Figure 5: test RMSE for best models</p>
</div>


Figure 3 and 4 show the train and validation RMSE curves respectively. We see that on the training data Rsteer and E2CNN perform similarly, however on the validation set the curve for Rsteer lies below the one for E2CNN. Therefore, the relaxed model Rsteer generalizes better to the validation set. Figure 5 shows the test set RMSE for the two models averaged over five seeds. 

We find that the relaxed equivariant model performs better, even though the data is fully C-4-equivariant, reaffirming the results obtained on the Isotropic Flow dataset.

To obtain insight into why the relaxed equivariant models outperform the fully equivariant ones on these datasets we will look at the hessian spectra and the sharpness of the loss landscape of these trained models.

<table>
  <tr>
    <td>
      <img src="https://hackmd.io/_uploads/rJJQKAyNC.png" alt="Epoch 3" style="max-width: 100%;">
      <p align="center">Figure 5: Description</p>
    </td>
    <td>
      <img src="https://hackmd.io/_uploads/S1jQF0JN0.png" alt="Epoch best" style="max-width: 100%;">
      <p align="center">Figure 6: Description</p>
    </td>
  </tr>
</table>

Figures 5 and 6 show hessian spectra for the same early and best checkpoints of E2CNN and Rsteer as used for the previous analysis. With regards to flatness of the loss landscape these plots allow us to make a similar conclusion. We see that for both checkpoints E2CNN has much larger eigenvalues than rsteer, this can lead to training instability, sharp minima and consequently poor generalization for E2CNN.

To evaluate the convexity of the loss landscape we need to look at the negative eigenvalues in the Hessian Spectra. We see that for both models, neither spectra shows any negative eigenvalues. This suggests that both the fully equivariant E2CNN and the relaxed Rsteer models exhibit "mostly convex" loss landscapes. Thus convexity of the loss landscapes does not seem to play a large role in the performance  


<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/S14l5XZ4A.png" alt="Figure 5" style="max-width: 100%;">
  <p>Figure 7: Description</p>
</div>

Next we examine checkpoints for the two models trained on the Smoke Plume Dataset with 0-equivariance error. We specifically look at a checkpoint during early training, epoch three, and the best model checkpoint. We see that the rsteer model has a significantly lower sharpness of the loss landscape for the best weights compared to E2CNN, which indicates a lower generalization gap, and thus more effective learning. We similarly observe lower sharpness for rsteer with early model weights, this matches with the lower validation RMSE curve for rsteer we saw earlier. Although a flatter loss landscape is not necessarily advantageous for faster convergence during training, it is interesting that the Rsteer model exhibits a generally flat loss landscape, rather than simply converging to a flat minimum.

### Super Resolution

Similarly to the previous subsection, we'll analyze the training dynamics of the superresolution models on the isotropic JHTDB dataset as a potential explanation for the superiority of the relaxed equivariant model over the fully equivariant one.


First we examine the training and validation Mean Average Error (MAE) curves for the Relaxed Equivariant (RGCNN), Fully Equivariant (GCNN) and non-equivariant (CNN) models (run on 6 different seeds).

<table>
  <tr>
    <td>
      <img src="https://hackmd.io/_uploads/BkHn_Of4C.png" alt="Figure 8" style="max-width: 100%;">
      <p align="center">Figure 8: Train MAE curve</p>
    </td>
    <td>
      <img src="https://hackmd.io/_uploads/BJ_A_uMER.png" alt="Figure 9" style="max-width: 100%;">
      <p align="center">Figure 9: Validation MAE curve</p>
    </td>
  </tr>
      <caption style="caption-side: bottom">Legend: RGCNN (pink), GCNN (cyan), CNN (grey). Shaded surrounding color represents standard devation.  </caption>
</table>

Here we can observe that pretty early (around epoch 3) RGCNN starts outperforming the other two models and keeps the lead until saturating around 0.1 MAE. As such, we'll take a checkpoint for each model on epoch 3 (early) and on its best epoch (Best) and examine the loss' sharpness at that point. Although we wanted to also compute the Hessian spectra, that wasn't possible since the second derivative of the 3d grid sampler used in both equivariant networks isn't yet implemented on PyTorch (see [PyTorch #34704](https://github.com/pytorch/pytorch/issues/34704)). 

<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/SkpgREzV0.png" alt="Figure 10" style="max-width: 100%;">
  <p>Figure 10: Description</p>
</div>

In any case, as can be seen in Figure 10 the sharpness of the loss landscape was the lowest for the relaxed equivariance model in both the early checkpoint and in the best checkpoint. This supports hypothesis and our previous findings on the SmokePlume dataset.


## Concluding Remarks

We reproduced two experiments: Smoke Plume originally conducted by (Wang et. al 2022) [[5]](#References) and Super Resolution by (Wang et al. 2024) [[1]](#References). Our results align with the original findings, reaffirming the effectiveness of relaxed equivariant models and demonstrating that they can outperform fully equivariant models even on perfectly equivariant datasets.

We further investigated the authors' speculation that this superior performance could be due to relaxed models having enhanced training dynamics. Our experiments empirically support this hypothesis, showing that relaxed models exhibit lower validation error, a flatter loss landscape around the final weights, and smaller Hessian eigenvalues, all of which are indicators of improved training dynamics and better generalization.

Our finding that relaxed models outperform fully equivariant models can be related to previous literature examining the impact of inductive bias on training dynamics. Specifically, [[3]](#References) highlights that the weak inductive bias of Vision Transformers (ViTs) results in a flat and non-convex loss landscape, compared to the sharp and convex loss landscape of CNNs. They introduced additional spatial locality to ViTs to limit non-convexity while still maintaining a flatter landscape. Similarly, for the Isotropic Flow dataset, we found that the relaxed model outperformed both the CNN and GCNN, attributable to a flatter loss landscape. 

Additionally, we demonstrated that the amount of equivariance in the data predominantly influences the amount of equivariance learned by relaxed equivariant models. This supports the view that relaxed equivariant models learn the appropriate level of equivariance to match the equivariance level of the training dataset. Furthermore, we have shown that on highly equivariant datasets relaxing the model's equivariance constraint can lead to even more learned equivariance, as the relaxed models have the necessary flexibility to learn equivariance with respect to groups that are larger than the one imposed by their architecture.  


## Author Contributions
- Nesta: Reproduction of [[5]](#References), including porting models to lighting and creating configuration. Creating experimentation scrip using Wandb API. Implementing Equivariance Error and parts of Hessian Spectra and Sharpness metric. Writing the analysis in the results section for the experiments using the Smoke Plume Dataset.
- Sebastian: Researching Lie derivatives, Researching Hessians, Researching Sharpness
- Jiapeng: Researching and implementing Lie derivatives, Researching Hessians, Researching Sharpness
- Thijs: Adapting code from gconv, Researching the octahedral group, Implementing Super-Resolution models, Implementing of 3D group upsampling, Reproducing code from [[5]](#References), Researching and implementing upconv for separable group convolutions.
- Diego: Integration with Hydra, Integration with W&B, Adapting and researching the gconv library, Implementation of Hessian Spectra, Reproduction code for [[5]](#References), Processing and implementation of the JHTDB dataset, Implementation of octahedral (relaxed) separable/lifting/regular group convolutions, SLURM setup.

## References

[1] Discovering Symmetry Breaking in Physical Systems with Relaxed Group Convolution

[2] The Lie Derivative for Measuring Learned Equivariance

[3] How do vision transformers work?

[4] Improving Convergence and Generalization using Parameter Symmetries

[5] Approximately Equivariant Networks for Imperfectly Symmetric Dynamics

[6] PhiFlow: A differentiable PDE solving framework for deep learning via physical simulations.

[7]  Y. Li, E. Perlman, M. Wan, Y. Yang, C. Meneveau, R. Burns, S. Chen, A. Szalay & G. Eyink. "A public turbulence database cluster and applications to study Lagrangian evolution of velocity increments in turbulence". Journal of Turbulence 9, No. 31, 2008.

[8] E. Perlman, R. Burns, Y. Li, and C. Meneveau. "Data Exploration of Turbulence Simulations using a Database Cluster". Supercomputing SC07, ACM, IEEE, 2007.

[9] Super-resolution of Velocity Fields in Three-dimensional Fluid Dynamics: https://huggingface.co/datasets/dl2-g32/jhtdb

[10] Weiler, M. and Cesa, G. General E(2)-equivariant steerable CNNs. In Advances in Neural Information Processing Systems (NeurIPS), pp. 14334–14345, 2019b.

[11] Turbulence SuperResolution Replication W&B Report: https://api.wandb.ai/links/uva-dl2/hxj68bs1

[12] Equivariance and Training Stability W&B Report: https://api.wandb.ai/links/uva-dl2/yu9a85jn

[13] Rotation SmokePlume Replication W&B Report: https://api.wandb.ai/links/uva-dl2/hjsmj1u7

[14] `gconv` library for regular group convnets: https://github.com/dgcnz/gconv
