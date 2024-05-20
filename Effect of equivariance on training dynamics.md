# Effect of equivariance on training dynamics

*Group Equivariant Convolutional Networks* (G-CNN) have gained significant traction in recent years owing to their ability to generalize the property of CNNs being equivariant to translations in its convolutional layers. With equivariance, the network is able to exploit groups of symmetries and a direct consequence of this is that it generally needs less data to perform well. However, incorporating such inductive knowledge into the network may not always be advantageous, especially when the data itself does not exhibit full equivariance. To adress this issue, the concept of *approximate equivariance* was introduced, offering a means to adjust the degree of equivariance imposed on the network, enabling it to operate on a level between full equivariance and no equivariance.

Interestingly, for rotational symmetries on fully equivariant data, [1] found that a fully equivariant network exhibits poorer performance compared to a partially equivariant network. One plausible rationale for this phenomenon is that the training dynamics benefit from relaxation of the equivariance constraint.  This proposition gains support from [3], which conducts an analysis on the spectra of maximum eigenvalues of the Hessian throughout training. This also calls for the question of how the amount of equivariance possessed by the data at hand influences the training dynamics. Equally intriguing, [2] shows that more equivariance imposed does not necessarily imply more equivariance learned by the network.

Inspired by the aforementioned observations, the purpose of this blog post is to investigate the following three relationships, where the first question encompasses two relationships:

- How does (1) the amount of equivariance imposed on the network and (2) the amount of equivariance possessed by the data affect the amount of equivariance learned by the network?
- How does the amount of equivariance imposed affect the training dynamics of the network?


Furthermore, to either substantiate some of the aforementioned observations or the effectiveness of approximate equivariance networks, we validate the following claims:

- Posed in [3], an approximate equivariant network outperforms an equivariant network on a fully equivariant dataset (isotropic flow).
- Posed in [5], an approximate equivariant network outperforms an equivariant network on non-equivariant smoke dataset. 

Prior to presenting the results, let us first recapitulate on G-CNN, approximate equivariant networks, and describe the data and the model used for the experiments, as well as, the metrics used to study those relationships.

## G-CNN

Consider the segmentation task depicted in the picture below.

![Alt text](https://analyticsindiamag.com/wp-content/uploads/2020/07/u-net-segmentation-e1542978983391.png)


Naturally, applying segmentation on a rotated or reflected image should give the same segmented image as applying such transformation after the segmentation. Mathematically, it means the neural network $f$ should satisfy: 

$$
f(gx) = gf(x)
$$

for all datapoints $x$ and transformations (in this case rotations and reflections) $g$.
A network that satisfies this property is considered to be equivariant w.r.t. the group of transformations comprised of 2D rotations and reflections.

To build such as network, it is sufficient that each of its layers are equivariant in the same sense. Recall in a CNN, its building block, the convolution layer, achieves equivariance to translations by means of weight sharing using kernels that are shifted along the image.
<!--
<span style="color:red;">Insert picture of convolution layer here</span>
-->

Adapting this idea of weight sharing to arbitrary groups of transformations led to $G$-equivariant group convolution, where $G$ is a group (for our purpose, the technical details are irrelevant, think of it as a set of transformations), defined between kernel $\psi: G \rightarrow \mathbb{R}^{n\times m}$ and $f: G \rightarrow \mathbb{R}^m$ on element $g \in G$ is defined as
$$
    (\psi *_{G} f)(g) = \sum_{h \in G}\psi(g^{-1}h)f(h)
$$
when $G$ is the group of translations, it reduces to the regular convolution. G-CNN is virtually generalization of CNN, convolving over any group instead of the spacial domain. For the group convolution to be practically feasible, $G$ has to be finite and relatively small in size (roughly up to a hundred elements). 

However, if one is interested in equivariance w.r.t. all 2D rotations, which includes an infinite number of transformations, $G$ would have to be a finite subset of those transformations, and it is unclear to what extent the network is truly rotationally equivariant.

## Approximate equivariant network

The desirability of equivariance for a network relies on the amount of equivariance possessed by the data of interest on which prediction is made. To this end, approximately equivariant networks builds on top of G-CNN using a modified kernel consisting of a linear combination of standard G-CNN kernels with weights that vary with the group element.

$$ (\psi \hat{*}_{G} f)(g) = \sum_{h \in G}\psi(g,h)f(h) = \sum_{h \in G}\sum_{l=1}^L w_l(h) \psi_l(g^{-1}h)f(h) $$

This allows the network to break symmetry constraints, making it more flexible on the expense of having less equivariance.

## Data and model

For our experiments, we use a synthetic $64 \times 64$ smoke dataset generated by PhiFlow [5] with varying boundary conditions and inflow positions. This dataset is rotationally equivariant if the buoyancy force is kept the same and only varying the inflow positions. This means, by varying the buoyancy force for simulations with different inflow positions, we are able to control the equivariance error possessed by the data. For our experiments, we specifically generate three such datasets with varying amounts of rotational data equivariance.

The model we use is a segmentation network composed of only G-CNN layers with padding such that its output is of the same spacial size as its input.

## Measuring amount of equivariance learned by a network
It is natural to measure the amount of equivariance a network $f$ has as the expected difference between the output of the transformed data and the transformed output of the original data.

$$ \mathbb{E}_{x \sim D}\left[\frac{1}{|G|}\sum_{g \in G}\|f(gx)-gf(x)\|\right]$$

We can estimate this expectation by computing the average for a series batches from our test set. However this approach has downsides, which we can tackle using the Lie derivative. 

### Lie derivative
<!--
<span style="color:red;">Add Lie group def and say that G' usually has that structure?</span>
-->


<!-- However, this approach is problematic as it only measures the amount of equivariance w.r.t. the finite group $G$. Instead, [2] proposed the use of (Lie) derivatives to evaluate the robustness of the network to infinitesimal transformations. For the notion of derivative to be defined, however, we need to assume the group to have a differential structure (Lie group). Since the space consisting of $G$ may be too peculiar to work in, we smoothly parameterize the representations of these transformations in the tangent space at the transformation that does nothing (identity element).  -->

In practice, even though we are imposing $G$-equivariance on a network, what we would like to achieve is $G'$-equivariance for a (usually infinite) group $G'$ which contains $G$. Therefore, this approach is problematic as it only measures the amount of acquired equivariance w.r.t. the finite group $G$, neglecting all other transformations, and thus doesn't give us the full picture of the network's equivariance.
 
 [2] proposed the use of Lie derivatives, which focus on the equivariance of the network thowards very small transformations in $G'$. The Lie derivative gives us a way to measure $G'$-equivariance of the network. The intuitive idea is the following: Imagine a smooth path $p(t)$ traversing the group $G'$ that starts at the identity element (transformation that does nothing) of the group, $e_{G'}$. This means that at every time-point $t \geq 0$, $p(t)$ is an element of $G'$ (i.e. some transformation), and $p(0) = e_{G'}$. Then, we can define the function:
  $$\Phi_{p(t)}f(x) := p(t)^{-1}f(p(t)x)
  $$
  This function makes some transformation $p(t)$ on the data, applies $f$ to the transformed data, and finally applies the inverse transformation $p(t)^{-1}$ to the output. Notice that if $f$ is $G'$-equivariant this value is constantly equal to $f(x)$, and that $\Phi_{p(0)}f(x) = f(x)$. The Lie derivative of $f$ along the path $p(t)$ is the derivative 
  $$L_pf(x) := d\Phi_{p(t)}f(x)/dt = \lim_{t \to 0+} \frac{\Phi_{p(t)}f(x) - f(x)}{t}
  $$
at time $t=0$. One might note that this only measures the local equivariance around the identity element of the group. Luckily, it is shown in [2] that $L_{p}f(x) = 0$ for all $x$ and $d$ specific paths, where $d$ corresponds to the dimensionality of $G'$, is equivalent to $f$ being $G'$- equivariant.

<!-- 
### Equivariance error (EE)

 Another alternative for measuring equivariance relies on the variant of approximate equivariance network we consider. Recall that what broke equivariance therein are the weights used in the linear combination of kernels that constituted the modified kernel. Therefore, a proxy for the amount of equivariance is naturally the difference between the individual kernels used over all possible transformations.
$$\frac{1}{L|G|}\sum_{l=1}^L\sum_{g \in G} |w_l(g)-w_l(e)|$$ 

-->

## Training dynamics of a network

For training dynamics, we quantify both the efficiency and efficacy of learning. 

To estimate the efficiency of learning, we analyze:
* convergence rate as measured by number of epochs trained until early stopping.
* learning curve shape


For efficacy of learning, we are interested in the final performance and the generalizability of the learned parameters, which is quantified by the flatness of the loss landscape at the final point [4], and the Hessian eigenvalue spectrum [3]. 

To measure the sharpness of the loss landscape after training, we consider changes in the loss averaged over random directions. Let $D$ denote a set of vectors randomly drawn from the unit sphere, and $T$ a set of displacements, i.e. real numbers. Then, the sharpness of the loss $\mathcal{L}$ at a point $w$ is: 

$$ \phi(w,D,T) = \frac{1}{|D||T|} \sum_{t \in T} \sum_{d \in D} |\mathcal{L}(w+dt)-\mathcal{L}(w)| 
$$

This definition is an adaptation from the one in [4] which does not normalize by $\mathcal{L}(w)$ inside the sum.

Therefore, to estimate the efficacy of learning we use the following metrics:
* final RMSE
* sharpness of the final point
* IQR and median of the max eigenvalue spectrum of the Hessian

## Reproduction results
We aim to reproduce the experiment of [1] using a 64x64 syntetic smoke dataset which has rotational symmetries. Specifically the data contains 40 simulations varied by inflow positions and buoyant forces, which exhibit perfect C4 rotational symmetry. However, buoyancy factors change with inflow locations, disrupting this symmetry. We compare our results with those of the original paper for the rsteer and rgroup models, which are the ones the paper introduces. The reconstruction RMSE for both methods is found in the table below. 

<div style="display: flex;">
  <table border="0">
    <tr>
      <th colspan="3">Results from [1]</th>
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
      <td>wip</td>
      <td>0.63</td>
    </tr>
    <tr>
      <td>Future</td>
      <td>wip</td>
      <td>0.84</td>
    </tr>
  </table>
</div>

We see that the rsteer model performs similary to what was seen in [1], the results on in domain data are slighly better while on future data the performance is a bit worse. Possibly due to a difference in the early stopping metric used. Overal we can conclude that the results from the original paper reproduce.  


Our reproduction efford includes multiple improvements to the original codebase that make it easier to reuse the code in the future. First of all the code from [1] did not include the weight constraint on the relaxed weights that is shown in the paper, we added this to the codebase. The code and paper from [1] also did not include any hyperparameters for the rgroup model. Additionally we have uploaded the smoke datasets of [1] to huggingface for ease of use and we have updated the datageneration notebook to work with the most recent version of PhiFlow [6].  

## Impact of amount of equivariance imposed on amount of equivariance learned

## Impact of amount of equivariance imposed on training dynamics




## References

[1] Discovering Symmetry Breaking in Physical Systems with Relaxed Group Convolution

[2] The Lie Derivative for Measuring Learned Equivariance

[3] How do vision transformers work?

[4] Improving Convergence and Generalization using Parameter Symmetries

[5] Approximately Equivariant Networks for Imperfectly Symmetric Dynamics

[6] PhiFlow: A differentiable PDE solving framework for deep learning via physical simulations.

Proposition 4.4, page 79