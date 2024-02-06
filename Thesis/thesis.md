---
author:
- João Rodrigo Olenscki
bibliography:
- references.bib
date: 07.02.2024
title: Improving spatio-temporal traffic prediction through transfer
  learning
---

# Abstract {#abstract .unnumbered}

In the traffic prediction field, deep learning models can be highly
accurate, but they require large amounts of data to work correctly. This
drawback, called the "cold-start" problem, can prevent cities from
starting their intelligent networks. To handle the situation, new models
based on the concept of transfer learning were proposed. These models
can learn complex [ST]{acronym-label="ST" acronym-form="singular+short"}
patterns from data-rich cities and transfer them to data-scarce
counterparts. In this work, we aim to further develop the traffic
prediction field by adapting elements of state-of-the-art models into a
single model, taking advantage of external data, and using multiple
cities as sources of knowledge.

# Acknowledgments {#acknowledgments .unnumbered}

I'd like to acknowledge List: - God - my supervisor Cheng and professor
Name (for the great counseling and discussions) (Larissa?) - my friends,
especially Lui, Luís, Tomás, João Pedro, and Ariel, for all the
attention and advice, companionship during debugging sessions, and
support - my family, my parents Ricardo and Camila, and my siblings
José, Luís, Tomás, Sofia, Beatriz, and André, for all the support over
the last years - to everyone with whom I shared this part of my life,
full of discoveries, a bit of hardship, and growth - this thesis was
"Research supported with Cloud TPUs from Google's TPU Research Cloud
(TRC)"

# Introduction {#ch:intro}

## Motivation

According to [@yin2015literature], a smart city can be defined as a
well-coordinated system that integrates advanced technological
infrastructure, relying on sophisticated data processing. The primary
objectives of such integration are to enhance city governance
efficiency, improve citizen satisfaction, foster business prosperity,
and promote environmental sustainability. Within a smart city, the
management of various individual systems that constitute the urban
environment is not solely reliant on the data collected within the city;
it also relies on different adaptable models that can learn and evolve
to suit the specific needs and characteristics of the city. In this
context, the development of traffic prediction models emerges as a
pivotal component for establishing the foundational framework of smart
city management.

Recent advances in deep learning have led to significant advancements in
prediction tasks related to traffic, such as crowd flow
[@zhang2018predicting; @jin2018spatio], traffic flow
[@polson2017deep; @wu2018hybrid], public transit flow
[@liu2019deeppf; @chai2018bike], travel demands
[@geng2019spatiotemporal], and traffic speeds [@yu2017spatio]. While
formidable in their predictive power, these models come with a
substantial data appetite. This data requirement poses a challenge for
initiating new intelligent networks because meaningful inferences remain
elusive despite the considerable investment needed to establish the
sensor network without access to substantial data history. This
difficulty is known in the field as the "cold-start" problem.

To address the aforementioned challenge, novel techniques rooted in
transfer learning [@pan2009survey] have been introduced. These
approaches enable training predictive traffic models for cities
constrained by limited data by taking advantage of patterns observed in
cities with abundant data resources. The fundamental concept behind
these models involves the application of Multi-task Learning, a type of
Inductive Transfer Learning, which entails initializing the network in
the source city and implementing fine-tuning to adapt the network to the
unique characteristics of the target city.

## Research Questions

The main objective of this work is to analyze and explore
state-of-the-art models for traffic prediction with the intent of
enhancing their accuracy. As a matter of organization, the following
secondary objectives were drawn:

-   Analyze current state-of-the-art models and identify cells and
    architectures that could be used when building a novel model;

-   Based on the results of the first objective, propose a novel model
    to be built, which should follow these requirements:

    -   capable of learning from multiple cities at the same time;

    -   capable of intra-city learning;

-   Analyze how impactful each feature derived from a requirement is to
    the model.

Concurrently, the following research questions were raised:

1.  []{#q1 label="q1"} Is it possible to encompass more than two cities
    as sources in a transfer learning process?

2.  []{#q2 label="q2"} What's the impact of the number of sources on the
    model's accuracy?

3.  []{#q3 label="q3"} Is there a limit on the number of sources?

4.  []{#q4 label="q4"} Is data augmentation possible in the traffic
    prediction field?

## Contribution

## Outline

This work is organized as follows: Chapter
[2](#ch:lit){reference-type="ref" reference="ch:lit"} introduces the
literature review that was performed in order to further understand the
problem and provides a comprehensive explanation of commonplace concepts
of the field. Chapter [3](#ch:method){reference-type="ref"
reference="ch:method"} proposes a methodological framework for the
entire work, including data acquisition, pre-processing, model building,
and testing setup. Chapter [4](#ch:results){reference-type="ref"
reference="ch:results"} analyzes the results of the proposed tests and
comparisons to proposed baselines. Chapter
[5](#ch:discussion){reference-type="ref" reference="ch:discussion"} uses
the results to answer and discuss the research questions raised on
Chapter [1](#ch:intro){reference-type="ref" reference="ch:intro"}.
Chapter [6](#ch:conclusion){reference-type="ref"
reference="ch:conclusion"} concludes the thesis and discusses the main
directions for future research in the field.

# Literature Review {#ch:lit}

This chapter presents a literature review that we conducted to further
the traffic forecasting field and its state-of-the-art.

## Traffic Forecasting {#sec:trafficforecasting}

Traffic Forecasting is a long-lasting field of study in Traffic
Engineering, conceived in the 1950s [@Beckmann1956; @Bevis1959].
Initially centered on traffic simulation, the area observed a
significant upward trend in recent years. By its very nature, a traffic
network constitutes a vast and complex system where events occurring at
various junctures within the road grid can exert profound influence over
the entire traffic flow. These inherent complexities render it an ideal
subject for examination by cutting-edge machine-learning algorithms,
such as [LSTM]{acronym-label="LSTM" acronym-form="singular+short"},
[GRU]{acronym-label="GRU" acronym-form="singular+short"}, and
[CNN]{acronym-label="CNN" acronym-form="singular+short"}.

Traffic network problems inherently comprise two domains: the temporal
and spatial domains in the early stages of deep learning model
development for the field of traffic, a natural division emerged,
allocating separate components to address each of these domains. For
example, [CNN]{acronym-label="CNN" acronym-form="singular+short"} has
conventionally been harnessed primarily for spatial feature extraction,
which involves identifying elements' physical locations or arrangements
within a given dataset. Conversely, [RNN]{acronym-label="RNN"
acronym-form="singular+short"} has specialized in temporal feature
extraction, focusing on discerning patterns that evolve or change over
time.

The pioneering work of [@Ma2017] marked one of the earliest instances of
employing [CNN]{acronym-label="CNN" acronym-form="singular+short"}s for
traffic prediction tasks. Subsequently, this methodology proliferated
across various model frameworks and underwent integration with other
architectural paradigms. An illustrative example of this evolutionary
trajectory emerges from the study conducted by [@Lin2021], which
employed a multiple [GCN]{acronym-label="GCN"
acronym-form="singular+short"} framework. In this network type, the
input to the convolutional layers consists of the city's graph
representation.

Similarly, [@Zhao2017] was among the early proponents of employing
[LSTM]{acronym-label="LSTM" acronym-form="singular+short"} cells for
short-term traffic prediction. This approach gained substantial
traction, integrating cells into various models, as exemplified by the
[ConvLSTM]{acronym-label="ConvLSTM" acronym-form="singular+short"}
module introduced by [@yao2018deep]. In this architecture, spatial
features were extracted at each time frame and subsequently fed into an
[LSTM]{acronym-label="LSTM" acronym-form="singular+short"} chain. This
innovative approach allowed for the extraction of spatial and temporal
features concurrently.

As a complex [ST]{acronym-label="ST" acronym-form="singular+short"}
problem, traffic forecasting offers a range of strategies, spanning from
problem formulation to data structuring, encompassing data type
selection. Regarding data representation, many authors
[@Wang2019b; @Yao2019; @Wang20224695] choose to apply a grid in the city
and treat each 1-by-1 region autonomously, computing variables (inflow
and outflow, for instance) inside these boundaries. In this approach,
the instantaneous snapshot of the city could be compared to an image in
the context of image classification algorithms, with each pixel being
equivalent to a region.

An alternative to this structure is transforming the raw data, typically
provided as a matrix or tensor, into a graph-based representation, with
each geographical region converted into a distinct node and an adjacency
matrix defining the connections between neighboring regions
[@Ouyang20231; @Jin2022731; @Lin2021; @Zhang2022; @Geng2019; @Wang2023; @Lu2022; @Tang2022; @Wei2016].
This method can better capture the network's complexity and regions'
nuanced connectivity. In contrast to the grid-based approach, where
regions may share a border without necessarily sharing a connecting
road, the graph representation effectively accounts for such subtleties.

A third strategy, applied in the works of [@Elmi20201088; @Li2022],
involves defining individual road segments as graph nodes. This approach
offers significantly greater detail and precision in modeling traffic
data as the data sources are condensed in a relatively small area. As a
drawback, it also requires the installation of many more sensors to
produce the data.

## Transfer Learning {#sec:transferlearning}

Transfer learning techniques [@pan2009survey] were introduced as
valuable tools for addressing problems where existing knowledge or
expertise from one domain could be employed to enhance learning and
performance in another domain. These techniques prove incredibly
beneficial when the latter domain suffers from a shortage of data. These
techniques are widely used in [NLP]{acronym-label="NLP"
acronym-form="singular+short"} problems such as sentiment and document
classification. Furthermore, the computer vision field also benefited
greatly from applying these approaches, as they are extensively used for
image classification.

As many cities started to prepare themselves to transition to become
"smart cities," they stumbled upon the "cold start" problem. This
problem refers to the lack of data that a city faces after installing
the sensor network and before acquiring enough data to justify deep
learning models, and it is not unique to smart cities but resonates with
the broader field of machine learning [@Ali2020]. In such a situation,
managers must wait at least three months to make reasonable predictions
with deep learning models despite all investments made in predicting
traffic. In these cases, based on the assumption that despite being
different, some cities can share a common framework of behavior and have
similar data distributions, transfer learning acts in favor of acquiring
knowledge from cities with abundant data and using this knowledge to
understand and predict cities with scarce data.

Furthermore, Transfer Learning techniques are also suitable for
intra-city transfer, i.e., transferring the learning from a domain (for
instance, bike sharing flow) to another in the same city (for example,
pedestrian flow). This can observed in the work of [@Wang20224695], in
which the author used a taxi trip dataset in New York to learn about
bike sharing in the same city.

Generally, a transfer learning algorithm consists of three parts:
feature extraction, in which spatio-temporal features are, through
various ways, obtained; domain adaptation, in which the knowledge is
transferred; and a predictor. The feature extraction step is also
present in deep learning approaches to the traffic forecasting problem.
It can be achieved by many different architectures, as discussed in
Section [2.1](#sec:trafficforecasting){reference-type="ref"
reference="sec:trafficforecasting"}. On the other hand, the domain
adaptation step is mainly present in the transfer learning networks and
aims to generate transferable latent features between the domains. With
it, one aims to learn domain-invariant knowledge about the problem.

### Domain Adaptation

On the efforts of domain adaptation, [@Wang20224695] proposes using
convolutional layers parallelly for both source and target features.
These convolutional layers are interconnected by calculating the
[MMD]{acronym-label="MMD" acronym-form="singular+short"} between layers
to form a transfer loss, which is then added to the overall model loss.
[MMD]{acronym-label="MMD" acronym-form="singular+short"}
[@10.5555/2188385.2188410] is a statistical metric that quantifies the
dissimilarity between two probability distributions. By minimizing its
value as a loss, the authors aimed to make the feature representation of
different domains more similar. In like manner, [@Jin2022731] also
proposes the use of [MMD]{acronym-label="MMD"
acronym-form="singular+short"} to minimize the distance between features
of different features and, in addition, the use of binary cross-entropy
on the classification of the edge types, as the authors use multi-view
graphs.

Despite working in a different field (network traffic prediction),
[@Wang202222] uses a similar architecture to [@Wang20224695], with the
substitution of the [MMD]{acronym-label="MMD"
acronym-form="singular+short"} cells for [CTD]{acronym-label="CTD"
acronym-form="singular+short"} ones. With a similar loss calculation
structure, the [CTD]{acronym-label="CTD" acronym-form="singular+short"}
cells aim to measure the domains' discrepancy. [@Ouyang20231] explores a
different approach. In their work, the authors propose using adversarial
training for transfer learning. This implies the design of a
discriminator and a predictor in the network to generate a transfer
loss.

A different, more typical approach is used in [@Tang2022], as the
authors employ the parameter-sharing technique: pre-training on the
source domain and fine-tuning on the target domain. In this technique,
the parameters of the fine-tuning stage are initialized with the trained
parameters from the pre-training phase. This kind of adaptation is only
possible when we assume that the difference between distinct domains is
not too significant and that both domains may share common features and
patterns.

In the same paper, [@Tang2022] also implements another knowledge
transfer technique, the [GRL]{acronym-label="GRL"
acronym-form="singular+short"}, originally proposed by
[@ganin2015unsupervised]. This architecture features an identity forward
pass and reverses the gradient signal during the backward pass.
Considering that different cities possess unique spatial structures and
compositions, the [GRL]{acronym-label="GRL"
acronym-form="singular+short"} is designed to mitigate these
discrepancies by fostering an adversarial training environment. This is
achieved by reversing the gradient, which effectively confuses the model
during training and is coupled with a domain classifier. The domain
classifier's task is to differentiate between source and target domains,
while the model, through the influence of the GRL, learns to generate
domain-invariant features, thus enhancing its ability to generalize
across different city datasets. This approach is particularly effective
in scenarios where the goal is to adapt a model trained on one city's
data (source domain) to perform accurately on data from another city
(target domain) despite the inherent differences in their spatial
characteristics.

# Methodology {#ch:method}

In this chapter, we build a complete picture of the Methodology applied
during the subject research. We briefly analyze the dataset used and
describe all components of the implemented model.

## Data Analysis and Exploration {#sec:data_analysis}

The dataset selected for sourcing the prediction model was part of the
NeurIPS2021 Traffic4cast competition [@pmlr-v176-eichenberger22a]. This
dataset consists of 360 days of data from 8 different cities with
similar sizes derived from trajectories of a fleet of probe vehicles.
The original data has 180 days from 2019 and another 180 days from 2020,
as one of the questions of the challenge was to asses how the COVID
pandemic affected traffic in different cities. As this represents a
shift in the temporal distribution of the data, we choose not to use the
2020 half of it, intending to have an assumed temporal invariant
distribution. Table[\[tab:cities\]](#tab:cities){reference-type="ref"
reference="tab:cities"} disposes of all cities available on the dataset
and the number of data points each city contains.

::: tabularx
M \| M \| M & &\
Antwerp & 180 & 240\
Bangkok & 180 & 240\
Barcelona & 180 & 240\
Berlin & 180 & 240\
Chicago & 180 & 240\
Istanbul & 180 & 240\
Melbourne & 180 & 240\
Moscow & 180 & 240
:::

For a given time snapshot, the data of one city can be represented by a
tensor of size $(495, 436, 8)$, where $495\times436$ represents the city
grid, and $8$ stands for the channels, or pieces of information, per
cell. These channels contain information on the volume and mean speed of
the probe cars heading in the four diagonal directions. All data was
normalized and discretized in the `uint8` range.

To better understand the data, distribution, and characteristics, the
following analysis is conducted on the 5-minute snapshot that started at
12:00 on 9 January 2019 in Melbourne. Figures
[\[fig:speed\]](#fig:speed){reference-type="ref" reference="fig:speed"}
and [3.1](#fig:volume){reference-type="ref" reference="fig:volume"} show
the distribution of values for both the speed (odd channels) and the
volume (even channels). Notice the logarithmic scale on the $y$ axis.
It's clear that while the volume has a more even distribution for
non-zero values, there remains a massive bias for the zero values, as
they represent more than 99% of the data points. This indicates that
most of our data comprises zeros and that activity should be treated as
rare.

Furthermore, Figure [3.2](#fig:heatmap){reference-type="ref"
reference="fig:heatmap"} confirms this theory, as it can be seen that
most of the data for the speed channels is zero, and the volume, despite
being better distributed, is also defined by a majority of zeros. Table
[\[tab:data_analysis\]](#tab:data_analysis){reference-type="ref"
reference="tab:data_analysis"} shows the statistics for each channel and
reinforces the proposed thesis.

<figure id="fig:volume">
<embed src="./figures/speed.pdf" style="width:90.0%" />
<embed src="./figures/volume.pdf" style="width:90.0%" />
<figcaption>Histogram for data distribution for the volume. Note the
first (very thin) bin with the null values.</figcaption>
</figure>

![Heatmap of the snapshot for the (a) Speed; and (b)
Volume.](./figures/heatmaps_speed_volume.pdf){#fig:heatmap width="90%"}

::: tabularx
M \| M \| M \| M \| M & & & &\
(volume) & 0.0216 & 0.0000 & 0.2861 & 1.08%\
(volume) & 0.0141 & 0.0000 & 0.2794 & 0.84%\
(volume) & 0.0133 & 0.0000 & 0.6236 & 0.72%\
(volume) & 0.0118 & 0.0000 & 0.5957 & 0.62%\
(speed) & 0.6758 & 0.0000 & 9.5058 & 0.73%\
(speed) & 0.8655 & 0.0000 & 11.3655 & 0.82%\
(speed) & 0.7419 & 0.0000 & 10.5863 & 0.71%\
(speed) & 0.6149 & 0.0000 & 9.8281 & 0.60%
:::

### Data processing

Some other data processing transformations were realized besides
disposing of the 2020 half of the original data. A significant part of
the motivation for this processing step is to reduce the input data's
size (or shape) to make the models trainable in a reasonable time, given
the limited computational resources available. The first one of the
transformations implemented was the collapse of the even (volume) and
odd (speed) channels by taking the mean of the data in every channel.
Therefore, we were left with two channels: the volume and average speed
of the cars in the particular region.

Furthermore, to allow efficient development of the model's components,
only the central square of size $50\times 50$ was employed for the
preliminary tests and experiments, including those used to tune the
model's parameters. This may seem limiting at first glance, as it
reduces the spatial coverage and may exclude potentially significant
peripheral data. Nonetheless, this strategy is practically beneficial as
It simplifies computational demands while still capturing a significant
portion of the traffic behaviors. The selected central area includes key
urban sections characterized by informative traffic dynamics, which
understatement will help scale the model to encompass the entire urban
layout.

## Model Outline

The proposed model consists of three modules: a Feature Extraction
Network, which is explained in Section
[3.3](#sec:fen){reference-type="ref" reference="sec:fen"}; an Embedding
Network, for which Section []{#sec:emb label="sec:emb"} is reserved; and
a Prediction Network, the final layer, which is thoughtfully dissected
in Section [3.5](#sec:pred){reference-type="ref" reference="sec:pred"}.
This Section will explain the basis for the model and the overall
architecture to be built.

### Task

The field of Spatio-Temporal prediction is vast, allowing researchers to
try different approaches to problems that may seem the same. This can be
observed from the bibliography presented in the bibliographic revision.
It's essential to clearly define the problem to be solved and rigorously
define the available information.

::: {#def:part .definition}
**Definition 3.1**. *A city $C$ is divided into a grid map of shape
$W_{C}\times H_{C}$. Each partition $r_{i, j}$, with $0\leq i \leq W_C$
and $0\leq j \leq H_C$, is referred to as a region of $C$. The set
containing all regions of the city is defined
$R_C=\{r_{0, 0}, ..., r_{W_C, H_C}\}$.*
:::

::: {#def:time .definition}
**Definition 3.2**. *The time range of available data of a city is
divided into $T_{C}$ intervals of equal size: $t=[1, ..., t_{C}]$.*
:::

::: {#def:ch .definition}
**Definition 3.3**. *For each region $r_{i, j}$, $N_{ch}$ data channels
are available. These channels are the same for every city.*
:::

By combining Definitions [3.1](#def:part){reference-type="ref"
reference="def:part"}, [3.2](#def:time){reference-type="ref"
reference="def:time"}, and [3.3](#def:ch){reference-type="ref"
reference="def:ch"} we can visualize the the 4D tensor of shape
$(W_C, H_C, N_{ch}, T_C)$. Figure
[3.3](#fig:data_tensor){reference-type="ref"
reference="fig:data_tensor"} shows a slice of this tensor. Each color
represents a different channel, and each square of four colors
represents a point in the 2D city grid. The stacked layers represent the
time dimension.

::: {#def:dim .definition}
**Definition 3.4**. *A 4D tensor defines the data of a city $C$:
$$\label{eq:defdim}
		\mathcal{X}_C = \{x_{r, t}^{ch} | r \in R_C, t \in T_c, ch \in N_{ch}\}$$*
:::

![Visualization of the data (as a
tensor).](./figures/data_tensor.pdf){#fig:data_tensor width="60%"}

With these definitions, it's possible then to define Problem
[3.1](#prob:tl){reference-type="ref" reference="prob:tl"}

::: {#prob:tl .problem}
**Problem 3.1**. *Given a data-scarce target city, $C_T$, and a set of
$n$ data-rich source cities $\{C_{S1}, ..., C_{Sn}\}$, the problem
proposed is to predict the value of the target city's data at $t_T+1$
with the historical data of the target city itself to that point and of
the source cities: $$\label{eq:probtl}
		\min_{\theta}\mathcal{L}(\tilde{\mathcal{X}}_{T, t_T + 1}, \mathcal{X}_{T, t_T + 1} )$$
where $$\label{eq:probtl2}
		\tilde{\mathcal{X}}_{T, t_T + 1} =  \theta(\mathcal{X}_{T, 1:t_T}, \{\mathcal{X}_{S1}, ..., \mathcal{X}_{Sn}\})$$*
:::

Note that $\mathcal{L}$ is the error criterion, which may vary depending
on the actual data requirements. Note also that
$t_{Sk} \gg t_T \forall k=1, ..., n$, indicating the target city's
scarcity and the sources' richness.

### Proposed Architecture

As explained at the beginning of this Section, the model comprises three
parts: a Feature Extraction Network, an Embedding Network, and a
Prediction Network. Figure
[3.4](#fig:network_simplified){reference-type="ref"
reference="fig:network_simplified"} outlines the proposed architecture.
Note that the individual modules will be developed and explained in
their sections. Furthermore, three proposed losses are to be evaluated
to train the model. The architecture adopted in this research draws upon
the works of [@Wang202222; @Wang20224695] as they proposed similar
divisions in their models to transfer knowledge. Compared to their
approach, we suggest using a [STGAE]{acronym-label="STGAE"
acronym-form="singular+short"} as a mechanism to train the feature
extractor and an adjacency matrix to represent connectivity between the
regions.

![Simplified version of the proposed
model.](./figures/network_simplified.pdf){#fig:network_simplified
width="90%"}

## Feature Extraction Network {#sec:fen}

As the first layer of the model, the Feature Extraction Network receives
an input of tensors from one or more source cities and the target city
and tries to extract, from these tensors, [ST]{acronym-label="ST"
acronym-form="singular+short"} features must be extracted to be used to
train the Prediction Network.

### Autoencoder

Selecting hyperparameters and fine-tuning an extractor are challenging
tasks in constructing a model, as it's very difficult to observe
causality between the change of a parameter and the change of the output
due to the highly non-linear characteristics of these modules. As a
result of these problems, the use of autoencoders for the feature
extraction task has been proposed by [@Hinton2006], and it's, as of
today, a well-established paradigm in the [ST]{acronym-label="ST"
acronym-form="singular+short"} field. By conceptualizing the feature
extraction process through the lens of an encoder, it becomes easier to
verify its quality by constructing a corresponding decoder. As the
encoder maps the input data from its original vectorial space to a
latent space, the decoder pursues the contrary operation, returning the
data from the latent space to the original one. In an ideal scenario, a
well-trained autoencoder will reconstruct the original data perfectly,
guaranteeing the quality of the features extracted by the encoder.

More recently, [@fan2023spatiotemporal] implemented a
[STAE]{acronym-label="STAE" acronym-form="singular+short"} by coupling
[GLU]{acronym-label="GLU" acronym-form="singular+short"} layers for time
convolution and Chebyshev convolution layers for spatial convolution. By
interpolating two temporal layers by a spatial one, the authors
extracted both spatial and temporal features. Additionally, using
Chebyshev filters of relatively large sizes ($K=6$), the proposed
autoencoder could properly derive features on both local and global
scales.

In another paper, [@sabbaqi2022graph] proposes a generic framework for
[STGAE]{acronym-label="STGAE" acronym-form="singular+short"} with
symmetric encoder-decoder architectures. The encoder finds a latent
graph representation by applying graph convolutions, temporal
downsampling layers, and activation functions. The decoder mirrors this
behavior but uses temporal upsampling layers between the convolutions
and activation functions.

Figure [3.5](#fig:autoencoder){reference-type="ref"
reference="fig:autoencoder"} illustrates the architecture of the
[STGAE]{acronym-label="STGAE" acronym-form="singular+short"} utilized in
our feature extraction framework. The encoder predominantly comprises a
[GConvLSTM]{acronym-label="GConvLSTM" acronym-form="singular+short"}
block responsible for the intricate task of extracting spatio-temporal
features. It is followed sequentially by an activation function, batch
normalization, a regularization dropout layer, and a linear
transformation layer. Very similarly, the decoder incorporates analogous
components, adding a Sigmoid activation function preceding the output.
This configuration leverages the data normalization previously applied,
wherein the value range of all channels was linearly transformed from
$[0, 255]$ to a unit interval $[0, 1]$.

![Diagram representing the autoencoder as a combination of an encoder
and a decoder.](./figures/Autoencoder.pdf){#fig:autoencoder width="90%"}

The [GConvLSTM]{acronym-label="GConvLSTM" acronym-form="singular+short"}
cell, as implemented by [@rozemberczki2021pytorch], and originally
proposed by [@Seo_2018], is parameterized by the number of input
channels $N_{\text{in}}$, the number of output channels
$N_{\text{out}}$, and the size of the Chebyshev polynomial filter $K$.
The cell executes graph-based convolutions on the input tensor $x$, with
the knowledge contained on the graph edge's descriptor tensor
`edge_index`, to yield the hidden state $h$ and the cell state $c$.
These states are then propagated through the sequence for subsequent
iterations, which are defined by the $k$ discrete temporal segments of
which the input $x$ is composed. This enables the model to capture and
encode the temporal dynamics of the data.

$$\text{GConvLSTM}: \mathbb{R}^{D_1 \times D_2 \times \ldots \times D_N \times N_{\text{in}}} \rightarrow \mathbb{R}^{D_1 \times D_2 \times \ldots \times D_N \times N_{\text{out}}}$$

Furthermore, as implemented in this layer, the order of the Chebyshev
filter plays a pivotal role, as it defines the range of neighborhood
aggregation. Specifically, it dictates how and how many local
neighborhoods are expanded around each node during the convolutional
process. This, in turn, influences the gradient computation during
backpropagation, affecting both the receptive field and the capacity of
the model to capture and integrate multi-hop relational information.

In a regular [LSTM]{acronym-label="LSTM" acronym-form="singular+short"}
implementation, all internal variables are calculated based on the
sigmoid of combinations of fully connected layers as highlighted in the
equations:

$$\begin{aligned}
i & =\sigma\left(\boxed{W_{x i} x_t} + \boxed{W_{h i} h_{t-1}} + w_{c i} \odot c_{t-1}+b_i\right), \\
f & =\sigma\left(\boxed{W_{x f} x_t} + \boxed{W_{h f} h_{t-1}} + w_{c f} \odot c_{t-1}+b_f\right), \\
c_t & =f_t \odot c_{t-1}+i_t \odot \tanh \left(\boxed{W_{x c} x_t}+\boxed{W_{h c} h_{t-1}}+b_c\right), \\
o & =\sigma\left(\boxed{W_{x o} x_t} + \boxed{W_{h o} h_{t-1}} + w_{c o} \odot c_t+b_o\right), \\
h_t & =o \odot \tanh \left(c_t\right),
\end{aligned}$$

Generalizing the [LSTM]{acronym-label="LSTM"
acronym-form="singular+short"}, a model developed for time-series
forecasting, for graph inputs requires the adjustment of these
operations for something that can handle graph-data input. For the
[GConvLSTM]{acronym-label="GConvLSTM" acronym-form="singular+short"},
the authors implemented the graph convolution operator
$\ast_\mathcal{G}$ proposed by [@cnn_graph], in which a graph signal
$x \in \mathbb{R}^{n}$ with $n$ nodes is filtered by a non-parametric
kernel $g_\theta$ composed of vectors of Fourier coefficients.

$$y = g_\theta \ast_\mathcal{G} x$$

On this implementation, $\ast_\mathcal{G}$ is modeled with the
normalized graph Laplacian decomposition $L=U \Lambda U^T$, which would
imply a model's complexity of $\mathcal{O}(n^2)$. To make it more
feasible, the authors propose a truncated expansion of $g_\theta$ using
Chebyshev polynomials $T_k$ and truncated laplacian $\tilde{L}$. This
reduces the complexity to $\mathcal{O}(|\epsilon|n)$, with $\epsilon$
number of edges of the graph.

$$y = g_\theta \ast_\mathcal{G} x = \sum_{k=0}^{K-1}\theta_k T_k(\tilde{L})x$$

Finally, it's possible to define the
[GConvLSTM]{acronym-label="GConvLSTM" acronym-form="singular+short"}
model.

$$\begin{aligned}
i & =\sigma\left(W_{x i}\ast_\mathcal{G} x_t+W_{h i}\ast_\mathcal{G} h_{t-1}+w_{c i} \odot c_{t-1}+b_i\right), \\
f & =\sigma\left(W_{x f}\ast_\mathcal{G} x_t+W_{h f}\ast_\mathcal{G} h_{t-1}+w_{c f} \odot c_{t-1}+b_f\right), \\
c_t & =f_t \odot c_{t-1}+i_t \odot \tanh \left(W_{x c}\ast_\mathcal{G} x_t+W_{h c}\ast_\mathcal{G} h_{t-1}+b_c\right), \\
o & =\sigma\left(W_{x o}\ast_\mathcal{G} x_t+W_{h o} \ast_\mathcal{G}h_{t-1}+w_{c o} \odot c_t+b_o\right), \\
h_t & =o \odot \tanh \left(c_t\right),
\end{aligned}$$

### Fine Tuning

The autoencoder, the most computationally demanding part of the entire
model, owes much of its complexity to the
[GConvLSTM]{acronym-label="GConvLSTM" acronym-form="singular+short"}
layers in both the encoder and decoder. These layers execute graph
convolution operations, which can become computationally intensive for
large graphs due to their quadratic complexity, $\mathcal{O}(n^2)$,
where $n$ is the number of nodes in the graph.

We suggest a pragmatic two-step training approach for the feature
extractor to handle the computational demand more effectively.
Initially, we train the autoencoder using the available source data,
which allows us to establish a solid initial data representation.
Subsequently, we fine-tune this representation with the target data,
adapting it to the specific characteristics of the target domain.

Moreover, this method results in a fixed feature extractor that is not
tightly coupled to the overall model. This flexibility is particularly
beneficial when considering the parameter definition of the latter parts
of the model, meaning that we can then focus on training both the
Embedding Network and the Prediction Network without having to mind the
effects that the choice of the feature extractor's parameter has on the
results.

## Domain Adaptation

As one of the main challenges of a transfer learning task, domain
adaptation is the process of adapting a model trained on one or more
source domains (where abundant data is available) to perform well on a
different but related target domain (where data is limited or has
different distribution characteristics). There are different approaches
that can be taken to tackle this problem in the different contexts that
appear during the development of the model.

We suggest using two domain adaptors for this model: parameter sharing
and a [GRL]{acronym-label="GRL" acronym-form="singular+short"}. The
parameter sharing, which would be applied during a fine-tuning process
after a pre-training, makes sure to start the network, which really
matters, the one that predicts the target values with the right weights
to make it possible for this network to properly work despite the lack
of data points.

The [GRL]{acronym-label="GRL" acronym-form="singular+short"}, on the
other hand, aims to help the feature extractor to produce
domain-invariant features, which will then be extremely useful for
transferring knowledge between different domains by making the encoder
less sensitive to the specific characteristics of the source domain,
thereby enhancing its ability to perform well on the target one.

Additionally, the [GRL]{acronym-label="GRL"
acronym-form="singular+short"} is also used as part of an adversarial
training strategy, which is performed by a domain discriminator or
classifier that tries to distinguish source and target domains. Since
the [GRL]{acronym-label="GRL" acronym-form="singular+short"} reverses
the gradient sign and scales it during backpropagation, it encourages
the model to generate features that try to "fool" the domain classifier,
which raises the loss but enforces the encoder to generate features that
are domain-agnostic.

For the Domain Discriminator, we model it as a sequence of three
operations: firstly, we apply a global mean pool to the features'
tensor, followed by a linear layer, and then the softmax function. This
will yield the probabilities that the features come from each domain (in
this case, we have only two: source and target). With these
probabilities, it's then possible to calculate a loss measure using
[BCE]{acronym-label="BCE" acronym-form="singular+short"} loss. Figure
[3.6](#fig:DomainDiscriminator){reference-type="ref"
reference="fig:DomainDiscriminator"} shows how this would look like and
at which points of the overall model both the [GRL]{acronym-label="GRL"
acronym-form="singular+short"} and Domain Discriminator would be placed.

We use the global mean pool operation on the features' tensor to reduce
the node-level features to a representation that captures the overall
(global) characteristics of the graph without focusing on the individual
ones (local)

![Diagram representing the architecture for the Domain Discriminator
module.](./figures/DomainDiscriminator.pdf){#fig:DomainDiscriminator
width="90%"}

## Prediction Network {#sec:pred}

## Model Training {#sec:training}

Figure [3.7](#fig:training_script){reference-type="ref"
reference="fig:training_script"} shows the

![Diagram representing the training script and domain adaptation
process.](./figures/training_script.pdf){#fig:training_script
width="90%"}

## Loss Functions {#sec:loss_func}

In this Section, we define and explain the loss functions, also known as
cost functions or criteria, that will be implemented or tested on the
model or its parts. From the following criteria,
[MAE]{acronym-label="MAE" acronym-form="singular+short"} and
[MAPE]{acronym-label="MAPE" acronym-form="singular+short"} are to be
used exclusively as performance metrics. The purpose of listing and
implementing a varied array of loss functions is to determine the best
fit for our problem and data.

In this section, we define and elucidate the various loss functions,
also recognized as cost functions or evaluation criteria, which are to
be implemented or assessed within the model or its components. Among the
specified criteria,[MAE]{acronym-label="MAE"
acronym-form="singular+short"} and [MAPE]{acronym-label="MAPE"
acronym-form="singular+short"} will be exclusively utilized as
performance metrics. Listing and implementing various loss functions is
to determine the ones with the best training performance considering the
problem and dataset.

### Mean Squared Error (MSE)

As the most known and traditional loss function,
[MSE]{acronym-label="MSE" acronym-form="singular+short"} is widely used
in almost all fields of machine learning. In particular, it's one of
traffic forecasting models' most "standard" loss functions as it focuses
on minimizing large errors.

$$\mathcal{L}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

### Mean Absolute Error (MAE)

The [MAE]{acronym-label="MAE" acronym-form="singular+short"} is not
supposed to be used as a criterion during the training phase, but
rather, it's widely used as a performance metric for testing. It
calculates the average absolute difference between the predicted and
actual values. It treats all errors equally and is particularly robust
when dealing with outliers.

$$\mathcal{L}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

### Mean Absolute Percentage Error (MAPE)

Similarly, [MAPE]{acronym-label="MAPE" acronym-form="singular+short"} is
another performance metric to be applied during the testing phase. It
uses the same logic as the [MAE]{acronym-label="MAE"
acronym-form="singular+short"} but calculates the percentage error of
the absolute difference between the predicted and actual values. It's
used in the same context as MAE, but it's easier to interpret and can be
very useful when dealing with data represented in different scales.

$$\mathcal{L}(y, \hat{y}) = \frac{100\%}{N} \sum_{i=1}^{N} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

### Weighted Mean Squared Error (WMSE)

As a variant of the MSE, the [WMSE]{acronym-label="WMSE"
acronym-form="singular+short"} amplifies the significance of errors in
certain parts of the dataset by multiplying each error by a specific
weight. It's useful in scenarios where certain data points are more
critical than others and should have more influence on the total loss.
Particularly, it's applied to heavenly imbalanced datasets, as is the
case for the data used in this work. This criterion is also known as
Zero Inflation Loss when defining a particular weight for zero values.

$$\mathcal{L}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} w \cdot (y_i - \hat{y}_i)^2$$

### Mean Squared Logarithm Error (MSLE)

This other variation of the MSE criterion consists of the mean of the
squares of the logarithmic differences between the predicted and actual
values. This criterion reduces the impact of large errors on large true
values. Applied on problems where the target values have a wide range.

$$\mathcal{L}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (\log(y_i + 1) - \log(\hat{y}_i + 1))^2$$

### Weighted Mean Squared Logarithm Error (WMSLE)

The Weighted Mean Squared Logarithm Error ([WMSLE]{acronym-label="WMSLE"
acronym-form="singular+short"}) is a sophisticated criterion that
combines the aspects of the Mean Squared Logarithm Error
([MSLE]{acronym-label="MSLE" acronym-form="singular+short"}) and the
Weighted Mean Squared Error ([WMSE]{acronym-label="WMSE"
acronym-form="singular+short"}). It specifically addresses the challenge
of imbalanced datasets, focusing on penalizing the errors associated
with non-zero targets more heavily.

This loss function is particularly adept at handling datasets where the
prediction of non-zero values is more crucial than the prediction of
zeros, which may be abundant but less informative. The
[WMSLE]{acronym-label="WMSLE" acronym-form="singular+short"} is
therefore especially useful when the cost of an error varies depending
on the magnitude of the true value, such as in datasets with many zero
entries but where accurate prediction of the non-zero values is
paramount, which happens to be the case in our dataset (as discussed in
Section [3.1](#sec:data_analysis){reference-type="ref"
reference="sec:data_analysis"}). The [WMSLE]{acronym-label="WMSLE"
acronym-form="singular+short"} is defined as follows:

$$\mathcal{L}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} w \cdot (\log(y_i + 1) - \log(\hat{y}_i + 1))^2$$

### Custom Huber Loss

The Custom Huber Loss combines the [MSE]{acronym-label="MSE"
acronym-form="singular+short"} and [MAE]{acronym-label="MAE"
acronym-form="singular+short"}. It's quadratic for small errors and
linear for large errors. We can also apply a Zero Inflation Loss for
zero target values. It's very useful when dealing with outliers and
versatile as it can have the best of both criteria on the domains they
are the best.

$$\mathcal{L}_{\delta, w}(y, \hat{y}) = 
\begin{cases} 
\frac{1}{2} w \cdot (y_i - \hat{y}_i)^2 & \text{for } y =0, \\
\frac{1}{2} (y - \hat{y})^2 & \text{for } |y - \hat{y}| \leq \delta, \\
\delta \cdot (|y - \hat{y}| - \frac{1}{2} \delta) & \text{otherwise.}
\end{cases}$$

### Log-Cosh Loss

The Log-Cosh error is defined by the logarithm of the hyperbolic cosine
of the difference between the prediction and actual values. It has a
shape similar to the [MSE]{acronym-label="MSE"
acronym-form="singular+short"} but is smoother and less sensitive to
outliers. It's useful as it has the robustness of the
[MAE]{acronym-label="MAE" acronym-form="singular+short"} while
maintaining the smooth gradient of the [MSE]{acronym-label="MSE"
acronym-form="singular+short"}. Note that this loss function is
differentiable, unlike [MAE]{acronym-label="MAE"
acronym-form="singular+short"}.

$$\mathcal{L}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} \log(\cosh(\hat{y}_i - y_i))$$

### Binary Cross Entropy

In the binary classification context, the [BCE]{acronym-label="BCE"
acronym-form="singular+short"} loss penalizes deviations from the actual
labels by comparing the predicted probabilities against the ground
truth, providing a robust gradient signal for model updates during
training.

$$\mathcal{L}(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^{N} [y_{i} \log(\sigma(\hat{y}_{i})) + (1 - y_{i}) \log(1 - \sigma(\hat{y}_{i}))]$$

# Results {#ch:results}

## Computational Specifications and Training Times

## Autoencoder Experiments

This Section will present the experiments on the variation of the
autoencoder's parameters. Experiments
[4.2.1](#ssec:exp1){reference-type="ref" reference="ssec:exp1"} to
[4.2.5](#ssec:exp5){reference-type="ref" reference="ssec:exp5"} are
related to the hyperparameter search for the autoencoder, while
[4.2.7](#ssec:exp6){reference-type="ref" reference="ssec:exp6"} presents
the impacts that parameter sharing as a domain adaptation technique.

### Experiment 1: Impact of the Chebyshev polynomial degree parameter on the autoencoder's performance {#ssec:exp1}

::: tabularx
M \| M &\
Batch size & 32\
Number of cities & 2\
Epochs & 2\
Convolution dimension & 16\
Linear dimension & 8\
Activation function & ReLU\
Dropout rate & 0.5\
Loss criterion & ZeroInflationLoss($w=100$)
:::

The Chebyshev polynomial degree is one of the parameters of the
`GConvLSTM` cell used as the backbone of the autoencoder. It dictates
the receptive field of the model, as it limits the number of neighbors
that will be used to structure a computational graph during
backpropagation. For instance, for a value of $K_{\text{cheb}}=1$, the
computational graph will have a depth of 1, and only the nodes
individually will be part of it. A $K_{\text{cheb}}=2$ implies that the
computational graphs will contain not only the nodes themselves but the
immediate neighbors of each node.

In this experiment, we evaluated the autoencoder's performance with
varying values of $K_{\text{cheb}}$, ranging from 1 to 6, while
maintaining all other parameters as specified in Table
[\[tab:exp01\]](#tab:exp01){reference-type="ref" reference="tab:exp01"}.
Figure [4.1](#fig:exp01boxplot){reference-type="ref"
reference="fig:exp01boxplot"} illustrates the distributions of both
[MAE]{acronym-label="MAE" acronym-form="singular+short"} and
[MSE]{acronym-label="MSE" acronym-form="singular+short"} for the
corresponding $K_{\text{cheb}}$ values. The boxplots reveal a
discernible trend where both [MAE]{acronym-label="MAE"
acronym-form="singular+short"} and [MSE]{acronym-label="MSE"
acronym-form="singular+short"} metrics decrease as $K_{\text{cheb}}$
increases from 1 up to 3, indicative of improved performance. The median
of [MAE]{acronym-label="MAE" acronym-form="singular+short"} reaches its
minimum at $K_{\text{cheb}} = 3$, suggesting this is the optimal
Chebyshev polynomial degree for capturing the [ST]{acronym-label="ST"
acronym-form="singular+short"} features within the dataset, given the
current parameter configuration. However, for $K_{\text{cheb}}$ values
greater than 3, there is an observable increase in variance and a slight
rise in error metrics, which may signal a risk of overfitting.

![[MAE]{acronym-label="MAE" acronym-form="singular+short"} and
[MSE]{acronym-label="MSE" acronym-form="singular+short"} for the
autoencoder with different values of
$K_{\text{cheb}}$](./figures/exp01/boxplot.pdf){#fig:exp01boxplot
width="90%"}

### Experiment 2: Impact of the number of source cities on the autoencoder's performance {#ssec:exp2}

::: tabularx
M \| M &\
Batch size & 8\
Epochs & 2\
Chebyshev polynomial degree & 2\
Convolution dimension & 16\
Linear dimension & 8\
Activation function & ReLU\
Dropout rate & 0.5\
Loss criterion & ZeroInflationLoss($w=100$)
:::

Experiment 2 deals with the impact of the number of cities on the
autoencoder's performance. The number of cities indicates the amount of
data available to the model and the diversity and complexity presented.
Four different values for the number of cities were considered: 1, 2, 4,
and 8. The fixed parameters of the experiment are presented in Table
[\[tab:exp02\]](#tab:exp02){reference-type="ref" reference="tab:exp02"},
and Figure [4.2](#fig:exp02boxplot){reference-type="ref"
reference="fig:exp02boxplot"} shows the results obtained.

For training over just one city $(N_{cities}=1)$, we observe a
relatively high median value for both [MAE]{acronym-label="MAE"
acronym-form="singular+short"} and [MSE]{acronym-label="MSE"
acronym-form="singular+short"}, with a wide [IQR]{acronym-label="IQR"
acronym-form="singular+short"}, suggesting that the model lacks
generalization capabilities compared to the other models. Introducing
another city to the model $(N_{cities}=2)$ seems to result in a slighter
better median for the [MSE]{acronym-label="MSE"
acronym-form="singular+short"}, but at the cost of outliers and a
significantly higher median value for the [MAE]{acronym-label="MAE"
acronym-form="singular+short"}.

When training over four $(N_{cities}=4)$ and eight $(N_{cities}=8)$
cities, the model seem to behave similarly. It's noticeably better than
the previous models when comparing the median values for both metrics,
indicating better performance and generalization capabilities. There are
outliers at $(N_{cities}=8)$, which may indicate a limit on how much
diversity can be inputted into the model while trying to increase
generalization capabilities.

The overall trend suggests that incorporating data from a larger number
of cities enables the model to learn more generalizable features across
different domains, reducing the model's bias toward specific cities'
traffic patterns. It's worth noting that increasing the number of cities
also increases the computational cost of training the model. Therefore,
it seems that using the values of $N_{cities}$ near 4 is more optimal.

![[MAE]{acronym-label="MAE" acronym-form="singular+short"} and
[MSE]{acronym-label="MSE" acronym-form="singular+short"} for the
autoencoder with different number of
cities](./figures/exp02/boxplot.pdf){#fig:exp02boxplot width="90%"}

### Experiment 3: Impact of the activation function on the autoencoder's performance {#ssec:exp3}

::: tabularx
M \| M &\
Batch size & 32\
Number of cities & 2\
Epochs & 2\
Chebyshev polynomial degree & 3\
Convolution dimension & 16\
Linear dimension & 8\
Dropout rate & 0.5\
Loss criterion & ZeroInflationLoss($w=100$)
:::

The choice of activation function within neural network layers
substantially impacts the model's ability to capture and represent
complex patterns within the data. This experiment examines how the
activation function selection influences the autoencoder's performance.
Three common activation functions were considered: ReLU, Sigmoid, and
Tanh.

For this analysis, we trained separate autoencoder models using each
activation function while keeping all other parameters constant, as
specified in Table [\[tab:exp03\]](#tab:exp03){reference-type="ref"
reference="tab:exp03"}. Figure
[4.3](#fig:exp03boxplot){reference-type="ref"
reference="fig:exp03boxplot"} presents the distributions of
[MAE]{acronym-label="MAE" acronym-form="singular+short"} and
[MSE]{acronym-label="MSE" acronym-form="singular+short"} across the
different activation functions.

The boxplots indicate that the model with the ReLU activation function
exhibits a slightly higher median [MAE]{acronym-label="MAE"
acronym-form="singular+short"} than the other functions but also has a
wider [IQR]{acronym-label="IQR" acronym-form="singular+short"},
suggesting less consistent predictions across different instances. On
the other hand, the model utilizing the Sigmoid function shows a tighter
[IQR]{acronym-label="IQR" acronym-form="singular+short"} in
[MAE]{acronym-label="MAE" acronym-form="singular+short"}, indicating
less variability in its predictions, although its median
[MAE]{acronym-label="MAE" acronym-form="singular+short"} is higher.
Interestingly, the Tanh function yields the lowest median
[MAE]{acronym-label="MAE" acronym-form="singular+short"}, suggesting it
may be the most effective at capturing the underlying
[ST]{acronym-label="ST" acronym-form="singular+short"} patterns for this
particular dataset and model configuration.

![[MAE]{acronym-label="MAE" acronym-form="singular+short"} and
[MSE]{acronym-label="MSE" acronym-form="singular+short"} for the
autoencoder with different activation
functions.](./figures/exp03/boxplot.pdf){#fig:exp03boxplot width="90%"}

### Experiment 4: Impact of the criterion function on the autoencoder's performance {#ssec:exp4}

::: tabularx
M \| M &\
Batch size & 32\
Number of cities & 2\
Epochs & 2\
Chebyshev polynomial degree & 4\
Convolution dimension & 16\
Linear dimension & 8\
Activation function & ReLU\
Dropout rate & 0.5
:::

Another interesting choice that can be made in the autoencoder training
is the criterion function that yields the backpropagated loss. In this
experiment, we analyze how different loss functions can influence the
autoencoder's performance. Note that all criteria used here were
presented and explained in Section
[3.7](#sec:loss_func){reference-type="ref" reference="sec:loss_func"}.
All other parameters used are exposed in Table
[\[tab:exp04\]](#tab:exp04){reference-type="ref" reference="tab:exp04"}.
The losses considered were:

-   [MSE]{acronym-label="MSE" acronym-form="singular+short"}

-   [WMSE]{acronym-label="WMSE" acronym-form="singular+short"} $(w=10)$

-   [WMSE]{acronym-label="WMSE" acronym-form="singular+short"} $(w=100)$

-   [MSLE]{acronym-label="MSLE" acronym-form="singular+short"}

-   [WMSLE]{acronym-label="WMSLE" acronym-form="singular+short"}
    $(w=10)$

-   [WMSLE]{acronym-label="WMSLE" acronym-form="singular+short"}
    $(w=100)$

-   Log-Cosh Loss

-   Focal Loss $(\alpha=0.25, \gamma=2)$

Figure [4.4](#fig:exp04boxplot){reference-type="ref"
reference="fig:exp04boxplot"} provides a comparative overview of the
performance of these criteria on the autoencoder. From the plots, it is
evident that both Focal Loss (with the input parameters) and Log-Cosh
Loss are worse than their pairs, showing bigger overall error
([MAE]{acronym-label="MAE" acronym-form="singular+short"} and
[MSE]{acronym-label="MSE" acronym-form="singular+short"}) and
[IQR]{acronym-label="IQR" acronym-form="singular+short"}, indicating
that they are not fit for being used as criteria.

The traditional [MSE]{acronym-label="MSE" acronym-form="singular+short"}
provides a baseline for comparison, exhibiting a moderate spread in
[MAE]{acronym-label="MAE" acronym-form="singular+short"} values. The
[MSLE]{acronym-label="MSLE" acronym-form="singular+short"} is designed
to be less sensitive to large errors by emphasizing the logarithmic
difference between the predicted and actual values, which is evident in
the lower median [MAE]{acronym-label="MAE"
acronym-form="singular+short"} it achieves compared to
[MSE]{acronym-label="MSE" acronym-form="singular+short"}.

A more nuanced approach is observed with both
[WMSE]{acronym-label="WMSE" acronym-form="singular+short"} and
[WMSLE]{acronym-label="WMSLE" acronym-form="singular+short"}, where
introducing a weight factor ($w$) aims to penalize errors differently
based on their magnitude. Particularly, for the
[WMSE]{acronym-label="WMSE" acronym-form="singular+short"}, as the
weight increases from $w=1$ ([MSE]{acronym-label="MSE"
acronym-form="singular+short"}) to $w=10$ to $w=100$, the spread and
median of the [MAE]{acronym-label="MAE" acronym-form="singular+short"}
decrease, suggesting a tighter grouping of errors around a lower central
value. However, an increase in weight also introduces a higher variance
in [MSE]{acronym-label="MSE" acronym-form="singular+short"}, as
indicated by the presence of outliers, particularly for $w=100$. This
implies that while [WMSE]{acronym-label="WMSE"
acronym-form="singular+short"} can potentially reduce the average error,
it may also lead to more extreme errors in some instances. For the
[WMSLE]{acronym-label="WMSLE" acronym-form="singular+short"} criterion,
a similar result is observed, with the increase of $w$ being associated
with a smaller [IQR]{acronym-label="IQR" acronym-form="singular+short"}
but a higher median of the [MAE]{acronym-label="MAE"
acronym-form="singular+short"}.

![[MAE]{acronym-label="MAE" acronym-form="singular+short"} and
[MSE]{acronym-label="MSE" acronym-form="singular+short"} for the
autoencoder with different criterion
functions.](./figures/exp04/boxplot.pdf){#fig:exp04boxplot width="90%"}

### Experiment 5: Impact of the latent dimension's size the autoencoder's performance {#ssec:exp5}

::: tabularx
M \| M &\
Batch size & 32\
Number of cities & 2\
Epochs & 2\
Chebyshev polynomial degree & 4\
Activation function & Tanh\
Dropout rate & 0.5\
Loss criterion & ZeroInflationLoss($w=100$)
:::

Experiment 5 focuses on the impact of inner (or latent) layer dimensions
on the autoencoder's performance. For this purpose, several pairs of
convolutional and linear dimensions were tested and are exposed in Table
[\[tab:exp05b\]](#tab:exp05b){reference-type="ref"
reference="tab:exp05b"}. Furthermore, Table
[\[tab:exp05\]](#tab:exp05){reference-type="ref" reference="tab:exp05"}
shows the fixed parameters for the experiment.

::: tabularx
M \| M &\
& 2\
& 4\
& 8\
& 4\
& 8\
& 16\
& 8\
& 16
:::

Analyzing Figure [4.5](#fig:exp05boxplot){reference-type="ref"
reference="fig:exp05boxplot"}, it's possible to note that the
convolutional dimension is the most important parameter between the two.
Models with smaller convolutional dimension ($\texttt{conv\_dim}=8$)
tend to have higher [MAE]{acronym-label="MAE"
acronym-form="singular+short"} and [MSE]{acronym-label="MSE"
acronym-form="singular+short"} values, indicating a lower performance.
This suggests that these models may not have enough complexity to
capture the data's relevant [ST]{acronym-label="ST"
acronym-form="singular+short"} features.

Going up to $\texttt{conv\_dim}=16$, we can observe a great reduction on
both metrics. This points to a better extraction feature, enabling the
autoencoder to capture more of the data's complexity. Finally, with
$\texttt{conv\_dim}=32$, we note an increase of the median and extremes
of the error, but a smaller [IQR]{acronym-label="IQR"
acronym-form="singular+short"}, suggesting that we've reached a point
where overfitting starts to take effect on the model.

Considering now the values of the linear dimensions, there's no clear
trend that we can extract from the results, as an increase in it for low
values of convolutional dimension led to bigger errors, but for other
values, it made no difference. It's important to note that an increase
in both the convolutional and linear dimensions implies higher model
complexity and, thus, higher computational cost for training it. For
this reason, settling for pairs like $(16, 4)$ or $(16, 8)$ is the best
option for balancing performance with cost.

![[MAE]{acronym-label="MAE" acronym-form="singular+short"} and
[MSE]{acronym-label="MSE" acronym-form="singular+short"} for the
autoencoder with different pairs of latent dimension
values.](./figures/exp05/boxplot.pdf){#fig:exp05boxplot width="90%"}

### Experiment 6: Pretraining as a domain adaptation technique for the autoencoder {#ssec:exp6}

::: tabularx
M \| M &\
Batch size & 16\
Epochs & 4\
Chebyshev polynomial degree & 4\
Convolution dimension & 16\
Linear dimension & 8\
Activation function & TanH\
Dropout rate & 0.5\
Loss criterion & ZeroInflationLoss($w=100$)
:::

Experiment 6 is the first try of applying a fine-tuning technique to
transfer knowledge from one model to another. It consisted of training
the model across three different setups: using solely the target city
data ("Target Only"), pretraining with data from one source city
followed by finetuning on the target city data ("Pretrained w/ 1
Source"), and pretraining with data from two source cities before
finetuning on the target city data ("Pretrained w/ 2 Sources"). The
results, illustrated in Figure
[4.6](#fig:exp06boxplot){reference-type="ref"
reference="fig:exp06boxplot"}, underscore the benefits of transfer
learning through pretraining.

The "Target Only" model shows the higher median and
[IQR]{acronym-label="IQR" acronym-form="singular+short"} for both
[MAE]{acronym-label="MAE" acronym-form="singular+short"} and
[MSE]{acronym-label="MSE" acronym-form="singular+short"} when compared
to the pre-trained ones. Conversely, both pre-trained setups greatly
reduce both error metrics' median and [IQR]{acronym-label="IQR"
acronym-form="singular+short"}. This suggests that the pre-training is
an effective method for domain adaptation, as the model gets
significantly better at generalizing and accurately predicting the
[ST]{acronym-label="ST" acronym-form="singular+short"} patterns.
Furthermore, the image also indicates that having more cities can
enhance this technique, as the diversity of data presented in the model
increases its generalization capacity, making it more accurate.

![[MAE]{acronym-label="MAE" acronym-form="singular+short"} and
[MSE]{acronym-label="MSE" acronym-form="singular+short"} for the
autoencoders tested.](./figures/exp06/boxplot.pdf){#fig:exp06boxplot
width="90%"}

### Experiment 7: Attaching the Domain Discriminator module to the autoencoder {#ssec:exp6}

::: tabularx
M \| M &\
Batch size & 16\
Epochs & 2\
Chebyshev polynomial degree & 3\
Convolution dimension & 16\
Linear dimension & 8\
Activation function & TanH\
Dropout rate & 0.5\
Loss criterion & [MSE]{acronym-label="MSE"
acronym-form="singular+short"}
:::

For Experiment 7, we attached a domain discriminator and a
[GRL]{acronym-label="GRL" acronym-form="singular+short"} to the
autoencoder.

## Predictor Experiments

Given then an autoencoder capable of effectively extracting the relevant
[ST]{acronym-label="ST" acronym-form="singular+short"} features, it's
necessary now to define good hyperparameters for the predictor part of
the model. Experiments \.... investigate the optimization of the model
by parameter variation, while Experiments \... deal with the knowledge
transfer part.

## Full Model Experiments

This Section delves into the experimentation of the full model, as
presented in Figure [3.7](#fig:training_script){reference-type="ref"
reference="fig:training_script"}

This Section will present the experiments on the variation of the
autoencoder's parameters. Experiments
[4.2.1](#ssec:exp1){reference-type="ref" reference="ssec:exp1"} to
[4.2.5](#ssec:exp5){reference-type="ref" reference="ssec:exp5"} are
related to the hyperparameter search for the autoencoder, while
[4.2.7](#ssec:exp6){reference-type="ref" reference="ssec:exp6"} presents
the impacts that parameter sharing as a domain adaptation technique.

# Discussion {#ch:discussion}

Intelligent transportation systems (ITS) are

# Conclusion {#ch:conclusion}

## Conclusion {#conclusion}

## Next Steps
