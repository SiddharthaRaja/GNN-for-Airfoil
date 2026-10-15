# GNN-for-Airfoil — Interview Questions & Model Answers

> Companion to [`GNN-for-Airfoil.md`](./GNN-for-Airfoil.md). Answers are first person and sized to say out loud in 30–90 s. "Deeper note" = extra detail if probed.
> 🟢 Easy · 🟡 Medium · 🔴 Hard

**If you only prep 10:** A1, A3, B2, C1, C3, D1, D3, E1, G1, and all the hostile follow-ups.

---

## (a) Project overview & motivation

**A1. 🟢 What is this project?**
> It's a graph-neural-network surrogate for airfoil CFD. AirfRANS has about a thousand RANS simulations of 2-D airfoils, each on an unstructured mesh of roughly 10⁵ nodes. I turn each mesh into a graph and train GNNs (a GCN baseline, plus GAT and GraphSAGE with angle-of-attack and Reynolds conditioning) to predict pressure at every node. The aim is to replace minutes of CFD with a millisecond forward pass.

**A2. 🟢 Why does a CFD surrogate matter?**
> Design optimisation and uncertainty quantification need thousands of flow evaluations. If each takes minutes, you can't explore the design space. A surrogate trades some accuracy for orders-of-magnitude speed. You use it to screen designs, then verify the best few with real CFD.

**A3. 🟡 What's the current state? Do you have results?**
> No trustworthy numbers yet, and I'd rather say that upfront. The README lists target outcomes (R² above 0.9, lift error under 10%), but no results are committed. When I audited the code, I found the raw-data loader doesn't match the `airfrans` API, the pre-processed path collapses train/val/test into overlapping slices, and there's a loss-shape bug. So the value today is the design and the audit. The next step is fixing those and running on the official split.

**A4. 🟢 Why AirfRANS?**
> It's a NeurIPS benchmark dataset with high-fidelity RANS solutions, standard train/test splits including extrapolation tasks in Reynolds and AoA, and published baselines to compare against. It's also big enough (about 9 GB) to be realistic.

**A5. 🟡 Why is this relevant to an ML role and not just mechanical engineering?**
> It's geometric deep learning on irregular data: variable-size graphs, conditioning on global parameters, and physics-aware evaluation. The same skills apply to molecules, meshes, point clouds and recommender graphs. My mechanical background is what let me spot that the "lift" metric wasn't physically correct.

---

## (b) Architecture & design

**B1. 🟢 Walk me through the pipeline.**
> Load the simulations, convert each to a PyG `Data` object (node features: position, inlet velocity and distance-to-airfoil; edges: a 6-nearest-neighbour graph; target: pressure), z-score normalise, and save as `.pt`. Then the training scripts train GCN, GAT and SAGE with Adam and MSE, keep the best checkpoint on validation, and evaluate on test with MSE, R² and lift error.

**B2. 🟡 Why build a k-NN graph instead of using the mesh?**
> It's simpler: you only need point positions, not the mesh connectivity files. But it's a compromise. The CFD mesh is graded, very fine near the wall, so a fixed k gives very different spatial reach in different regions. And k-NN in raw coordinates can link nodes across the thin trailing edge, i.e. through the solid. Using the real mesh edges, or adding relative-position edge features, is better.

**B3. 🟡 How is angle of attack/Reynolds injected?**
> A small MLP embeds `[AoA, Re]` into a 64-d vector, which is broadcast to every node of that graph and concatenated after the convolutions. One caveat: the per-node inlet velocity, U∞(cos α, sin α), already encodes AoA and speed, so it's partly redundant. And Re (about 10⁶) must be normalised or log-scaled first, which the code doesn't do.

**B4. 🟡 Why three architectures?**
> To compare aggregation schemes. GCN is degree-normalised averaging (a low-pass filter). GraphSAGE keeps the self and neighbour terms separate, which preserves sharper local features. GAT learns per-edge attention, which in principle can focus on the upstream or near-wall neighbours that matter. It's a sensible ablation, as long as the evaluation is valid.

**B5. 🔴 What architecture would you use if you started again?**
> An encoder-processor-decoder like MeshGraphNets. Encode node and edge features (relative positions) with MLPs, run about 10–15 message-passing blocks with residuals and LayerNorm, then decode u, v, p and ν_t jointly. Pressure is governed by an elliptic equation, so information has to travel across the whole domain. Either deep message passing, multi-scale pooling (Graph U-Net), or a global attention mechanism (Transolver-style) is needed. Two layers isn't enough.

---

## (c) Code-level / implementation

**C1. 🔴 What does `af.dataset.load` return, and how does the code use it?**
> In airfrans 0.1.5.1 it returns a tuple: a list of `(N, 12)` numpy arrays and a list of simulation names. The columns are x, y, inlet u, inlet v, signed distance, two normal components, then u, v, p/ρ, ν_t and a surface flag. The code does `dataset[:n_samples]`, which slices the *tuple*, then expects each sample to have `.pos` or dict keys. Arrays don't, so every sample throws and preprocessing ends with "no valid samples processed". The fix is unpacking the tuple and indexing columns.

**C2. 🟡 What's wrong with the `.pth` loading branch?**
> It ignores the `split` argument and returns `all_data[:n]` for train, val and test. So val and test are the first 50 graphs, which are also inside the first 100 training graphs. The model is tested on data it trained on.

**C3. 🔴 There's a broadcasting bug in the loss. Explain it.**
> `y` is stored as `(N,1)`. The model output is squeezed to `(N,)`. `nn.MSELoss` broadcasts `(N,)` against `(N,1)` to an `(N,N)` matrix and only warns. So the loss is the mean squared difference between every prediction and every target, which is meaningless. With 10⁵-node meshes in batches of 8, that's an impossible allocation. The fix is to keep both as `(N,1)` or flatten both.

**C4. 🟡 How does PyG batch multiple graphs, and where does this code get it wrong?**
> PyG concatenates node tensors along dim 0, offsets `edge_index`, and adds a `batch` vector mapping nodes to graphs. Graph-level attributes stored as 1-D tensors get concatenated too. Two features per graph becomes `(2B,)`, not `(B,2)`. The graph MLP expects `(B,2)`. Storing `graph_features` as `(1,2)` per graph fixes it, and then `graph_emb[batch]` broadcasts to nodes without the Python loop.

**C5. 🟡 How is the per-graph lift computed in a batch, and is that right?**
> It assumes each graph has `N_total / num_graphs` nodes and slices evenly. AirfRANS meshes have different node counts, so the slices straddle graphs. `batch.ptr` gives the exact boundaries. `evaluate.py` happens to avoid this by using batch size 1.

**C6. 🟢 How does checkpointing work?**
> After each epoch, if validation loss improves I save `state_dict` to `best_model.pt`. After training I reload the best and evaluate on test. It's early-stopping-style model selection, without stopping early.

**C7. 🟡 What will break on a newer environment?**
> `torch.load` defaults to `weights_only=True` from PyTorch 2.6, so loading pickled `Data` lists fails without `weights_only=False`. `torch_geometric.data.DataLoader` is the deprecated location (it's `torch_geometric.loader` now). And `np.trapz` is deprecated in NumPy 2. I'd pin versions and update the imports.

**C8. 🟡 How are features normalised and what's the issue?**
> Z-scores computed over all nodes of a split. The issue is it's done separately per split, and `norm_stats.pt` gets overwritten each time, so the saved stats are the test set's. Normalisation should be fit once on train and applied everywhere.

---

## (d) Underlying ML/CS theory

**D1. 🟡 Explain message passing.**
> Each layer, every node computes messages from its neighbours' features, aggregates them with a permutation-invariant function (sum, mean or max), and combines the result with its own state through a learned function. After L layers, a node's representation depends on its L-hop neighbourhood. GCN, SAGE and GAT are different choices of message and aggregation.

**D2. 🟡 What's the GCN propagation rule?**
> H' = σ(D̂^{-1/2} Â D̂^{-1/2} H W), with Â = A + I. So it's a symmetric degree-normalised average of self and neighbours, then a linear map. Spectrally it's a first-order approximation of a graph low-pass filter. That's why deep GCNs over-smooth.

**D3. 🔴 Why might a 2-layer GNN fundamentally struggle to predict pressure?**
> Pressure in incompressible flow satisfies a Poisson-type equation. It's elliptic, so every point depends on boundary conditions everywhere. A 2-layer GNN on a 6-NN graph has a receptive field of about 40 nearby nodes, a tiny patch of a 10⁵-node mesh. The model can't "see" the airfoil from most of the field. You need depth, multi-scale hierarchies, or global attention.

**D4. 🟡 GAT vs GraphSAGE: when would each win?**
> GAT helps when neighbours have unequal importance and the graph is noisy: attention can down-weight irrelevant ones. SAGE is cheaper, and its separate self/neighbour weights avoid washing out a node's own signal. For dense, regular meshes SAGE-style or edge-conditioned convolutions are often as good. GAT's attention doesn't use geometry unless you add edge features.

**D5. 🟡 What's the pressure coefficient, and why normalise by dynamic pressure?**
> Cp = (p − p∞) / (½ρU∞²). Dividing by dynamic pressure makes it dimensionless, so cases at different speeds are comparable, and Cp distributions are what aerodynamicists read. AirfRANS stores p/ρ, so you'd subtract the freestream and divide by ½U∞².

**D6. 🔴 How do you compute lift from a pressure field correctly?**
> Integrate pressure (and wall shear) over the airfoil surface, using the surface normals, and project onto the direction perpendicular to the freestream: C_L = (1/c)∮(−Cp n)·e_L ds, plus a viscous term. AirfRANS gives you the surface flag and normals. The thin-airfoil shortcut is ∫(Cp_lower − Cp_upper) dx/c. My code instead integrates −Cp dx over a horizontal band of points near y=0. That's not the surface, and it mixes the two sides.

**D7. 🟢 What's over-smoothing?**
> As you stack averaging layers, node representations converge towards each other, so you lose discriminative local information. Residual connections, normalisation, gating or jumping-knowledge connections mitigate it.

---

## (e) Evaluation & metrics

**E1. 🔴 Is your evaluation valid? What would make it valid?**
> As written, no. Test overlaps train on the `.pth` path and equals val on the raw path. Normalisation is per split. Metrics pool all nodes in normalised units. To make it valid: the official AirfRANS `full` split, validation carved from training by geometry, train-only normalisation, surface Cp error, integrated C_L and C_D with Spearman correlation as in the AirfRANS paper, and a no-message-passing MLP baseline to show the graph structure helps.

**E2. 🟡 Why is pooled-node R² misleading here?**
> Most nodes are far from the airfoil, where pressure is close to freestream. A model that's right there and wrong at the leading-edge suction peak can still get a high R². Engineers care about surface pressure and integrated forces, so those should be the headline metrics.

**E3. 🟡 What does "unseen airfoil" generalisation require?**
> The test set must contain geometries the model never saw at any AoA/Re. The official AirfRANS split guarantees that. A random split over simulations would leak the same shape at different conditions.

**E4. 🟡 How would you test extrapolation?**
> AirfRANS has `reynolds` and `aoa` tasks, where test conditions lie outside the training range. Surrogates usually degrade badly there. Reporting both interpolation and extrapolation is honest and informative.

**E5. 🟢 What baseline would you include?**
> A per-node MLP on the same features (no edges) tells you whether message passing adds anything. A k-nearest-simulation interpolation baseline is also useful. And the published AirfRANS results for GraphSAGE, PointNet and the MLP give an external yardstick.

---

## (f) Scaling, production & system design

**F1. 🟡 Each mesh has ~10⁵ nodes. How do you train efficiently?**
> Subsample nodes per step (AirfRANS papers sample around 32k points), use mixed precision, and use neighbour-sampling loaders. At inference you run on the full mesh. Also precompute graphs once and store them compactly instead of re-running k-NN every time.

**F2. 🟡 How would you deploy a surrogate in a design loop?**
> Wrap the model with a preprocessing step that turns a new geometry into a mesh, or a point cloud, plus features. Return Cp on the surface plus C_L and C_D, with uncertainty, e.g. from an ensemble or MC dropout. Then gate the decision: if uncertainty is high or the inputs are outside the training range, fall back to real CFD.

**F3. 🔴 How do you make the surrogate trustworthy for engineers?**
> Physics checks: lift-curve slope near 2π/rad at low AoA, a Cp of about 1 at stagnation, smooth Cp along the surface. Out-of-distribution detection on geometry and conditions. Calibrated uncertainty. And continuous comparison against spot-checked CFD runs. Optionally, physics-informed losses (divergence-free velocity) if predicting the full state.

**F4. 🟢 Storage concerns?**
> The raw dataset is 9.3 GB. Preprocessed graphs with `edge_index` are larger than the raw arrays (2×6N integers per graph). I'd store float16 features and int32 indices, or rebuild edges on the fly.

---

## (g) Failure modes & debugging

**G1. 🔴 Preprocessing prints "error processing sample 0", then "no valid samples processed". How do you debug it?**
> First, inspect `type(dataset)`. It's a tuple from `af.dataset.load`. Then `type(sample)` inside the loop, which is a list, not a simulation. Then check the library docstring for the return format. That's the root cause: unpack `data_list, names` and index array columns. Lesson: when a broad `except` swallows errors, print the exception type and the sample type before anything else.

**G2. 🟡 Training immediately runs out of memory. Why?**
> Most likely the `(N,)` vs `(N,1)` MSE broadcast creating an N×N tensor. Otherwise it's batch size 8 on 10⁵-node graphs with GAT's 4 heads. Check `out.shape` and `batch.y.shape` first, then reduce batch size or subsample nodes.

**G3. 🟡 Validation loss is suspiciously close to training loss from epoch 1. Why?**
> On the `.pth` path, validation graphs are a subset of training graphs. Hash the sample IDs or names in each split and assert they're disjoint.

**G4. 🟡 The lift-vs-AoA plot looks flat. What would you check?**
> Whether AoA is actually varying. `create_graph_data` defaults AoA to 0 if missing, so all samples might share one value. Also, the "lift" integrates a band of points near y=0 rather than the surface, which can hide the true trend.

**G5. 🟢 Results differ between runs. Why?**
> No seeds set, shuffling, dropout, and non-deterministic scatter operations on GPU. Set `torch.manual_seed`, `numpy` seeds and deterministic flags where possible, and report the mean ± std over several seeds.

---

## (h) "Why didn't you…" / alternatives

**H1. 🟡 Why not a CNN on a grid?**
> You'd interpolate the unstructured mesh onto a Cartesian grid, losing the fine near-wall resolution where the important physics is, and wasting cells inside the airfoil. Graphs or point clouds keep the native resolution.

**H2. 🟡 Why not a physics-informed neural network (PINN)?**
> PINNs solve one PDE instance by minimising residuals. They're slow to train per case and don't amortise across geometries. A data-driven surrogate amortises: train once, then predict new cases instantly. Hybrids, data-driven plus physics regularisation, are a good middle ground.

**H3. 🟡 Why not neural operators (FNO/DeepONet)?**
> FNO assumes regular grids for its FFTs. Geometry-aware variants (Geo-FNO, GINO) exist and would be a strong comparison. GNNs were the natural first step for unstructured meshes.

**H4. 🟢 Why not predict velocity and turbulent viscosity too?**
> I started with pressure because it determines lift and is a single output. Jointly predicting u, v, p and ν_t is the AirfRANS benchmark task, and multi-task learning often regularises the shared representation. It's on the improvement list.

**H5. 🟡 Why k=6?**
> It's roughly the number of mesh neighbours in a 2-D triangulation. But it wasn't tuned, and with graded meshes a radius-based graph or the true mesh edges would be more principled.

---

## (i) Behavioural (STAR)

**I1. 🟡 Tell me about debugging something that "worked" but was wrong.**
> **S:** The training scripts produced loss curves and plots, so it looked like it worked. **T:** Before presenting results, I wanted to verify the data path end to end. **A:** I read the `airfrans` source and found `load()` returns a tuple, so the raw path couldn't produce samples, meaning results must have come from pre-processed `.pth` files. Then I saw that path ignores the split argument, so test was inside train. **R:** I stopped treating any numbers as results and wrote down the fixes: the correct API usage, official splits, train-only normalisation and the loss-shape fix. Lesson: assert split disjointness in code.

**I2. 🟢 Tell me about working with a large dataset under constraints.**
> **S:** AirfRANS is 9.3 GB and can't be partially downloaded. **T:** Make it usable for teammates on slow connections. **A:** I added detection of manually placed zips with integrity checks, recursive manifest discovery, a status script, and a quick-preprocess mode. **R:** People could reuse a downloaded zip instead of re-fetching. In hindsight I'd remove the interactive prompts so it also works in batch jobs.

**I3. 🟡 Tell me about bringing domain knowledge to an ML problem.**
> **S:** The project reported a "lift error". **T:** Check that the metric meant what it claimed. **A:** From aerodynamics, lift comes from integrating pressure times the surface normal over the airfoil. The code integrated a horizontal band of field points and mixed upper and lower surfaces, using kinematic pressure, not Cp. **R:** I specified the correct computation using AirfRANS's surface flag and normals. That's the kind of check a pure-ML person might miss.

**I4. 🟡 Tell me about a collaboration on this project.**
> *(Fill in truthfully.)* The README's quick-start points at `anaya-rs/GNN-Airfoil-Project`, so this was a team effort. Template: "I owned X (e.g. the GAT/SAGE conditioning and evaluation). My teammate owned Y. We split work by pipeline stage. In hindsight we needed a shared contract for data shapes, since the `(N,1)` vs `(N,)` mismatch was exactly an interface problem between stages."

---

## Hostile follow-ups

**X1. 🔴 "So you've never actually trained this on the real dataset?"**
> Not through the raw path in this repo, because it can't load the official format. Any runs went through pre-processed `.pth` graphs, and that path collapses the splits. So I don't claim generalisation results. What I can show is the design, the audit and a concrete fix list. I'd rather tell you that than quote a number I can't defend.

**X2. 🔴 "Your 'lift' isn't lift. Why should I trust anything else?"**
> Fair challenge. The lift function is a placeholder approximation, and the comment in the code even says "simplified". The rest (MSE and R² on node pressure) is mechanically correct but measured on bad splits. The right move is to rebuild evaluation around the AirfRANS surface flag and normals, which I've specified.

**X3. 🔴 "Two-layer GNN on 10⁵ nodes: did you not know pressure is global?"**
> I did by the end. That's exactly the limitation I'd fix first on the modelling side. The shallow model was a baseline to get the pipeline running. The literature (MeshGraphNets, AirfRANS baselines, Transolver) uses much deeper or multi-scale models for this reason.

**X4. 🔴 "Your MSE is an N×N broadcast. How did training even run?"**
> If the `.pth` graphs stored `y` as 1-D, the shapes would match and it would run. On graphs produced by `create_graph_data`, it wouldn't, or it would hit OOM. Either way it's an interface bug between preprocessing and training. A single shape assert would have caught it.

**X5. 🔴 "Why is val normalised with different statistics than train?"**
> Because normalisation is called per split. It's a bug. It leaks test information and changes the target scale per split, and the saved stats end up being the test set's. Fit on train once, save, apply everywhere.

**X6. 🔴 "The README says R² > 0.9. Where's that from?"**
> It's listed under "expected outcomes", a target, not a measurement. I should make that clearer, or remove it until I have a run on the official split.

**X7. 🔴 "Did you write this, or is it your teammate's repo?"**
> *(Answer truthfully.)* The README clones from a teammate's repository. Be ready to state which files and decisions are yours and to walk through them line by line.

**X8. 🟡 "Isn't AoA already in the inlet velocity? Why the graph MLP?"**
> Yes. Inlet velocity is U∞(cos α, sin α), so AoA and speed are encoded per node. The graph-level path mainly adds Re, which isn't otherwise present, but it's un-normalised. I'd keep a global conditioning path but feed it log-Re and normalised AoA, or use FiLM-style modulation of each layer.
