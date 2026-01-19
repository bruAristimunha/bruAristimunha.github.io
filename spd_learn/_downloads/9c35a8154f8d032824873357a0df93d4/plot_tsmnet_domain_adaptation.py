"""
.. _tsmnet-domain-adaptation:

Cross-Session Transfer with TSMNet
==================================

This tutorial demonstrates how to use TSMNet for cross-session motor imagery
classification with domain adaptation. TSMNet's SPDBatchNorm layer enables
adaptation to new sessions without labeled data from the target session.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

######################################################################
# Introduction
# ------------
#
# In EEG-based BCIs, a common challenge is **session-to-session variability**:
# models trained on one day often perform poorly on another day
# due to changes in electrode impedance, mental state, and environment.
#
# TSMNet :cite:p:`kobler2022spd` addresses this through **SPDBatchNorm**,
# which:
#
# 1. Normalizes SPD matrices using the Fréchet mean
# 2. Maintains running statistics that can be updated on new data
# 3. Enables **Source-Free Unsupervised Domain Adaptation (SFUDA)**
#
# This means we can adapt to a new subject using only unlabeled data!
#

######################################################################
# Setup and Imports
# -----------------
#

import warnings

import matplotlib.pyplot as plt
import torch

from braindecode import EEGClassifier
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from skorch.callbacks import EpochScoring, GradientNormClipping
from skorch.dataset import ValidSplit

from spd_learn.models import TSMNet


warnings.filterwarnings("ignore")

######################################################################
# Loading the Dataset
# -------------------
#
# BNCI2014_001 contains EEG recordings from 9 subjects performing
# - **22 EEG channels**: Standard 10-20 montage
# - **250 Hz sampling rate**: After resampling
#
# We'll demonstrate **cross-session transfer**:
#
# - **Source domain**: Subject 1, Session 1 (training)
# - **Target domain**: Subject 1, Session 2 (testing/adaptation)
#
# Cross-session transfer is a realistic BCI scenario where we want to avoid
# recalibration for a returning user.
#

dataset = BNCI2014_001()
paradigm = MotorImagery(n_classes=4)

print(f"Dataset: {dataset.code}")
print("Cross-subject transfer: Subject 1 (source) -> Subject 2 (target)")

######################################################################
# Creating the TSMNet Model
# -------------------------
#
# TSMNet architecture:
#
# 1. **Temporal Conv**: Learns temporal filters
# 2. **Spatial Conv**: Learns spatial combinations
# 3. **CovLayer**: Computes covariance matrices
# 4. **BiMap + ReEig**: SPD dimensionality reduction
# 5. **SPDBatchNorm**: Riemannian batch normalization (key for adaptation)
# 6. **LogEig**: Projects to tangent space
# 7. **Linear**: Classification head
#

n_chans = 22
n_outputs = 4

model = TSMNet(
    n_chans=n_chans,
    n_outputs=n_outputs,
    n_temp_filters=8,  # Temporal filters (increased)
    temp_kernel_length=50,  # ~200ms at 250Hz
    n_spatiotemp_filters=32,  # Spatiotemporal features
    n_bimap_filters=16,  # BiMap output dimension
    reeig_threshold=1e-4,  # ReEig threshold
)

print("TSMNet Architecture:")
print(model)

######################################################################
# Training on Source Domain
# -------------------------
#
# First, we train TSMNet on Session 1 (source domain).
#

source_subject = 1
target_subject = 1  # Same subject, different session
batch_size = 32
max_epochs = 100
learning_rate = 1e-4  # Low learning rate for stable SPD learning

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

# Cache configuration
cache_config = dict(
    save_raw=True,
    save_epochs=True,
    save_array=True,
    use=True,
    overwrite_raw=False,
    overwrite_epochs=False,
    overwrite_array=False,
)

# Load data for both subjects
X, labels, meta = paradigm.get_data(
    dataset=dataset,
    subjects=[source_subject, target_subject],
    cache_config=cache_config,
)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

# Split by session
# Session '0train' is the first session (source)
# Session '1test' is the second session (target)
source_idx = meta.query("session == '0train'").index.to_numpy()
target_idx = meta.query("session == '1test'").index.to_numpy()

X_source, y_source = X[source_idx], y[source_idx]
X_target, y_target = X[target_idx], y[target_idx]

print(f"\nSource domain (Session 1): {len(source_idx)} samples")
print(f"Target domain (Session 2): {len(target_idx)} samples")

# Create classifier
# Note: SPD networks benefit from gradient clipping to prevent
# divergence during training on the Riemannian manifold.
clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    optimizer__lr=learning_rate,
    train_split=ValidSplit(0.1, stratified=True, random_state=42),
    batch_size=batch_size,
    max_epochs=max_epochs,
    callbacks=[
        (
            "train_acc",
            EpochScoring(
                "accuracy", lower_is_better=False, on_train=True, name="train_acc"
            ),
        ),
        ("gradient_clip", GradientNormClipping(gradient_clip_value=1.0)),
    ],
    device=device,
    verbose=1,
)

# Train on source domain
print("\n" + "=" * 50)
print("Training on Source Domain")
print("=" * 50)
clf.fit(X_source, y_source)

######################################################################
# Evaluating Without Adaptation
# -----------------------------
#
# Let's first see how the model performs on the target domain
# WITHOUT any adaptation.
#

# Evaluate on source (should be high)
y_pred_source = clf.predict(X_source)
source_acc = accuracy_score(y_source, y_pred_source)

# Evaluate on target WITHOUT adaptation
y_pred_target_no_adapt = clf.predict(X_target)
target_acc_no_adapt = accuracy_score(y_target, y_pred_target_no_adapt)

print(f"\n{'='*50}")
print("Results WITHOUT Domain Adaptation")
print(f"{'='*50}")
print(f"Source Domain Accuracy: {source_acc*100:.2f}%")
print(f"Target Domain Accuracy: {target_acc_no_adapt*100:.2f}%")
print(f"Performance Drop: {(source_acc - target_acc_no_adapt)*100:.2f}%")

######################################################################
# Domain Adaptation via SPDBatchNorm
# ----------------------------------
#
# Now we perform **Source-Free Unsupervised Domain Adaptation (SFUDA)**:
#
# 1. Put the model in eval mode (freeze all parameters)
# 2. Put SPDBatchNorm in train mode (update running statistics)
# 3. Pass target domain data through the model (no labels needed!)
# 4. The running mean adapts to the target domain distribution
#


def adapt_spdbn(model, X_target, n_passes=3, reset_stats=True):
    """Adapt SPDBatchNorm statistics to target domain.

    Parameters
    ----------
    model : nn.Module
        TSMNet model with SPDBatchNorm layer.
    X_target : array
        Target domain data (unlabeled).
    n_passes : int
        Number of passes through the data for statistics update.
    reset_stats : bool
        If True, reset running statistics before adaptation.
        This allows the model to fully adapt to the target domain.

    Returns
    -------
    model : nn.Module
        The adapted model with updated SPDBatchNorm statistics.
    """
    model.eval()  # Freeze other layers

    # Find SPDBatchNorm layers (may be wrapped as ParametrizedSPDBatchNorm)
    spdbn_modules = []
    for module in model.modules():
        class_name = module.__class__.__name__
        if "SPDBatchNorm" in class_name:
            spdbn_modules.append(module)
            if reset_stats:
                # Reset to identity mean and unit variance for fresh adaptation
                module.reset_running_stats()
            module.train()  # Enable running stats update

    print(f"Found {len(spdbn_modules)} SPDBatchNorm layer(s) to adapt")

    # Convert to tensor
    X_tensor = torch.tensor(X_target, dtype=torch.float32)
    if next(model.parameters()).is_cuda:
        X_tensor = X_tensor.cuda()

    # Shuffle data for better statistics estimation
    perm = torch.randperm(len(X_tensor))
    X_tensor = X_tensor[perm]

    # Pass data through model multiple times to update statistics
    with torch.no_grad():
        for pass_idx in range(n_passes):
            # Process in batches
            for i in range(0, len(X_tensor), 32):
                batch = X_tensor[i : i + 32]
                _ = model(batch)
            print(f"  Adaptation pass {pass_idx + 1}/{n_passes} complete")

    model.eval()  # Set everything back to eval
    return model


# Get the underlying model from the classifier
underlying_model = clf.module_

# Adapt to target domain
print("\n" + "=" * 50)
print("Adapting SPDBatchNorm to Target Domain")
print("=" * 50)
adapted_model = adapt_spdbn(underlying_model, X_target, n_passes=5)

######################################################################
# Evaluating After Adaptation
# ---------------------------
#
# .. note::
#
#    Cross-session transfer typically shows distribution shifts
#    that SPDBatchNorm can correct.
#    The improvement depends on:
#
#    - Non-stationarity between sessions
#    - Training convergence on the source session
#    - How well the learned features generalize
#
#    Typical improvements range from 3-10% for cross-session transfer.
#

# Convert target data for prediction
X_target_tensor = torch.tensor(X_target, dtype=torch.float32)
if next(adapted_model.parameters()).is_cuda:
    X_target_tensor = X_target_tensor.cuda()

# Predict with adapted model
with torch.no_grad():
    logits = adapted_model(X_target_tensor)
    y_pred_target_adapted = logits.argmax(dim=1).cpu().numpy()

target_acc_adapted = accuracy_score(y_target, y_pred_target_adapted)
improvement = target_acc_adapted - target_acc_no_adapt

print(f"\n{'='*50}")
print("Results WITH Domain Adaptation")
print(f"{'='*50}")
print(f"Target Accuracy (No Adaptation):   {target_acc_no_adapt*100:.2f}%")
print(f"Target Accuracy (With Adaptation): {target_acc_adapted*100:.2f}%")
if improvement >= 0:
    print(f"Improvement: +{improvement*100:.2f}%")
else:
    print(f"Improvement: {improvement*100:.2f}%")

######################################################################
# Visualizing Results
# -------------------
#

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy comparison
ax1 = axes[0]
conditions = ["Source\n(Train)", "Target\n(No Adapt)", "Target\n(Adapted)"]
accuracies = [source_acc * 100, target_acc_no_adapt * 100, target_acc_adapted * 100]
colors = ["#2ecc71", "#e74c3c", "#3498db"]
bars = ax1.bar(conditions, accuracies, color=colors, edgecolor="black", linewidth=1.5)
ax1.set_ylabel("Accuracy (%)", fontsize=12)
ax1.set_title("Domain Adaptation Results", fontsize=14)
ax1.set_ylim([0, 100])
ax1.axhline(y=25, color="gray", linestyle="--", alpha=0.5, label="Chance level")

# Add value labels
for bar, acc in zip(bars, accuracies):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f"{acc:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

# Add improvement annotation with arrow between target bars
if improvement != 0:
    arrow_y = max(target_acc_no_adapt, target_acc_adapted) * 100 + 15
    ax1.annotate(
        "",
        xy=(2, target_acc_adapted * 100 + 5),
        xytext=(1, target_acc_no_adapt * 100 + 5),
        arrowprops=dict(
            arrowstyle="->",
            color="green" if improvement > 0 else "red",
            lw=2,
            connectionstyle="arc3,rad=0.2",
        ),
    )
    sign = "+" if improvement > 0 else ""
    ax1.text(
        1.5,
        arrow_y,
        f"{sign}{improvement*100:.1f}%",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="green" if improvement > 0 else "red",
    )

ax1.legend()

# Training history
ax2 = axes[1]
history = clf.history
epochs = range(1, len(history) + 1)
ax2.plot(epochs, history[:, "train_loss"], "b-", label="Train Loss", linewidth=2)
ax2.plot(epochs, history[:, "valid_loss"], "r--", label="Valid Loss", linewidth=2)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Loss", fontsize=12)
ax2.set_title("Training History (Source Domain)", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

######################################################################
# Understanding SPDBatchNorm Adaptation
# -------------------------------------
#
# The key insight is that **session variability manifests as a shift in
# the distribution of SPD matrices**. SPDBatchNorm counters this by:
#
# 1. **Centering**: Removes the batch mean (Fréchet mean on SPD manifold)
#
#    .. math::
#
#       \tilde{P}_i = G^{-1/2} P_i G^{-1/2}
#
# 2. **Scaling**: Normalizes dispersion
#
#    .. math::
#
#       \hat{P}_i = \tilde{P}_i^{w/\sqrt{\sigma^2 + \varepsilon}}
#
# When we adapt, we update the running mean :math:`G` and variance
# :math:`\sigma^2` to match the target domain, aligning the distributions
# without any labeled data.
#

######################################################################
# Summary
# -------
#
# In this tutorial, we demonstrated:
#
# 1. Training TSMNet on source session
# 2. Observing performance drop on target session
# 3. Adapting SPDBatchNorm statistics using unlabeled target data
# 4. Achieving improved cross-session transfer performance
#
# Cross-session non-stationarity is a key challenge in BCI. SPDBatchNorm
# adaptation compensates for these shifts by re-centering the covariance
# distribution.
#
# This **source-free unsupervised domain adaptation** is particularly
# valuable in BCI applications where:
#
# - Calibration time should be minimized
# - Users return for multiple sessions
# - Signal properties drift over time
#
