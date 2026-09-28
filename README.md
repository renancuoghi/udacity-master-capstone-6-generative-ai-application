# Generative AI Applications: Handwritten-Digit VAE

This project implements an unconditional Variational Autoencoder (VAE) that learns to generate 8x8 grayscale handwritten-digit images. It was created for the Generative AI Applications capstone project.

The model trains on the UCI Optical Recognition of Handwritten Digits dataset. It learns a 12-dimensional latent representation and creates novel digit-like samples by decoding points drawn from a standard Gaussian distribution.

## Project contents

- generative_model.ipynb — executed notebook with data inspection, VAE implementation, training, diagnostics, generated images, qualitative evaluation, and summary.
- Generative_AI_Analysis_Report.pdf — cited report covering the task, data, architecture, evaluation, ethics, limitations, and references.
- DATASET.md — dataset provenance and citation.
- artifacts/ — generated sample grid, reconstructions, loss plot, trained weights, and training metrics.
- requirements.txt — package versions from the working Python environment.

## Dataset

The included UCI Optical Recognition of Handwritten Digits data contain real handwritten digits from 43 writers, preprocessed to 8x8 pixel grids. Each row has 64 pixel intensities from 0 to 16 followed by a digit label.

The VAE uses pixel values only. Labels are retained solely for inspecting data coverage, so the model is unconditional and cannot be directed to generate a specified class.

Citation: Alpaydin, E., and Kaynak, C. (1998). Optical Recognition of Handwritten Digits [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49

## Setup

The project was developed with Python 3.11. Create and activate a virtual environment, then install the recorded dependencies:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Run the notebook

Start Jupyter from the project root:

    jupyter notebook generative_model.ipynb

Run all cells from top to bottom. The notebook uses a fixed seed of 42 and automatically selects CUDA when available; otherwise it runs on CPU. Outputs are saved in artifacts/.

To execute it non-interactively:

    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 generative_model.ipynb

## Model and evaluation

The encoder maps 64 normalized pixels through dense ReLU layers to the mean and log variance of a 12-dimensional Gaussian latent distribution. The decoder mirrors this architecture and produces 64 sigmoid-bounded pixel values. Training minimizes binary-cross-entropy reconstruction loss plus KL divergence to a unit-Gaussian prior.

The submitted run trains for 20 epochs using Adam with learning rate 0.001 and batches of 128. It produces a loss curve, 24 novel prior samples, and held-out reconstruction comparisons. Qualitative evaluation notes that the model learns common digit strokes and loops while sometimes producing blurred, incomplete, or ambiguous hybrid forms.

## Responsible use

This is an educational generative-modeling exercise, not an identity, authentication, handwriting-verification, or fraud-detection system. The training data cover a limited and undocumented set of writing styles, so outputs should not be treated as representative of broader populations or real-world handwriting.
