# Neural Network Accelerated Process Design of Polycrystalline Microstructures

This repository provides code to model microstructure deformation. The main idea is given pairs of ODFs before and after deformations, model the behavior of latent deformation differential equations that governs the deformation processes.

## Requirements
- Python == 3.8.5
- torch == 1.8.0+cu111
- numpy == 1.19.2

## How to run the code
Run the following to install necessary packages
```python
pip install -r requirements.txt
```
To train and evaluate the model, simply run
```python
python3 main.py
```

## Dataset
We use a synthetic dataset for experiment. `init_ODF.npy` contains 5000 uniformly sampled valid ODFs. The correspoding ODFs generated from the finite-elements (FE) simulator after different deformations are stored in `SIMU_*.npy` files. The crystal properies are taken for Copper (Cu).

## Questions and Comments
If there's anything you're interested in with respect to these algorithms, feel free to send the authors an email.