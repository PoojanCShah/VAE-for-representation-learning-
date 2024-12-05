# VAE-for-representation-learning-

Running code for training. save the model in the same directory with name "vae.path"
 Save the GMM parameters in the same folder. You can use pickle to save the parameters. 
```
python vae.py path_to_train_dataset path_to_val_dataset train vae.pth gmm_params.pkl
```

 Running code for vae reconstruction.
 This should save the reconstruced images in numpy format. see below for more details.

```
python vae.py path_to_test_dataset_recon test_reconstruction vae.pth
```

Running code for class prediction during testing
```
python vae.py path_to_test_dataset test_classifier vae.pth gmm_params.pkl
```