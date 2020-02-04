# simple-vae
A simple vae with binary cross entropy loss futo start you off in your protein VAE creating adventures

You can run it by executing
```
python run.py
```

It has 3 datasets: small, medium and large. You can control which dataset is being used in simple_vae.pyu
```python
DATASET_LENGTH = "small" # (small|medium|large)
```

If you make extra models add to the models directory and include them into the run script here:
```
model = VAE(INPUT_DIM, 20).to(device) # 20 is number of hidden dimension
```
