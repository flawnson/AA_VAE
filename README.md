# simple-vae
A simple vae to start you off in your protein VAE creating adventures

## Loading data
You first need to download the data. For this you need the gdown package. You can install it with
```
pip install gdown
```

Then you need to download the data using
```
python download_data.py
```

## Running a model

You can run the current basic model by executing
```
python run.py
```

You can control which size of data set is used with the following line in run.py
```python
DATASET_LENGTH = "small" # (small|medium|large)
```

It also has 2 protein lengths, which you can change with this line
```python
FIXED_PROTEIN_LENGTH = 50 # (50|200)
```

## Making a new model
If you make extra models add to the models directory and include them into the run script here:
```
model = VAE(INPUT_DIM, 20).to(device) # 20 is number of hidden dimension
```

## Exporting embeddings
If you want to export embeddings adapt the script in export.py. The script can be run to export some naive embeddings by running
```
python export.py
```
