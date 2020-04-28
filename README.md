# Amino acid vae
Amino acid vae using multiple methods, most important ones are the transformer convolutional and global context vae

## Installing dependencies

```
pip install -r requirements.txt
```

## Loading data
Data can be downloaded using either S3 connector or from google drive. 
Data download from gdrive can be done by running the following:
```
python download_data.py
```

## Running a model

The VAE can be trained using run.py file, to get the exact commands, run the file with help command
```
python run.py --help
```

## Making a new model
New models can be added in the models directory, and the constructor can be connected from the utils/modelfactory.py 

## Exporting embeddings
To export embeddings adapt the script in export.py. The script can be run to export some naive embeddings by running
```
python export.py
```

## Architecture
The code is divided into two parts, the core components, the builders and the executors.
To avoid creating a monolith, the executors can be multiple. The core components and the builders 
should not have any redundancies. 
