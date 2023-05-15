
# A hybrid feature learning approach based on convolutional kernels for ATM fault prediction using event-log data

This is the companion code for the paper "A hybrid feature learning approach based on convolutional kernels for ATM fault prediction using event-log data".

Authors are:
* Víctor Manuel Vargas (@victormvy)
* Riccardo Rosati (@rosati1392)
* César Hervás-Martínez (chervas@uco.es)
* Adriano Mancini (a.mancini@staff.univpm.it)
* Luca Romeo (@whylearning22)
* Pedro Antonio Gutiérrez (@pagutierrez)

  
## Instructions

The following has been tested to run on an up-to-date Linux installation (Debian 10 buster).

### Preparing the environment

You can use `anaconda` or `miniconda` with an environment that has at least Python 3.9 installed.

Then, you can install the requirements:

```bash
pip install -r requirements.txt
```

### Preparing the data

The dataset presented in this work is contained in this repository in a compressed zip file. To run the experiments, you should unpack it first:

```bash
cd data
unzip sigma_dataset.npy.zip
```

The file obtained after unpacking the zip file is a numpy binary format. The function to load this dataset is enclosed in `functions.py`.
However, the dataset in time series `.ts` format is also included in `data/sigma_dataset.ts.zip`.

### Running the experiments

All the experiments can be run using the `run.py` script:

```bash
python run.py
```

## Citation
### BibTex
```bibtex
@article{vargas2023hybrid,
	title = {An ordinal CNN approach for the assessment of neurological damage in Parkinson’s disease patients},
	journal = {Engineering Applications of Artificial Intelligence},
	year = {2023},
	author = {Víctor Manuel Vargas and Riccardo Rosati and César Hervás-Martínez and Adriano Mancini and Luca Romeo and Pedro Antonio Gutiérrez}
}
```

