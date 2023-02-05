# BFReg

Supplements of toy data and code for Biological Regulatory Network.

We provide a preprossed toy dataset for missing value prediction task. Other data and preprocessing codes will be released after the paper is accepted.

## File introductions

c_main.py cell classification task.

mask_main.py missing value prediction task.

c_model.py model construction.

module.py BFReg-NN blocks.


## Usage

### enviroment
python == 3.7

pytorch == 1.31

pygeometric

geomloss

torchdiff

### run toy data

missing value task.

```
$ python mask_main.py
```

cell classification task.
```
$ python c_main.py
```

future prediction task.
```
$ python d_main.py
```

cell trajectory task. Data are available without preprocessing at https://www.synapse.org/#!Synapse:syn20366914/wiki/593925 
```
$ python gt_main.py
```

