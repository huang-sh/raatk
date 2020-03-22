# RAATK
A python-based reduce amino acid toolkit of machine learning for protein-dependent inference.

Installation
------------
create a new virtual environment:
```{.sourceCode .bash}
$conda create -n test python=3.6
$source activate test
```
install
``` {.sourceCode .bash}
$ pip install git+https://github.com/huang-sh/raatk.git@dev -U
```
 Function
 ------------
- [view reduced amio acid alphabet](#sc-view)
- [reduce amino acid sequence](#sc-reduce)
- [extract sequence feature](#sc-extract)    
- [evaluation](#sc-eval)    
- [visualization](#sc-plot)    
- [train model](#sc-train)    
- [prediction](#sc-predict)    
- [feature selection](#sc-ifs)   
- [hyper-parameter optimization](#sc-hpo)    
- [split data](#sc-split)


sub-command
------------
### <a name="sc-view">view</a>
view reduced amino acid alphabets
```bash
Usage: raatk view [-h] -t {1,2,3,...,73,74} -s {2,3,...,18,19}

optional arguments:
  -h, --help            show this help message and exit
  -t, --type            type id
  -s, --size            reduced size
```
example:
``` {.sourceCode .bash}
$ raa view -t 1 2 -s 4 5
type1  4  LVIMC-AGSTP-FYW-EDNQKRH                 BLOSUM50
type1  5  LVIMC-AGSTP-FYW-EDNQ-KRH                BLOSUM50
type2  4  ARNDQEGHKPST-C-ILMFYV-W                 BLOSUM40
type2  5  AGPST-RNDQEHK-C-ILMFYV-W                BLOSUM40
```

### <a name="sc-reduce">reduce</a>
reduce amino acid sequences
```
Usage: raatk reduce
```

example：
``` {.sourceCode .bash}
$ raatk reduce -f file1.fa file2.fa -k 1 2 -t 1-10 -s 2-19 -o test
```


### <a name="sc-extract">extract</a>

extract features of  amino acid sequence 

```
Usage: raatk extract
```

example:

```bash
$ raatk extract
```

### <a name="sc-eval">eval</a>

evaluate different reduced amino acid alphabets and model
```
Usage: raatk eval
```



示例：
``` bash
$ raatk eval -input test -k 1 2 -cv -1 -hpo 1
```
### <a name="sc-plot">plot</a>
visualization of evaluation reports
```
Usage: raatk plot
```



example:
``` bash
$ raatk plot -f 1n_result.json -fmt png -o test
```
### <a name="sc-train">train</a>
train model
```
Usage: raatk train
```

example:
``` bash
$ raatk train
```

### <a name="sc-predict">predict</a>

predict new data 

```
Usage: raatk predict
```

example:

```bash
$ raatk predict 
```

### <a name="sc-ifs">ifs</a>

feature selection

```
Usage: raatk ifs
```

example:

```bash
$ raatk ifs
```

### <a name="sc-hpo">hpo</a>

hyper-parameter optimization

```
Usage: raatk hpo
```

example:

```bash
$ raatk hpo
```

### <a name="sc-split">split</a>

split data

```
Usage: raatk split
```

example:

```bash
$ raatk split
```

