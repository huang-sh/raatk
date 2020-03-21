# RAATK
A python-based reduce amino acid toolkit of machine learning for protein-dependent inference.

## sub-commands
- [view](#sc-view)
- [reduce](#sc-reduce)
- [extract](#sc-extract)    
- [eval](#sc-eval)    
- [plot](#sc-plot)    
- [train](#sc-train)    
- [predict](#sc-predict)    
- [ifs](#sc-ifs)   
- [hpo](#sc-hpo)    
- [split](#sc-split)


Installation
------------
create a new virtual environment:
```{.sourceCode .bash}
$conda create -n test python=3.6
$source activate test
```
install with pip:
``` {.sourceCode .bash}
$ pip install git+https://github.com/huang-sh/raa-assess.git@dev
```

Function
------------
### <a name="sc-view">view</a>
view reduced amino acid alphabets
```
Usage: raatk view [-h] -t {1,2,3,...,73,74} -s {2,3,...,18,19}

optional arguments:
  -h, --help            show this help message and exit
  -t, --type            type id
  -s, --size            reduced size
```
for example:
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
Usage: raatk reduce [-h] -t {1,2,3,...,73,74} -s {2,3,...,18,19}

optional arguments:
  -h, --help            show this help message and exit
  -t, --type            type id
  -s, --size            reduced size
```

示例：
``` {.sourceCode .bash}
$ raa reduce -f cls1.fa cls2.fa -k 1 2 -t 1-10 -s 2-19 -o test
```
### <a name="sc-eval">eval</a>
利用不同约化序列文件，评估不同约化氨基酸方案。使用参数：
- -input 序列特征文件夹名（不包含后缀：_1n, _2n, _3n）,结果保存在input参数文件夹里里
- -k k-mer，限制了k只能等于1,2,3 
- -cv 验证方法，cv=-1代表留一法, cv>于1则代表cv折交叉验证
- -hpo 选择数据的百分之几来进行超参数寻优，如hpo=1则表示全部训练数据进行超参数巡优，0.5则表示数据的一半进行参数寻优
- -v 评估数据是否进行可视化
- -fmt 可视化图片格式
- -p cpu核数，不能大于系统最大核数。默认系统核数一半

示例：
``` {.sourceCode .bash}
$ raa eval -input test -k 1 2 -cv -1 -hpo 1
```
### <a name="sc-eval">plot</a>
评估结果作图。使用参数：
- -f json数据文件地址。可多个地址，以空格隔开
- -fmt 可视化图片格式
- -o 输出图片保存地址 

示例：
``` {.sourceCode .bash}
$ raa plot -f 1n_result.json 2n_result.json -fmt png -o test
```
### <a name="sc-fs">fs</a>
选择特定氨基酸约化方案进行特征筛选。使用参数：
- -f 特征文件地址。可多个，以空格隔开
- -o 输出结果保存地址 
- -cv 同上
- -hpo 同上
- -fmt 可视化图片格式
- -mix 如果有多个特征文件，此参数表示将多个特征文件的特征放在一起进行特征筛选

示例：
``` {.sourceCode .bash}
$ raa fs -f test_1n/type1/6_2n.csv -cv -1 -hpo 1 -fmt jpg
```

