# raa-assess
一个命令行软件，用来评估不同约化氨基酸方案。

有以下几个功能：
- [view](#sc-view)     
    查看氨基酸约化方案
- [reduce](#sc-reduce)    
将序列文件进行约化，并进行序列特征提取
- [eval](#sc-eval)    
利用约化后的序列特征文件评估不同约化氨基酸方案，评估结果保存在json文件里
- [plot](#sc-plot)    
将json里的评估数据绘制评估图
- [fs](#sc-fs)   
指定特征文件进行特征筛选
- [own](#sc-own)    
选择自己的约化氨基酸方案进行约化并评估


Installation
------------
### Linux 
先创建一个虚拟环境,用conda（或virtualenv, pipenv等）:
```{.sourceCode .bash}
$conda create -n test python=3.6
$source activate test
```

安装 raa-assess, 可以简单地用 pip:

``` {.sourceCode .bash}
$ pip install git+https://github.com/huang-sh/raa-assess.git@dev
```

Function
------------
### <a name="sc-view">view</a>
用来查看指定type, size的约化方案。使用参数：
- --type 指定type。可以多个type, 如：--type 1 2 3 或者 --type 1-3 。表示选择type 1,2,3
- --size 和type类似

示例:
``` {.sourceCode .bash}
$ raa view --type 1 2 --size 4 5
type1  4  LVIMC-AGSTP-FYW-EDNQKRH                 BLOSUM50
type1  5  LVIMC-AGSTP-FYW-EDNQ-KRH                BLOSUM50
type2  4  ARNDQEGHKPST-C-ILMFYV-W                 BLOSUM40
type2  5  AGPST-RNDQEHK-C-ILMFYV-W                BLOSUM40
```
第一列是type,第二列是size,第三列是约化方案,第四列是方法。
### <a name="sc-reduce">reduce</a>
将氨基酸fasta序列文件进行约化，然后提取序列特征。使用参数：
- -f 输入序列文件。同一类别的序列占一个文件。
- -k k-mer，限制了k只能等于1,2,3 
- -t 选择type,多个type用"-"连接，如1-15表示type1到type15
- -s 选择size,多个size用"-"连接，如2-16表示size 2到size 16
- -o 输出约化序列特征保存文件名(不含后缀)，不同k 值以不同后缀表示，如-k 1 2时，-o test 则会生成test_1n, test_2n两个文件夹
- -p cpu核数，不能大于系统最大核数 

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
### <a name="sc-own">own</a>
利用不同约化序列文件，评估不同约化氨基酸方案。使用参数：
- -f 输入序列文件。同一类别的序列占一个文件。
- -cluster 一个约化方案，如："SWGA-DRHQYNE-KLVIF-C-PM-T"
- -k 同上
- -cv 同上
- -hpo 同上
- -o 同上

示例：
``` {.sourceCode .bash}
$ raa own -f cls1.fa cls2.fa -cluster SWGA-DRHQYNE-KLVIF-C-PM-T -k 1 2 3 -cv -1 -hpo 1 -o my_clster
```
