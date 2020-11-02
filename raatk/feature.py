from itertools import product
from functools import partial
from collections import Counter

        
def seq_aac(seq, raa, k=1, gap=0, lam=0, count=False):
    """ extract amino acid sequence composition feature
    :param seq: an amino acid seq 
    :param raa: representative aa, list
    :param k: 
    :param gap: 
    :param lam: 
    :param count:
    :return:
    """

    def three_f(idx, k, gap, lam, seq_dic):
        a = (seq_dic.get(idx+idx*gap+(lam+1)*i, '') for i in range(k))
        return ''.join(a)
    seq_dic = {i: v for i,v in enumerate(seq)}
    aa = [''.join(aa) for aa in product(raa, repeat=k)]
    f3 = partial(three_f, gap=gap, k=k, lam=lam, seq_dic=seq_dic)
    aa_list = [f3(i) for i in range(len(seq))]
    aa_list = [i for i in aa_list if len(i) ==k]
    aa_dict = Counter(aa_list)
    all_count = len(aa_list)
    if count:
        all_count = 1 
    aa_fre = [aa_dict[i] / all_count for i in aa]
    return aa_fre

#TODO - add other sequence feature
