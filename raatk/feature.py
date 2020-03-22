from itertools import product
from functools import partial
from collections import Counter

        
def seq_aac(seq, raa, k=1, gap=0, lam=0):
    """ extract amino acid sequence composition feature
    :param seq: an amino acid seq 
    :param raa: representative aa, list
    :param k: 
    :param gap: 
    :param lam: 
    :return:
    """
    def three_f(idx, seq, k, gap, lam):
        a = (seq[idx + idx*gap + (lam+1)*i] for i in range(k) 
             if (idx + idx*gap + (lam+1)*i) < len(seq))
        return ''.join(a)
    aa = [''.join(aa) for aa in product(raa, repeat=k)]
    f3 = partial(three_f, seq=seq, gap=gap, k=k, lam=lam)
    aa_list = [f3(i) for i in range(len(seq))]
    aa_list = [i for i in aa_list if len(i) ==k]
    aa_dict = Counter(aa_list)
    count = len(aa_list)
    aa_fre = [aa_dict[i] / count for i in aa]
    return aa_fre

# TODO - add other sequence feature
