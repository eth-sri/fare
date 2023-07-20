# only used for ablation

import numpy
a = numpy.load(f"result/_eval/ACSIncome-ALL-2014-L/tree.npy", allow_pickle=True).item()
a_s = {0.0:{}, 0.01:{}, 0.05:{}, 0.1:{}}
for k in a.keys():
    if not k.startswith('k='):
        continue
    try:
        i = a[k][0]['params']['i']
    except:
        continue
    for k_s in a_s.keys():
        if i == k_s:
            a_s[k_s][k] = a[k]
for k in a_s.keys():
    a_s[k]['checksums'] = a['checksums']
    numpy.save(f'result/_eval/ACSIncome-ALL-2014-L/tree_i={k}.npy',a_s[k])
