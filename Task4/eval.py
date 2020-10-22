# coding=utf-8


import optparse
from util.conlleval import *

optparser = optparse.OptionParser()
optparser.add_option('--pred_file', default='result/epoch_28_pred.txt', type=str, help='to calculate the score.')
opts = optparser.parse_args()[0]

if __name__ == '__main__':
    true_seqs, pred_seqs = [], []
    with open(opts.pred_file,'r',encoding='utf-8') as f:
        for line in f:
            #print(line)
            cols = line.strip().split()
            # each non-empty line must contain >= 3 columns
            if not cols:
                true_seqs.append('O')
                pred_seqs.append('O')
            elif len(cols) < 3:
                raise IOError("conlleval: too few columns in line %s\n" % line)
            else:
                # extract tags from last 2 columns
                true_seqs.append(cols[-2])
                pred_seqs.append(cols[-1])
    result = evaluate(true_seqs, pred_seqs)
    recall = result[0]
    precision = result[1]
    f1 = result[2]
    print('Testing Result: %3.6f,  new_test_pre: %3.6f ,  new_test_F: %3.6f     ' % (recall, precision, f1))