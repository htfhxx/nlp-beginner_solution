# coding=utf-8
#from __future__ import print_function

import yaml
import optparse
import sys
import time
from util import loader
import itertools
import _pickle as cPickle
from collections import OrderedDict

import torch
from torch.autograd import Variable

from util.utils import *
from util.loader import *
from util.model import BiLSTM_CRF
from util.conlleval import *


optparser = optparse.OptionParser()
optparser.add_option("--config", default="util/config.yaml", type=str, help="config file path")
optparser.add_option("--tag_scheme", default="iobes",help="Tagging scheme, IOB or IOBES")
optparser.add_option('--use_gpu', default='1',type='int', help='whether or not to use gpu')
optparser.add_option('--reload', default='0',type='int', help='whether or not to reload pretrained model')
optparser.add_option('--pretrained_model', default='',type=str, help='pretrained model path')


opts = optparser.parse_args()[0]
config = AttrDict(yaml.load(open(opts.config, 'r')))
# to beifen
parameters = OrderedDict()
parameters['tag_scheme'] = opts.tag_scheme
parameters['lower'] =  config['lower']
parameters['zeros'] = config['zeros']
parameters['char_dim'] = config['char_dim']
parameters['char_lstm_dim'] = config['char_lstm_dim']
parameters['char_bidirect'] = config['char_bidirect']
parameters['word_dim'] = config['word_dim']
parameters['word_lstm_dim'] = config['word_lstm_dim']
parameters['word_bidirect'] =  config['word_bidirect']== 1
parameters['pre_emb'] = config['pre_emb']
parameters['all_emb'] =  config['all_emb'] == 1
parameters['cap_dim'] = config['cap_dim']
parameters['crf'] = config['crf']
parameters['dropout'] = config['dropout']
parameters['reload'] = opts.reload
#parameters['name'] = config['name']
parameters['use_gpu'] = opts.use_gpu == 1 and torch.cuda.is_available()

models_path = config['models_path']
#name = config['name']
#model_name = models_path + name

lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']
use_gpu = parameters['use_gpu']
learning_rate = config['learning_rate']
printloss_after =  config['printloss_after']
#eval_after = config['eval_after']

if not os.path.exists(models_path):
    os.makedirs(models_path)



def train(model, train_data,test_data, dev_data, id_to_tag,tag_to_id ):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    loss = 0.0
    #count = 0
    all_count=0
    model.train(True)
    for epoch in range(1, 100):
        cnt = 0
        for i, index in enumerate(np.random.permutation(len(train_data))):
            all_count += 1
            cnt+=1
            if cnt>200:
                break
            data = train_data[index]
            model.zero_grad()

            sentence_in = data['words']
            sentence_in = Variable(torch.LongTensor(sentence_in))
            tags = data['tags']
            chars2 = data['chars']

            ######### char lstm
            chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

            targets = torch.LongTensor(tags)
            caps = Variable(torch.LongTensor(data['caps']))
            if use_gpu:
                neg_log_likelihood = model.neg_log_likelihood(sentence_in.cuda(), targets.cuda(), chars2_mask.cuda(),
                                                              caps.cuda(), chars2_length, d)
            else:
                neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets, chars2_mask, caps, chars2_length, d)
            # loss += neg_log_likelihood.data[0] / len(data['words'])
            loss += neg_log_likelihood.item() / len(data['words'])
            neg_log_likelihood.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if cnt % printloss_after == 0:
                loss /= printloss_after
                print(cnt,'/',len(train_data), ' : ', loss)
                loss = 0.0

        model.train(False)
        new_dev_recall, new_dev_pre, new_dev_F = evaluating(model, dev_data,id_to_tag,tag_to_id, epoch)
        new_test_recall, new_test_pre, new_test_F = evaluating(model, test_data, id_to_tag, tag_to_id, epoch)
        print('new_dev_recall:%3.6f,  new_dev_pre: %3.6f ,  new_dev_F: %3.6f     ' % (new_dev_recall,  new_dev_pre ,  new_dev_F))
        print('new_test_recall:%3.6f,  new_test_pre: %3.6f ,  new_test_F: %3.6f     ' % (new_test_recall ,new_test_pre , new_test_F  ))
        torch.save(model,models_path+'epoch_%d_%3.6f_%3.6f_%3.6f_%3.6f_%3.6f_%3.6f.model'%(epoch, new_dev_recall,new_dev_pre ,new_dev_F, new_test_recall, new_test_pre, new_test_F ))
        model.train(True)

        adjust_learning_rate(optimizer, lr=learning_rate / (1 + 0.05 * all_count / len(train_data)))

def evaluating(model, datas,id_to_tag,tag_to_id , epoch):
    prediction = []
    save = False
    new_F = -1.0
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
    for data in datas:
        ground_truth_id = data['tags']
        words = data['str_words']
        chars2 = data['chars']
        caps = data['caps']

        chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
        d = {}
        for i, ci in enumerate(chars2):
            for j, cj in enumerate(chars2_sorted):
                if ci == cj and not j in d and not i in d.values():
                    d[j] = i
                    continue
        chars2_length = [len(c) for c in chars2_sorted]
        char_maxl = max(chars2_length)
        chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
        for i, c in enumerate(chars2_sorted):
            chars2_mask[i, :chars2_length[i]] = c
        chars2_mask = Variable(torch.LongTensor(chars2_mask))
        dwords = Variable(torch.LongTensor(data['words']))
        dcaps = Variable(torch.LongTensor(caps))
        if use_gpu:
            val, out = model(dwords.cuda(), chars2_mask.cuda(), dcaps.cuda(), chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, dcaps, chars2_length, d)
        predicted_id = out
        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            prediction.append(line)
            confusion_matrix[true_id, pred_id] += 1
        prediction.append('')

    # save the result
    predf = models_path + '/epoch_%d_pred.txt'%(epoch)
    with open(predf, 'w', encoding='utf-8') as f:
        f.write('\n'.join(prediction))

    true_seqs, pred_seqs = [], []
    with open(predf,'r',encoding='utf-8') as f:
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
    print('result:       ',result)
    print(type(result))
    recall = result[0]
    precision = result[1]
    f1 = result[2]
    return recall, precision, f1

def main():
    # load data
    train_sentences = loader.load_sentences(config['train'], lower, zeros)
    dev_sentences = loader.load_sentences(config['dev'], lower, zeros)
    test_sentences = loader.load_sentences(config['test'], lower, zeros)

    #print(train_sentences)  #  [['a', 'DT', 'I-NP', 'O'], ... , ['lot', 'NN', 'I-NP', 'O']]

    # check tags
    update_tag_scheme(train_sentences, tag_scheme)
    update_tag_scheme(dev_sentences, tag_scheme)
    update_tag_scheme(test_sentences, tag_scheme)

    # get dictionary
    dico_words_train = word_mapping(train_sentences, lower)[0]
    dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )
    dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
    dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

    # prepare data
    train_data = prepare_dataset(train_sentences, word_to_id, char_to_id, tag_to_id, lower )
    dev_data = prepare_dataset(dev_sentences, word_to_id, char_to_id, tag_to_id, lower )
    test_data = prepare_dataset( test_sentences, word_to_id, char_to_id, tag_to_id, lower)
    print("%i / %i / %i sentences in train / dev / test." % (len(train_data), len(dev_data), len(test_data)))

    #prepare word_embeds
    all_word_embeds = {}
    for i, line in enumerate(codecs.open(config['pre_emb'], 'r', 'utf-8')):
        s = line.strip().split()
        if len(s) == parameters['word_dim'] + 1:
            all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])
    word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), config['word_dim']))
    for w in word_to_id:
        if w in all_word_embeds:
            word_embeds[word_to_id[w]] = all_word_embeds[w]
        elif w.lower() in all_word_embeds:
            word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]
    print('Loaded %i pretrained embeddings.' % len(all_word_embeds))

    # create or load model
    model = BiLSTM_CRF(vocab_size=len(word_to_id), tag_to_ix=tag_to_id, embedding_dim=parameters['word_dim'], hidden_dim=parameters['word_lstm_dim'], use_gpu=use_gpu, char_to_ix=char_to_id, pre_word_embeds=word_embeds, use_crf=parameters['crf'])
    if parameters['reload']:
        model.load_state_dict(torch.load(opts.pretrained_model))
    if use_gpu:
        model.cuda()

    #train
    train(model, train_data, test_data, dev_data, id_to_tag, tag_to_id)

if __name__ == '__main__':
    main()