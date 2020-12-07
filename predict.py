#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
from collections import defaultdict

import tensorflow as tf

from conllu import parse

from model import Model
from data import convert_conllu_to_dataset_for_prediction
from batch import generate_batch
from train import get_string_by_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='CoNLL-U file with train set')
    parser.add_argument('-m', '--model', help='Model prefix')
    args = parser.parse_args()

    tf.reset_default_graph()

    m = Model()
    m.load(args.model)

    session_config = tf.ConfigProto()
    session = tf.Session(config=session_config)

    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    sentences = parse(open(args.input, 'r', encoding='utf-8').read())
    for sent in sentences:
        for tok in sent:
            tok['lemma'] = tok['form']

    data_set, _ = convert_conllu_to_dataset_for_prediction(sentences, 'form',
                                            m.config['encoder']['c2i'], m.config['encoder']['max_len'],
                                            m.config['encoder']['t2i'], m.config['encoder']['feats'], 0, 0,
                                            m.config['encoder']['feats_to_use'])

    uniq_items = []
    for form_len in sorted(data_set.keys()):
        for item in data_set[form_len].values():
            uniq_items.append(item)

    for start in range(0, len(uniq_items), m.config['batch_size']):
        #sys.stderr.write('Batch starts from %d\n' % start)
        batch = generate_batch(uniq_items,
                               options=m.config,
                               randomize=False,
                               gold=False,
                               batch_size=m.config['batch_size'],
                               start=start)

        output = session.run([ m.output['sample_id'] ],
                             feed_dict=m.feed_dict(batch, gold=False))

        #rnn_output, sample_id = output[0].rnn_output.tolist(), output[0].sample_id.tolist()
        #sample_id = output[0].sample_id.tolist()
        #sample_id = output[0].tolist()

        sample_id = []
        for j in range(0, 5):
            sample_id.append(output[0][:,:,j].tolist())

        for i in range(m.config['batch_size']):
            idx = start + i
            if idx >= len(uniq_items):
                break
            dst_str = get_string_by_id(sample_id[0][i], m.options['decoder']['i2c'], m.options['decoder']['c2i'][m.options['EOS']])
            uniq_items[idx]['raw']['dst'] = dst_str
            #print(' '.join([ str(x) for x in sample_id[0][i] ]))
            #print(dst_str)
        #if start == 10000:
        #    break

    for item in uniq_items:
        for token in item['tokens']:
            token_object = sentences[token['sent_idx']][token['tok_idx']]
            token_object['lemma'] = item['raw']['dst']

    for sent in sentences:
        print(sent.serialize(), end='')

    pass


def find_token_with_id(sent, id):
    i = id
    while i <= len(sent):
        if sent[i-1]['id'] == id:
            return sent[i-1]
        i += 1
    return None


def leave_uniq_only(data_set):
    d = defaultdict(lambda: defaultdict(dict))

    for item in data_set:
        form_len = len(item['raw']['src'])
        d[form_len][item['id']] = item

    l = []
    for form_len in sorted(d.keys()):
        for item in d[form_len].values():
            l.append(item)

    return l

if __name__ == "__main__":
    main()
