#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018-2020 CEA LIST
#
# This file is part of LIMA.
#
# LIMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# LIMA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with LIMA.  If not, see <https://www.gnu.org/licenses/>

import sys
import argparse
import json
import time
import math

import tensorflow as tf

from conllu import parse

from model import Model
from data import convert_conllu_to_dataset, build_dict, build_tag_dict, build_feats_dict, UNK, GO, EOS
from batch import generate_batch
from config.default import options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train-set', help='CoNLL-U file with train set')
    parser.add_argument('-d', '--dev-set', help='CoNLL-U file with dev set')
    parser.add_argument('-a', '--aux-set', help='Plain text file with aux set for self-supervised training')
    parser.add_argument('-c', '--morpho-conf', help='Name of configuration file from morphosyntax')
    parser.add_argument('-m', '--model-prefix', help='Saved model file names prefix (.json and .pb)')
    parser.add_argument('-n', '--num-models', type=int, help='How many models to train')
    parser.add_argument('-l', '--iter-len', type=int, default=1000, help='Iteration length (in batches)')
    parser.add_argument('-i', '--num-iters', type=int, help='Train for num_iters iterations')

    args = parser.parse_args()

    if args.morpho_conf is not None:
        train_set, options['encoder']['max_len'], options['decoder']['max_len'], \
        options['encoder']['c2i'], options['decoder']['c2i'], \
        options['encoder']['t2i'], options['encoder']['i2t'], \
        options['encoder']['feats'], options['encoder']['feats_to_use'] \
            = load_set(args.train_set, 'form', 'lemma', 'upostag', args.morpho_conf)
    else:
        train_set, options['encoder']['max_len'], options['decoder']['max_len'], \
        options['encoder']['c2i'], options['decoder']['c2i'], \
        options['encoder']['t2i'], options['encoder']['i2t'], \
        options['encoder']['feats'], options['encoder']['feats_to_use'] \
            = load_set(args.train_set, 'form', 'lemma', 'upostag')

    options['encoder']['i2c'] = sorted(options['encoder']['c2i'].keys(), key=lambda x: options['encoder']['c2i'][x])
    options['decoder']['i2c'] = sorted(options['decoder']['c2i'].keys(), key=lambda x: options['decoder']['c2i'][x])

    options['encoder']['embd']['char']['size'] = len(options['encoder']['c2i'].keys())
    options['encoder']['embd']['tag']['size'] = len(options['encoder']['t2i'].keys())
    options['decoder']['embd']['char']['size'] = len(options['decoder']['c2i'].keys())

    options['EOS'], options['UNK'], options['GO'] = EOS, UNK, GO

    if len(args.dev_set) > 0:
        dev_set = parse(open(args.dev_set, 'r', encoding='utf-8').read())
        sys.stderr.write('Loaded: %d from %s\n' % (len(dev_set), args.dev_set))

        dev_set, _ = convert_conllu_to_dataset(dev_set, 'form', 'lemma',
                                             options['encoder']['c2i'], options['encoder']['max_len'],
                                             options['decoder']['c2i'], options['decoder']['max_len'],
                                             options['encoder']['t2i'], options['encoder']['feats'], 0, 0, # 3, 3
                                             options['encoder']['feats_to_use'])

        print('INFO: devset size: %d' % len(dev_set))
    else:
        dev_set = None

    train_model(options, train_set, dev_set, args)


def load_set(fn, src_field, dst_field, ctag, morpho_conf=None):
    conllu_data = parse(open(fn, 'r', encoding='utf-8').read())
    sys.stderr.write('Loaded: %d from %s\n' % (len(conllu_data), fn))

    src_descr = {}
    src_descr['i2c'], src_descr['c2i'], src_descr['max_len'] = build_dict(conllu_data, src_field, 3, [UNK, EOS])
    dst_descr = {}
    dst_descr['i2c'], dst_descr['c2i'], dst_descr['max_len'] = build_dict(conllu_data, dst_field, 3, [UNK, GO, EOS])
    dst_descr['max_len'] += 1
    pos_descr = {}
    pos_descr['i2c'], pos_descr['c2i'] = build_tag_dict(conllu_data, ctag, 1)

    feats_to_use = []
    if morpho_conf is None:
        feats_dict = build_feats_dict(conllu_data)
    else:
        feats_dict = load_feats_dict_from_morpho_config(morpho_conf)
        feats_to_use = list(feats_dict.keys())

    feats_dict['FirstWord'] = {'name_ft': None, 'c2i': {'#None': 0, 'Yes': 1}, 'i2c': ['#None', 'Yes']}
    feats_to_use.append('FirstWord')

    if 0 == len(feats_to_use):
        data_set, feats_to_use = convert_conllu_to_dataset(conllu_data, src_field, dst_field,
                                             src_descr['c2i'], src_descr['max_len'],
                                             dst_descr['c2i'], dst_descr['max_len'],
                                             pos_descr['c2i'], feats_dict, 0, 0) # 3, 3)
    else:
        data_set, _ = convert_conllu_to_dataset(conllu_data, src_field, dst_field,
                                             src_descr['c2i'], src_descr['max_len'],
                                             dst_descr['c2i'], dst_descr['max_len'],
                                             pos_descr['c2i'], feats_dict, 0, 0, feats_to_use)

    print('INFO: trainset size: %d' % len(data_set))

    return data_set, src_descr['max_len'], dst_descr['max_len'], src_descr['c2i'], dst_descr['c2i'], \
           pos_descr['c2i'], pos_descr['i2c'], feats_dict, feats_to_use


def load_feats_dict_from_morpho_config(fn):
    d = {}

    config = json.load(open(fn, 'r'))

    for k in config['output']:
        if '/feat/' not in k:
            continue

        feat_name = k[k.find('/feat/')+6:]
        feat_name_orig = feat_name.replace('-_', '[').replace('_-', ']')
        d[feat_name_orig] = { 'name_tf': feat_name, 'c2i': {} }
        d[feat_name_orig]['i2c'] = config['output'][k]['i2t']
        for i in range(len(d[feat_name_orig]['i2c'])):
            d[feat_name_orig]['c2i'][d[feat_name_orig]['i2c'][i]] = i

    return d

def train_model(options, train_set, dev_set, args):

    m = Model()
    m.build(options)

    #for var in tf.trainable_variables():
    #    sys.stderr.write(var + "\n")
    #sys.exit(0)

    options['input_nodes'] = m.input_nodes
    options['output_nodes'] = m.output_nodes

    with open(args.model_prefix + '.conf', 'w') as f:
        f.write(json.dumps(options, sort_keys=True, indent=2))

    session_config = tf.ConfigProto()
    #session_config.optimizer.set_jit(True)
    session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    session = tf.Session(config=session_config)

    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    iter = 0
    best_acc = 0
    best_acc_iter = 0
    best_train_acc = 0
    lr = 0.01
    decay_step = 1
    #lr_dec = 0.95

    iter_saved = 0
    time_before = time.time()
    while True:

        batch = generate_batch(train_set,
                               options=options,
                               randomize=True,
                               batch_size=options['batch_size'])
                               #max_len=len(train_set[0]['src']))

        _, metrics = session.run([
            m.train_op,
            m.metrics
        ],
        #feed_dict=m.feed_dict(batch, lr, 0.9, 0.9, 0.9))
        feed_dict=m.feed_dict(batch, lr, 0.6, 0.6, 0.6))

        sys.stdout.write('I %d | LOSS %8.6f\r' % (iter, metrics['loss']))

        if iter > 0 and iter % args.iter_len == 0:
            #print('')
            if dev_set is not None:
                duration = time.time() - time_before
                train_accuracy = evaluate_model(train_set, m, session, options, False)
                if train_accuracy > best_train_acc:
                    best_train_acc = train_accuracy

                if train_accuracy < 1:
                    if train_accuracy < best_train_acc:
                        lr = lr * (math.atan(decay_step*2) - 0.58)
                        decay_step += 1

                dev_accuracy = evaluate_model(dev_set, m, session, options, False)
                save_flag = ''
                if dev_accuracy > best_acc:
                    m.save(session, args.model_prefix)
                    iter_saved = iter
                    best_acc = dev_accuracy
                    best_acc_iter = iter
                    #sys.stdout.write("saved at %8.6f\n" % dev_accuracy)
                    save_flag = '*'

                if 1 == train_accuracy:
                    if dev_accuracy < best_acc:
                        lr = lr * (math.atan(decay_step*2) - 0.58)
                        decay_step += 1

                sys.stdout.write('I %d | LR %8.6f | LOSS %8.6f | TRAIN ACC %6.4f (%6.4f) | DEV ACC %6.4f (%6.4f) | %d | %s\n'
                                 % (iter, lr, metrics['loss'], train_accuracy, best_train_acc, dev_accuracy, best_acc, duration, save_flag))

                time_before = time.time()
            else:
                m.save(session, args.model_prefix)
                iter_saved = iter

        if iter - best_acc_iter > 200000:
            break
        if args.num_iters is not None and iter > args.num_iters:
            if iter_saved == 0:
                m.save(session, args.model_prefix)
                iter_saved = iter
            break

        iter += 1


def evaluate_model(data_set, model, session, options, print_details=True):
    counter = {
        'total': 0,
        'correct': 0
    }
    for start in range(0, min(options['batch_size'] * 10, len(data_set)), options['batch_size']):
        batch = generate_batch(data_set,
                               options=options,
                               randomize=True,
                               gold=True,
                               batch_size=options['batch_size'],
                               start=start)

        output = session.run([ model.decoder['output']['infer'] ],
                             feed_dict=model.feed_dict(batch, gold=False))

        #rnn_output, sample_id = output[0].rnn_output.tolist(), output[0].sample_id.tolist()
        rnn_output = output[0].beam_search_decoder_output.scores.tolist()
        sample_id = []
        for j in range(0, 5):
            sample_id.append(output[0].predicted_ids[:,:,j].tolist())
        #return 0

        correct = compare_strings(batch['gold'], sample_id[0], options['decoder']['i2c'], options['decoder']['c2i'][EOS])
        counter['total'] += options['batch_size']
        counter['correct'] += correct

        if start == 0 and print_details:
            for i in range(len(batch['input'])):
                for j in range(0, 1):
                    src_str = get_string_by_id(batch['input'][i], options['encoder']['i2c'], options['encoder']['c2i'][EOS])
                    true_str = get_string_by_id(batch['gold'][i], options['decoder']['i2c'], options['decoder']['c2i'][EOS])
                    dst_str = get_string_by_id(sample_id[j][i], options['decoder']['i2c'], options['decoder']['c2i'][EOS])
                    feat_str = options['encoder']['i2t'][batch['context'][i][0]]
                    for feat_idx in range(len(batch['feats'])):
                        feat_val = batch['feats'][feat_idx][i]
                        if len(feat_str) > 0:
                            feat_str += "+"
                        feat_name = options['encoder']['feats_to_use'][feat_idx]
                        feat_str += options['encoder']['feats'][feat_name]['i2c'][feat_val]

                    mark = ' '
                    if dst_str == true_str:
                        mark = '='

                    print('%22s -> %22s %s G: %22s %s' % (src_str, dst_str, mark, true_str, feat_str))

    accuracy = float(counter['correct']) / counter['total']
    return accuracy


def compare_strings(gold, pred, i2c, EOS):
    correct_words = 0
    for i in range(len(gold)):
        for j in range(len(gold[i])):
            if gold[i][j] == EOS and pred[i][j] == EOS:
                correct_words += 1
                break
            if gold[i][j] == pred[i][j]:
                continue

            break

    return correct_words


def get_string_by_id(ids, i2c, EOS):
    l = []
    for x in ids:
        if x == EOS:
            break
        l.append(i2c[x])

    return ''.join(l)


if __name__ == "__main__":
    main()
