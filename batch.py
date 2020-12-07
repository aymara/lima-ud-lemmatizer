#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import sample

from data import EOS, GO


def generate_batch(data_set,
                   batch_size,
                   options,
                   gold=True,
                   randomize=True,
                   start=0):

    if randomize:
        items = sample(range(len(data_set)), k=batch_size)
    else:
        items = list(range(start, start + batch_size))

    batch = {
        'input': [],
        'length': [],
        'context': [],
        'feats': [ [] for _ in data_set[0]['feats'] ]
    }

    if gold:
        batch['gold'] = []
        batch['gold_input'] = []
        batch['gold_length'] = []

    max_gold_len = 0
    for idx in items:
        if idx >= len(data_set):
            break
        item = data_set[idx]
        batch['input'].append(item['src'])
        batch['length'].append(min(len(item['raw']['src']), options['encoder']['max_len']))
        batch['context'].append(item['ctx'])
        for i in range(len(item['feats'])):
            batch['feats'][i].append(item['feats'][i])
        if gold:
            gold_ = []
            gold_input_ = [options['decoder']['c2i'][GO]]
            for idx in item['dst']:
                if idx == options['decoder']['c2i'][EOS]:
                    break
                gold_.append(idx)
                gold_input_.append(idx)

            batch['gold'].append(gold_)
            batch['gold_input'].append(gold_input_)
            batch['gold_length'].append(len(gold_) + 1)
            if len(gold_) > max_gold_len:
                max_gold_len = len(gold_)

    if gold:
        for l in batch['gold']:
            while len(l) < max_gold_len + 1:
                l.append(options['decoder']['c2i'][EOS])

        for l in batch['gold_input']:
            while len(l) < max_gold_len + 2:
                l.append(options['decoder']['c2i'][EOS])

    if len(batch['input']) < batch_size:
        batch = add_padding_to_batch(batch, batch_size, options)

    return batch


def add_padding_to_batch(batch, batch_size, options):
    for idx in range(len(batch['input']), batch_size):
        batch['input'].append([ options['encoder']['c2i'][EOS] ] * options['encoder']['max_len'])
        batch['length'].append(0)
        batch['context'].append([0])
        for feat in batch['feats']:
            feat.append(0)

    return batch