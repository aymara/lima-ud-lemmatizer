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
import re

from collections import OrderedDict


UNK = '<UNK>'
GO = '<GO>'
EOS = '<EOS>'
PAD = '<PAD>'

def is_word(token):
    if isinstance(token['id'], int) or re.fullmatch(r'[0-9]+', str(token['id'])):
        return True
    if isinstance(token['id'], tuple):
        return False
    else:
        sys.stderr.write('ERROR: strange token\n')
        raise


def is_garbage(s):
    if re.fullmatch(r'<?http.?://.*', s, re.IGNORECASE):
        return True
    if re.fullmatch(r'[a-z]+[a-z\.\-]+[a-z][a-z]\/[a-z0-9]+[a-z\.\-0-9\/]+[a-z0-9]+', s, re.IGNORECASE):
        return True
    if re.fullmatch(r'[a-z0-9\-\_\.]+\.com(\/[a-z0-9\-\_\.\/]+)*', s, re.IGNORECASE):
        return True
    if re.fullmatch(r'[^@]+@[^@]+', s, re.IGNORECASE):
        return True
    if len(s) > 10 and re.fullmatch(r'[0-9\-\.\,\*\+\_\!\?\@\#\$\%\^\&\(\)\[\]\{\}\\\/\'\"\`\~\=\>\<\|]+', s, re.IGNORECASE):
        return True
    if re.fullmatch(r'[a-z]+:.+', s, re.IGNORECASE):
        return True
    if re.match(r'^System\\\\', s, re.IGNORECASE):
        return True
    return False


def build_dict(sentences, field, min_ipm, extra=[]):
    chars = {}
    total = 0
    max_len = 0

    for sent in sentences:
        for tok in sent:
            if not is_word(tok):
                continue

            raw = tok[field] #.lower()

            if len(raw) > max_len and not is_garbage(raw):
                max_len = len(raw)

            for c in raw:
                if c not in chars:
                    chars[c] = 0
                chars[c] += 1
                total += 1

    i2c = [] + extra
    c2i = {}
    for i in range(len(i2c)):
        c2i[i2c[i]] = i

    for c in sorted(chars):
        ipm = (float(chars[c]) * 1000000.0) / total
        if ipm >= min_ipm:
            i2c.append(c)
            c2i[c] = len(i2c) - 1

    return i2c, c2i, max_len


def build_tag_dict(sentences, field, min_ipm=1):
    tags = {}

    for sent in sentences:
        for tok in sent:
            if not is_word(tok):
                continue

            raw = tok[field]
            if raw == '_':
                continue

            if raw not in tags:
                tags[raw] = 0
            tags[raw] += 1

    i2c = [ PAD, 'INTJ', 'SYM' ] # INTJ is missing in German-GSD (train) but present in German-GSD (dev)
    c2i = {}
    for i in range(len(i2c)):
        c2i[i2c[i]] = i
    total = sum([ tags[x] for x in tags.keys() ])

    for t in sorted(tags):
        #ipm = (float(tags[t]) * 1000000.0) / total
        #if ipm >= min_ipm:
        if t not in i2c:
            i2c.append(t)
            c2i[t] = len(i2c) - 1

    return i2c, c2i


def build_feats_dict(sentences):
    d = {}

    for sent in sentences:
        for tok in sent:
            if tok['feats'] is None:
                continue
            for feat in tok['feats']:
                if feat not in d:
                    d[feat] = { 'c2i': { '#None': 0 }, 'i2c': [ '#None' ] }
                value = tok['feats'][feat]
                if value not in d[feat]['c2i']:
                    d[feat]['i2c'].append(value)
                    d[feat]['c2i'][value] = len(d[feat]['i2c']) - 1

    return d


def str2idx(s, d, l, prefix=[], back_padding=0):
    idx = [d[EOS]] * l

    i = 0
    while i < len(prefix):
        idx[i] = d[prefix[i]]
        i += 1

    offset = len(prefix)
    while i - offset < len(s) and i < l - back_padding:
        c = s[i - offset]
        if c in d:
            idx[i] = d[c]
        else:
            idx[i] = d[UNK]
        i += 1

    return idx


def generate_key(s, tok):
    return  s + '_' + tok['upostag']


def feats_to_string(feats):
    if feats is None:
        return ''
    s = '|'.join([ '%s=%s' % (k, feats[k]) for k in feats ])
    return s


def get_common_feats(l):
    feats = {}
    for s in l:
        for f in s.split('|'):
            if f not in feats:
                feats[f] = 0
            feats[f] += 1

    common = []
    for f in feats:
        if feats[f] == len(l) and len(f) > 0:
            common.append(f)

    return common


def get_diff_feats(l):
    all_features_expressed = []
    for a in l:
        for f in a:
            k, v = f.split('=')
            if k not in all_features_expressed:
                all_features_expressed.append(k)

    all_features_expressed.sort()

    counter = {}
    for a in l:
        features = { k: '#None' for k in all_features_expressed }
        for f in a:
            k, v = f.split('=')
            features[k] = v

        for k in features:
            v = features[k]
            if k not in counter:
                counter[k] = {}
            if v not in counter[k]:
                counter[k][v] = 0
            counter[k][v] += 1

    diff_feats = []

    categories = sorted(list(counter.keys()), key=lambda x: len(counter[x].keys()), reverse=True)
    for f in categories:
        if len(counter[f].keys()) > 1: #len(l):
            diff_feats.append(f)
        #    continue
        #else: #if len(counter[f].keys()) == 1:
        #    continue
        # if len(diff_feats) == 0:
        #     diff_feats.append(f)
        # else:
        #     sys.stderr.write('bu')

    return diff_feats


def find_frequent_and_uniq(sentences, src, dst):
    temp = {}
    counter = {}

    for sent in sentences:
        for tok in sent:
            if not is_word(tok):
                continue
            if src not in tok or tok[src] is None or tok[src] == '_':
                #sys.stderr.write('WARNING: tok[src] == %s\n' % (str(tok[src])))
                #raise
                continue
            if dst not in tok or tok[dst] is None or tok[dst] == '_':
                #sys.stderr.write('WARNING: tok[dst] == %s\n' % (str(tok[dst])))
                #raise
                continue
            if tok['upostag'] == 'X' or tok['upostag'] == 'PUNCT' or tok['upostag'] == 'NUM':
                continue

            i = tok[src].lower()
            o = tok[dst].lower()

            k = generate_key(i, tok)

            if k not in temp:
                temp[k] = {}

            if o not in temp[k]:
                temp[k][o] = {}

            feats = feats_to_string(tok['feats'])

            if feats not in temp[k][o]:
                temp[k][o][feats] = 0

            temp[k][o][feats] += 1

            if k not in counter:
                counter[k] = 0

            counter[k] += 1

    diff_feats_stat = {}

    for k in temp:
        if len(temp[k]) > 1:
            total = sum([ sum(d.values()) for d in [ temp[k][x] for x in temp[k] ] ])
            to_delete = []
            for v in temp[k]:
                freq = sum([ temp[k][v][x] for x in temp[k][v] ])
                rel_freq = float(freq) / total
                if rel_freq < 0.01:
                    sys.stderr.write('Removing \'%s\'->\'%s\' as non frequent (%f)\n' % (k, v, rel_freq))
                    to_delete.append(v)

            for v in to_delete:
                del temp[k][v]

            if len(k.split('_')[0]) == 1:
                continue
                # Very short words are not reliable

            dst2feats = {}
            for v in temp[k]:
                common_feats = get_common_feats(list(temp[k][v].keys()))
                dst2feats[v] = common_feats
                pass

            diff_feats = get_diff_feats(list(dst2feats.values()))

            if len(diff_feats) > 0:
                upostag = k.split('_')[1]
                if upostag not in diff_feats_stat:
                    diff_feats_stat[upostag] = {}
                for category in diff_feats:
                    if category not in diff_feats_stat[upostag]:
                        diff_feats_stat[upostag][category] = 0
                    diff_feats_stat[upostag][category] += 1

    for tag in diff_feats_stat:
        total = sum([ diff_feats_stat[tag][x] for x in diff_feats_stat[tag] ])

        to_delete = []
        for c in diff_feats_stat[tag]:
            freq = diff_feats_stat[tag][c]
            rel_freq = float(freq) / total
            if rel_freq < 0.02 or freq == 1:
                sys.stderr.write('Removing \'%s\' for \'%s\' as non frequent (%f)\n' % (c, tag, rel_freq))
                to_delete.append(c)

        for c in to_delete:
            del diff_feats_stat[tag][c]

    feats_to_use = []
    for tag in diff_feats_stat:
        if len(diff_feats_stat[tag].keys()) == 0:
            continue
        for feat in diff_feats_stat[tag]:
            if feat not in feats_to_use and feat not in ['Foreign']:
                feats_to_use.append(feat)

    feats_to_use.sort()

    uniq = []
    for k in temp:
        if len(temp[k]) == 1:
            uniq.append(k)
        else:
            #sys.stderr.write('amb: %s\n' % k)
            pass

    uniq.sort(key=lambda x: counter[x], reverse=True)

    top = uniq[:int(len(uniq)/100)]
    #last_freq = counter[top[-1]]

    sys.stderr.write('feats_to_use == %s\n' % str(feats_to_use))

    return top, feats_to_use


def get_feats_list(tok, feats_to_use, feats_dict):
    rv_str = ['#None'] * len(feats_to_use)

    if tok['feats'] is not None:
        for i in range(len(feats_to_use)):
            feat = feats_to_use[i]
            if feat in tok['feats']:
                rv_str[i] = tok['feats'][feat]

    rv = []
    for i in range(len(feats_to_use)):
        feat = feats_to_use[i]
        if rv_str[i] not in feats_dict[feat]['c2i']:
            sys.stderr.write('ERROR: unknown feature ("%s") value: \"%s\"\n' % (feat, rv_str[i]))
            raise
        rv.append(feats_dict[feat]['c2i'][rv_str[i]])

    return rv_str, rv


def convert_conllu_to_dataset(sentences, src, dst, src_dict, src_len, dst_dict, dst_len, pos_dict, feats_dict, lctx, rctx, feats_to_use=[]):
    ds = []
    known_src = {}
    pad_id = pos_dict[PAD]
    pos_i2c = sorted(list(pos_dict.keys()), key=lambda x: pos_dict[x])
    convert_to_lower_case = False

    frequent_and_uniq = []
    if len(feats_to_use) == 0:
        frequent_and_uniq, feats_to_use = find_frequent_and_uniq(sentences, src, dst)

    #feats_to_use = [ 'Animacy', 'Case', 'Degree', 'Gender', 'Number', 'Aspect', 'Mood', 'Tense', 'VerbForm', 'Voice', 'Person', 'FirstWord' ]
    sys.stderr.write('feats_to_use == %s\n' % str(feats_to_use))

    for sent in sentences:
        tags = [ pad_id ] * lctx

        tokens_only = []
        idx = 0
        for tok in sent:
            if idx == 0:
                if tok['feats'] is None:
                    tok['feats'] = OrderedDict()
                tok['feats']['FirstWord'] = 'Yes'
            #else:
            #    tok['feats']['FirstWord'] = '#None'
            idx += 1

            if not is_word(tok):
                continue
            if src not in tok or tok[src] is None or tok[src] == '_':
                #sys.stderr.write('WARNING: tok[src] == %s\n' % (str(tok[src])))
                #raise
                continue
            if dst not in tok or tok[dst] is None or tok[dst] == '_':
                #sys.stderr.write('WARNING: tok[dst] == %s\n' % (str(tok[dst])))
                #raise
                continue

            tokens_only.append(tok)

        tags.extend([ pos_dict[t['upostag']] for t in tokens_only ])
        tags.extend([ pad_id ] * rctx)

        for idx in range(len(tokens_only)):
            tok = tokens_only[idx]

            if tok['upostag'] == 'X' or tok['upostag'] == 'PUNCT' or tok['upostag'] == 'NUM':
                continue

            if convert_to_lower_case:
                i = tok[src].lower()
            else:
                i = tok[src]

            k = generate_key(i, tok) # + "_" + tok['upostag']
            #if k in frequent_and_uniq:
            #    continue

            if 'feats' in tok and tok['feats'] is not None and 'Abbr' in tok['feats'] and tok['feats']['Abbr'] in ['Yes', 'yes']:
                if convert_to_lower_case:
                    o = tok[src].lower()
                else:
                    o = tok[src]
            else:
                if convert_to_lower_case:
                    o = tok[dst].lower()
                else:
                    o = tok[dst]

            if o == 'null' or o == 'NULL':
                continue

            ctx = tags[idx:idx + lctx + rctx + 1]
            if len(ctx) != lctx + rctx + 1:
                raise

            feats_str, feats = get_feats_list(tok, feats_to_use, feats_dict)

            ctx_str = '_'.join([ pos_i2c[x] for x in ctx ])
            item_id = '_'.join([ i, ctx_str ] + feats_str)

            if item_id in known_src:
                if o not in known_src[item_id]:
                    #sys.stderr.write('WARNING: new dst for src \"%s\" %s: \"%s\" (old: %s)\n' % (i, item_id, o, json.dumps(known_src[item_id], ensure_ascii=False)))
                    known_src[item_id][o] = 1
                else:
                    known_src[item_id][o] += 1
                    continue
            else:
                known_src[item_id] = {o: 1}

            ds.append({
                'raw':
                    {
                        'src': i,
                        'dst': o,
                        'ctx': ctx_str,
                        'feats': feats_str
                    },
                'src': str2idx(i, src_dict, src_len),
                'dst': str2idx(o, dst_dict, dst_len, [], 1),
                'ctx': ctx,
                'feats': feats,
                'id': item_id
            })

    return ds, feats_to_use


def convert_conllu_to_dataset_for_prediction(sentences, src, src_dict, src_len, pos_dict, feats_dict, lctx, rctx, feats_to_use):
    ds = {}
    known_src = {}
    pad_id = pos_dict[PAD]
    pos_i2c = sorted(list(pos_dict.keys()), key=lambda x: pos_dict[x])
    convert_to_lower_case = False

    if len(feats_to_use) == 0:
        raise

    for sent_idx in range(len(sentences)):
        sent = sentences[sent_idx]
        tags = [ pad_id ] * lctx

        tokens_only = []
        idx = 0
        for tok in sent:
            if idx == 0:
                if tok['feats'] is None:
                    tok['feats'] = OrderedDict()
                tok['feats']['FirstWord'] = 'Yes'

            idx += 1

            if not is_word(tok):
                continue
            if src not in tok or tok[src] is None or tok[src] == '_':
                continue

            tokens_only.append({'token': tok, 'tok_idx': idx-1, 'sent_idx': sent_idx})

        tags.extend([ pos_dict[t['token']['upostag']] for t in tokens_only ])
        tags.extend([ pad_id ] * rctx)

        for idx in range(len(tokens_only)):
            tok = tokens_only[idx]['token']

            if tok['upostag'] == 'X' or tok['upostag'] == 'PUNCT' or tok['upostag'] == 'NUM':
                continue

            if convert_to_lower_case:
                i = tok[src].lower()
            else:
                i = tok[src]

            k = generate_key(i, tok)

            ctx = tags[idx:idx + lctx + rctx + 1]
            if len(ctx) != lctx + rctx + 1:
                raise

            feats_str, feats = get_feats_list(tok, feats_to_use, feats_dict)

            ctx_str = '_'.join([ pos_i2c[x] for x in ctx ])
            item_id = '_'.join([ i, ctx_str ] + feats_str)

            if len(i) not in ds:
                ds[len(i)] = {}
            if item_id not in ds[len(i)]:
                ds[len(i)][item_id] = {
                    'raw':
                        {
                            'src': i,
                            'ctx': ctx_str,
                            'feats': feats_str
                        },
                    'src': str2idx(i, src_dict, src_len),
                    'ctx': ctx,
                    'feats': feats,
                    'id': item_id,
                    'tokens': [ { 'tok_idx': tokens_only[idx]['tok_idx'], 'sent_idx': tokens_only[idx]['sent_idx'] } ]
                }
            else:
                ds[len(i)][item_id]['tokens'].append({ 'tok_idx': tokens_only[idx]['tok_idx'], 'sent_idx': tokens_only[idx]['sent_idx'] })

    return ds, feats_to_use