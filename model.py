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
import math
import json

import tensorflow as tf

from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.graph_util import convert_variables_to_constants

from data import GO, EOS


def convert_feature_name(name):
    new_name = name.replace('[', '-_')
    new_name = new_name.replace(']', '_-')
    return new_name


def convert_feature_name_back(name):
    new_name = name.replace('-_', '[')
    new_name = new_name.replace('_-', ']')
    return new_name


class Model:

    def __init__(self):
        self.options = None #options
        self.input = {}
        self.output = {}
        self.embd = []
        self.metrics = {}
        self.input_nodes = {}
        self.output_nodes = {}
        self.helpers = {}
        self.GO = None #options['decoder']['c2i'][GO] #'<GO>'
        self.EOS = None #options['decoder']['c2i'][EOS] #'<EOS>'

    def build(self, options):
        self.options = options
        self.GO = self.options['decoder']['c2i'][GO] #'<GO>'
        self.EOS = self.options['decoder']['c2i'][EOS] #'<EOS>'

        self.input = {
            'length': tf.placeholder(tf.int32,
                                               self.options['batch_size'],
                                               'length'),
            'input': tf.placeholder(tf.int32,
                                              (self.options['batch_size'],
                                               self.options['encoder']['max_len']),
                                              'input'),
            'context': tf.placeholder(tf.int32,
                                              (self.options['batch_size'],
                                               self.options['encoder']['ctx_len']),
                                              'context'),
            'gold': tf.placeholder(tf.int32,
                                             (self.options['batch_size'],
                                              None), #self.options['decoder']['max_len']),
                                             'gold'),
            'gold_input': tf.placeholder(tf.int32,
                                   (self.options['batch_size'],
                                    None),  # self.options['decoder']['max_len']),
                                   'gold_input'),
            'gold_length': tf.placeholder(tf.int32,
                                               self.options['batch_size'],
                                               'gold_length'),
            'keep_prob':
                {
                    'input': tf.placeholder_with_default(1.0,
                                                                   [],
                                                                   'input_keep_prob'),
                    'output': tf.placeholder_with_default(1.0,
                                                                    [],
                                                                    'output_keep_prob'),
                    'state':  tf.placeholder_with_default(1.0,
                                                                    [],
                                                                    'state_keep_prob')
                },
            'lr': tf.placeholder_with_default(0.01, [], 'learning_rate'),

        }

        self.feat_embd = []

        for i in range(len(self.options['encoder']['feats_to_use'])):
            name = 'feat_%s' % convert_feature_name(self.options['encoder']['feats_to_use'][i])
            self.input[name] = tf.placeholder(tf.int32, (self.options['batch_size']), name)
            self.input_nodes[name] = self.input[name].name.split(':')[0]

        self.input_mask = tf.sequence_mask(self.input['length'], dtype=tf.float32, name='input_mask')
        self.output_mask = tf.sequence_mask(self.input['gold_length'], dtype=tf.float32, name='output_mask')

        self.input_nodes['input'] = self.input['input'].name.split(':')[0]
        self.input_nodes['length'] = self.input['length'].name.split(':')[0]
        self.input_nodes['context'] = self.input['context'].name.split(':')[0]

        self.context = self.build_context_embd()
        self.encoder = self.build_encoder()

        sys.stderr.write('SHAPE: self.encoder[\'rnn\'][-1][\'output\'][\'bi\'] = %s\n' % str(self.encoder['rnn'][-1]['output']['bi'].shape))

        self.decoder = self.build_decoder()
        logits = tf.identity(self.decoder['output']['train'].rnn_output, 'logits')
        sys.stderr.write('SHAPE: logits = %s\n' % str(logits.shape))

        seq2seq_loss = tf.contrib.seq2seq.sequence_loss(logits,
                                                        self.input['gold'],
                                                        self.output_mask)

        embd_losses = []
        input_trainable_ids, _ = tf.unique(tf.reshape(self.input['input'], [-1]))
        input_trainable_embd_slice = tf.gather(self.encoder['embd'], input_trainable_ids)
        embd_losses.append(tf.nn.l2_loss(input_trainable_embd_slice) * 0.001)

        input_gold_ids, _ = tf.unique(tf.reshape(self.input['gold_input'], [-1]))
        input_gold_embd_slice = tf.gather(self.decoder['char_embd'], input_gold_ids)
        embd_losses.append(tf.nn.l2_loss(input_gold_embd_slice) * 0.001)

        for i in range(len(self.options['encoder']['feats_to_use'])):
            name = 'feat_%s' % convert_feature_name(self.options['encoder']['feats_to_use'][i])

            ids, _ = tf.unique(tf.reshape(self.input[name], [-1]))
            embd_slice = tf.gather(self.feat_embd[i], ids)
            embd_losses.append(tf.nn.l2_loss(embd_slice) * 0.001)

        self.loss = seq2seq_loss #+ tf.reduce_sum(embd_losses)

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.input['lr'], beta2=0.9).minimize(
            self.loss,
            global_step=tf.train.get_or_create_global_step()
        )
        self.metrics['loss'] = self.loss

        #self.output_nodes['rnn_output'] = self.decoder['output']['infer'].rnn_output.name.split(':')[0]
        #self.output_nodes['sample_id'] = self.decoder['output']['infer'].sample_id.name.split(':')[0]
        self.output_nodes['sample_id'] = self.decoder['output']['infer'].predicted_ids.name.split(':')[0]


    def embd_dim(self, num_values):
        dim = int(math.ceil(math.sqrt(num_values)))
        if dim % 2 == 1:
            dim += 1
        if dim < 2:
            dim = 2
        return dim * 5


    def build_context_embd(self):
        with tf.variable_scope('Context', reuse=tf.AUTO_REUSE):
            tag_embd_def = self.options['encoder']['embd']['tag']
            tag_embd = tf.get_variable('encoder_tag_embeddings',
                                       dtype=tf.float32,
                                       shape=(
                                           tag_embd_def['size'],
                                           tag_embd_def['dim']
                                       ),
                                       initializer=tf.contrib.layers.xavier_initializer()
                                       )

            input_ctx_tag = tf.nn.embedding_lookup(tag_embd, self.input['context'])
            sys.stderr.write('SHAPE: input_ctx_tag = %s\n' % str(input_ctx_tag.shape))
            input_ctx_tag = tf.reshape(input_ctx_tag, [ input_ctx_tag.shape[0], -1 ])
            sys.stderr.write('SHAPE: input_ctx_tag = %s\n' % str(input_ctx_tag.shape))

            input_feat = []
            feat_def = self.options['encoder']['feats']
            feats_to_use = self.options['encoder']['feats_to_use']
            for i in range(len(feats_to_use)):
                s = len(feat_def[feats_to_use[i]]['i2c'])
                dim = int(math.ceil(math.sqrt(math.sqrt(s))))
                if dim % 2 == 1:
                    dim += 1
                if dim < 2:
                    dim = 2
                dim = 4
                feat_embd = tf.get_variable('encoder_feat_%s_embeddings' % convert_feature_name(feats_to_use[i]),
                                       dtype=tf.float32,
                                       shape=(
                                           len(feat_def[feats_to_use[i]]['i2c']),
                                           tag_embd_def['dim'] # dim
                                       ),
                                       initializer=tf.contrib.layers.xavier_initializer())
                self.feat_embd.append(feat_embd)

                input_feat.append(tf.nn.embedding_lookup(feat_embd, self.input['feat_%s' % convert_feature_name(feats_to_use[i])]))

            input_ctx_tag = tf.concat([input_ctx_tag] + input_feat, -1)
            sys.stderr.write('SHAPE: input_ctx_tag = %s\n' % str(input_ctx_tag.shape))

            #input_ctx_tag = tf.layers.dense(input_ctx_tag,
            #                               self.options['encoder']['ctx_dim'],
            #                               #activation=tf.nn.relu,
            #                               kernel_initializer=tf.contrib.layers.xavier_initializer()
            #                               )

            sys.stderr.write('SHAPE: input_ctx_tag = %s\n' % str(input_ctx_tag.shape))

            return input_ctx_tag


    def build_encoder_rnn_cell(self, units, name, input_size):
        cell = tf.nn.rnn_cell.DropoutWrapper(
            tf.contrib.rnn.LSTMBlockCell(
                units,
                name=name
            ),
            input_keep_prob=self.input['keep_prob']['input'],
            output_keep_prob=self.input['keep_prob']['output'],
            state_keep_prob=self.input['keep_prob']['state'],
            input_size=input_size,
            dtype=tf.float32,
            variational_recurrent=True
        )

        return cell


    def build_rnn(self, input):
        all_outputs = []

        encoder_options = self.options['encoder']

        for i in range(len(encoder_options['layers'])):
            layer = encoder_options['layers'][i]

            if i == 0:
                this_layer_input = input
            else:
                this_layer_input = all_outputs[-1]['output']['bi']

            fw_cell = self.build_encoder_rnn_cell(layer['dim'], 'layer_%d_fw' % i, this_layer_input.shape[-1])
            bw_cell = self.build_encoder_rnn_cell(layer['dim'], 'layer_%d_bw' % i, this_layer_input.shape[-1])

            if True: #i == 0:
                initial_state_fw = LSTMStateTuple(
                    tf.layers.dense(self.context, layer['dim'], kernel_initializer=tf.contrib.layers.xavier_initializer()),
                    tf.layers.dense(self.context, layer['dim'], kernel_initializer=tf.contrib.layers.xavier_initializer())
                )

                initial_state_bw = LSTMStateTuple(
                    tf.layers.dense(self.context, layer['dim'], kernel_initializer=tf.contrib.layers.xavier_initializer()),
                    tf.layers.dense(self.context, layer['dim'], kernel_initializer=tf.contrib.layers.xavier_initializer())
                )

                outputs, state = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                                 bw_cell,
                                                                 this_layer_input,
                                                                 self.input['length'],
                                                                 initial_state_fw=initial_state_fw,
                                                                 initial_state_bw=initial_state_bw,
                                                                 dtype=tf.float32
                                                                )
            else:
                outputs, state = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                                 bw_cell,
                                                                 this_layer_input,
                                                                 self.input['length'],
                                                                 dtype=tf.float32
                                                                )

            all_outputs.append(
                {
                    'state':
                        {
                            'fw': state[0],
                            'bw': state[1]
                        },
                    'output':
                        {
                            'fw': outputs[0],
                            'bw': outputs[1],
                            'bi': tf.concat(outputs, axis=-1)
                        }
                }
            )

        return all_outputs


    def build_decoder(self):
        decoder_options = self.options['decoder']
        encoder_rnn = self.encoder['rnn']

        if True:
            sys.stderr.write('Output size = %s\n' % str(encoder_rnn[-1]['output']['bi'].shape))
            sys.stderr.write('self.context size = %s\n' % str(self.context.shape))
            expanded_context = tf.layers.dense(self.context,
                                               encoder_rnn[-1]['output']['bi'].shape[2],
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
            sys.stderr.write('expanded_context size = %s\n' % str(expanded_context.shape))
            expanded_context = tf.expand_dims(expanded_context, 1)
            sys.stderr.write('expanded_context size = %s\n' % str(expanded_context.shape))
            the_memory = tf.concat([encoder_rnn[-1]['output']['bi'], expanded_context], axis=1)
            sys.stderr.write('the_memory size = %s\n' % str(the_memory.shape))
            the_memory_length = tf.add(self.input['length'], tf.constant(1, shape=self.input['length'].shape))
        else:
            the_memory = encoder_rnn[-1]['output']['bi']
            the_memory_length = self.input['length']

        with tf.variable_scope('Decoder'):
            with tf.variable_scope('shared_attention_mechanism'):
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=decoder_options['attention']['dim'],
                    #memory=tf.contrib.seq2seq.tile_batch(encoder_rnn[-1]['output']['bi'], decoder_options['beam_size']),
                    memory=tf.contrib.seq2seq.tile_batch(the_memory, decoder_options['beam_size']),
                    #memory_sequence_length=tf.contrib.seq2seq.tile_batch(self.input['length'], decoder_options['beam_size'])
                    memory_sequence_length=tf.contrib.seq2seq.tile_batch(the_memory_length,
                                                                         decoder_options['beam_size'])
                )

                #attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                #    num_units=decoder_options['attention']['dim'],
                #    memory=tf.contrib.seq2seq.tile_batch(encoder_rnn[-1]['output']['bi'], decoder_options['beam_size']),
                #    memory_sequence_length=tf.contrib.seq2seq.tile_batch(self.input['length'], decoder_options['beam_size']))

            embd_def = self.options['decoder']['embd']['char']

            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                tf.nn.rnn_cell.LSTMCell(decoder_options['layers'][0]['dim'] + self.options['encoder']['ctx_dim']),
                attention_mechanism
            )
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attention_cell, embd_def['size']
            )

            char_embd = tf.get_variable('decoder_char_embeddings',
                                                  dtype=tf.float32,
                                                  shape=(
                                                      embd_def['size'],
                                                      #self.embd_dim(embd_def['size'])
                                                      embd_def['dim']
                                                  ),
                                                  initializer=tf.contrib.layers.xavier_initializer())

            self.helpers['train'] = tf.contrib.seq2seq.TrainingHelper(
                inputs=tf.nn.embedding_lookup(char_embd, self.input['gold_input']), #self.input['input'],
                sequence_length=self.input['gold_length'])

            self.helpers['infer'] = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=char_embd,
                start_tokens=tf.tile([self.GO], [self.options['batch_size']]),
                end_token=self.EOS)

            sys.stderr.write('SHAPE: encoder_rnn[-1][\'state\'][\'fw\'].c = %s\n' % str(encoder_rnn[-1]['state']['fw'].c.shape))

            encoder_state_c = tf.concat([encoder_rnn[-1]['state']['fw'].c,
                                         encoder_rnn[-1]['state']['bw'].c#,
                                         #self.context
                                         ], 1)
            encoder_state_h = tf.concat([encoder_rnn[-1]['state']['fw'].h,
                                         encoder_rnn[-1]['state']['bw'].h#,
                                         #self.context
                                         ], 1)
            initial_state = LSTMStateTuple(encoder_state_c, encoder_state_h)

            #output_layer = tf.layers.Dense(embd_def['size'], name="output_projection")

            if False:
                decoder = {
                    'train': tf.contrib.seq2seq.BasicDecoder(
                        cell=out_cell, #attention_cell,
                        helper=self.helpers['train'],
                        initial_state=attention_cell.zero_state(self.options['batch_size'], tf.float32).clone(cell_state=initial_state)),
                    'infer': tf.contrib.seq2seq.BasicDecoder(
                        cell=out_cell, #attention_cell,
                        helper=self.helpers['infer'],
                        initial_state=attention_cell.zero_state(self.options['batch_size'], tf.float32).clone(cell_state=initial_state))
                }

                outputs_train, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder['train'],
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=decoder_options['max_len'])

                outputs_infer, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder['infer'],
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=decoder_options['max_len'])
            else:
                enc_state = tf.contrib.seq2seq.tile_batch(initial_state, decoder_options['beam_size'])
                decoder_initial_state = attention_cell.zero_state(
                    self.options['batch_size'] * decoder_options['beam_size'],
                    tf.float32).clone(cell_state=enc_state)
                #sys.stderr.write('SHAPE: decoder_initial_state = %s\n' % str(decoder_initial_state.shape))

                with tf.variable_scope('shared_attention_mechanism', reuse=True):
                    attention_mechanism_ = tf.contrib.seq2seq.BahdanauAttention(
                        num_units=decoder_options['attention']['dim'],
                        memory=the_memory, #encoder_rnn[-1]['output']['bi'],
                        memory_sequence_length=the_memory_length) #self.input['length'])

                    #attention_mechanism_ = tf.contrib.seq2seq.LuongAttention(
                    #    num_units=decoder_options['attention']['dim'],
                    #    memory=encoder_rnn[-1]['output']['bi'],
                    #    memory_sequence_length=self.input['length'])

                attention_cell_ = tf.contrib.seq2seq.AttentionWrapper(
                    tf.nn.rnn_cell.LSTMCell(decoder_options['layers'][0]['dim'] + self.options['encoder']['ctx_dim']),
                    attention_mechanism_
                )
                out_cell_ = tf.contrib.rnn.OutputProjectionWrapper(
                    attention_cell_, embd_def['size']
                )

                decoder = {}
                decoder['train'] = tf.contrib.seq2seq.BasicDecoder(
                        cell=out_cell_, #attention_cell,
                        helper=self.helpers['train'],
                        initial_state=attention_cell_.zero_state(
                            self.options['batch_size'], tf.float32).clone(cell_state=initial_state))
                decoder['infer'] = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=out_cell,
                    embedding=char_embd,
                    start_tokens=tf.tile([self.GO], [self.options['batch_size']]),
                    end_token=self.EOS,
                    initial_state=attention_cell.zero_state(
                            self.options['batch_size'] * decoder_options['beam_size'], tf.float32).clone(cell_state=enc_state),
                    beam_width=decoder_options['beam_size'])

                with tf.variable_scope('decoder_with_shared_attention_mechanism'):
                    outputs_train, _, _ = tf.contrib.seq2seq.dynamic_decode(
                        decoder = decoder['train'],
                        output_time_major = False,
                        impute_finished = True,
                        maximum_iterations = decoder_options['max_len'],
                        swap_memory=True)

                with tf.variable_scope('decoder_with_shared_attention_mechanism', reuse=True):
                    outputs_infer, _, _ = tf.contrib.seq2seq.dynamic_decode(
                        decoder = decoder['infer'],
                        output_time_major = False,
                        impute_finished = False,
                        maximum_iterations = decoder_options['max_len'],
                        swap_memory=True)


            return {
                'char_embd': char_embd,
                'output':
                    {
                        'train': outputs_train,
                        'infer': outputs_infer
                    },
                'decoder': decoder
            }


    def build_encoder(self):
        with tf.variable_scope('Encoder'):
            embd_def = self.options['encoder']['embd']['char']
            char_embd = tf.get_variable('encoder_char_embeddings',
                                                  dtype=tf.float32,
                                                  shape=(
                                                      embd_def['size'],
                                                      #self.embd_dim(embd_def['size'])
                                                      embd_def['dim']
                                                  ),
                                                  initializer=tf.contrib.layers.xavier_initializer())

            rnn_input = tf.nn.embedding_lookup(char_embd, self.input['input'])

            return { 'rnn': self.build_rnn(rnn_input),
                     'embd': char_embd }


    def get_config(self):
        return {
            'input_nodes': self.input_nodes,
            'output_nodes': self.output_nodes
        }


    def feed_dict(self, batch, lr=0.01, input_keep_prob=1, output_keep_prob=1, state_keep_prob=1, gold=True):
        fd = {
            self.input['input']: batch['input'],
            self.input['length']: batch['length'],
            self.input['context']: batch['context']
        }

        if 'lr' in self.input:
            fd[self.input['lr']] = lr

        for i in range(len(self.options['encoder']['feats_to_use'])):
            name = 'feat_%s' % convert_feature_name(self.options['encoder']['feats_to_use'][i])
            fd[self.input[name]] = batch['feats'][i]

        if 'keep_prob' in self.input:
            if 'input' in self.input['keep_prob']:
                fd[self.input['keep_prob']['input']] = input_keep_prob

            if 'output' in self.input['keep_prob']:
                fd[self.input['keep_prob']['output']] = output_keep_prob

            if 'state' in self.input['keep_prob']:
                fd[self.input['keep_prob']['state']] = state_keep_prob

        if 'gold' in batch and gold:
            fd[self.input['gold']] = batch['gold']
            fd[self.input['gold_input']] = batch['gold_input']
            fd[self.input['gold_length']] = batch['gold_length']

        return fd


    def save(self, sess, prefix):
        graph_def = convert_variables_to_constants(sess,
                                                   tf.get_default_graph().as_graph_def(),
                                                   list(self.output_nodes.values()))

        with tf.gfile.GFile('%s.model' % prefix, "wb") as f:
            f.write(graph_def.SerializeToString())


    def load_config(self, prefix):
        self.config = json.load(open('%s.conf' % prefix, 'r'))

        self.input_nodes = self.config['input_nodes']
        self.output_nodes = self.config['output_nodes']
        self.options = self.config


    def load(self, prefix):
        self.load_config(prefix)

        with tf.gfile.GFile('%s.model' % prefix, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, name='')

        for k in self.input_nodes.keys():
            val = self.input_nodes[k] + ':0'
            self.input[k] = tf.get_default_graph().get_tensor_by_name(val)

        for k in self.output_nodes.keys():
            val = self.output_nodes[k] + ':0'
            self.output[k] = tf.get_default_graph().get_tensor_by_name(val)

