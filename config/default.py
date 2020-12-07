#!/usr/bin/env python3
# -*- coding: utf-8 -*-

options = {
    'batch_size': 50,
    'encoder':
        {
            'ctx_len': 1,
            'ctx_dim': 0, #16,
            'layers':
                [
                    {
                        'dim': 150
                    },
                    {
                        'dim': 150
                    }
                ],
            'embd':
                {
                    'char':
                        {
                            'dim': 200
                        },
                    'tag':
                        {
                            'dim': 6
                        }
                }
        },
    'decoder':
        {
            'layers':
                [
                    {
                        'dim': 300
                    }
                ],
            'embd':
                {
                    'char':
                        {
                            'dim': 100
                        }
                },
            'attention':
                {
                    'dim': 300 #76
                },
            'beam_size': 5
        }
}
