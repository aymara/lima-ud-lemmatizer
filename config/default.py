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
