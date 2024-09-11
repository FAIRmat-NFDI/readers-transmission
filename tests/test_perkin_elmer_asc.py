#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import pint

from fairmat_readers_transmission.perkin_elmers_asc import (
    read_attenuation_percentage,
    read_long_line,
    read_sample_name,
    read_start_datetime,
)

ureg = pint.get_application_registry()


@pytest.mark.parametrize(
    'param, expected',
    [
        ([''], {'sample': None, 'reference': None}),
        (['S:100'], {'sample': 100, 'reference': None}),
        (['R:100'], {'sample': None, 'reference': 100}),
        (['S:100 R:100'], {'sample': 100, 'reference': 100}),
        (['S: R:100'], {'sample': None, 'reference': 100}),
        (['3350/S: 3350/R:100'], {'sample': None, 'reference': 100}),
    ],
)
def test_read_attenuation_percentage(param, expected):
    param_list = ['' for i in range(47)]
    param_list.extend(param)
    assert read_attenuation_percentage(param_list, logger=None) == expected


@pytest.mark.parametrize(
    'param, expected',
    [
        ([''], None),
        (['sample'], 'sample'),
        (['sample.txt'], 'sample'),
        (['sample.txt.txt'], 'sample'),
    ],
)
def test_read_sample_name(param, expected):
    param_list = ['' for i in range(2)]
    param_list.extend(param)
    assert read_sample_name(param_list, logger=None) == expected


@pytest.mark.parametrize(
    'param, expected',
    [
        (['', ''], None),
        (['19/06/25', '11:03:40.00'], '2019-06-25T11:03:40.00Z'),
    ],
)
def test_read_start_datetime(param, expected):
    param_list = ['' for i in range(3)]
    param_list.extend(param)
    assert read_start_datetime(param_list, logger=None) == expected


@pytest.mark.parametrize(
    'param, expected',
    [
        (
            '3350/2.4 860.8/2.05',
            [
                {'wavelength': 3350 * ureg.nm, 'value': 2.4},
                {'wavelength': 860.8 * ureg.nm, 'value': 2.05},
            ],
        ),
        (
            '2.4 860.8/2.05',
            [
                {'wavelength': None, 'value': 2.4},
                {'wavelength': 860.8 * ureg.nm, 'value': 2.05},
            ],
        ),
        (
            '2.4',
            [
                {'wavelength': None, 'value': 2.4},
            ],
        ),
    ],
)
def test_read_long_line(param, expected):
    assert read_long_line(param, logger=None) == expected
