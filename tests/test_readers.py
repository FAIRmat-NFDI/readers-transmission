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

import os
import json
import glob
import numpy as np
import pint
import pytest

from fairmat_readers_transmission.readers import read_perkin_elmer_asc

ureg = pint.get_application_registry()


def convert_quantity_to_string(data_dict):
    """
    In a dict, recursively convert every pint.Quantity into str containing its shape.

    Args:
        data_dict (dict): A nested dictionary containing pint.Quantity and other data.
    """
    for k, v in data_dict.items():
        if isinstance(v, ureg.Quantity):
            if isinstance(v.magnitude, np.ndarray):
                data_dict[k] = str(v.shape)
            else:
                data_dict[k] = str(v.magnitude)
        if isinstance(v, dict):
            convert_quantity_to_string(v)
        if isinstance(v, list):
            for i in v:
                convert_quantity_to_string(i)


asc_test_files = glob.glob(os.path.join(os.path.dirname(__file__), 'data', '*.asc'))


@pytest.mark.parametrize(
    'asc_file',
    asc_test_files,
)
def test_read_perkin_elmer_asc(asc_file):
    output = read_perkin_elmer_asc(asc_file)
    convert_quantity_to_string(output)
    with open(f'{asc_file.replace(".asc",".json")}', 'r', encoding='utf-8') as f:
        reference = json.load(f)
    assert output == reference
