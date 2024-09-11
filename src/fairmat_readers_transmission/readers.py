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

from collections import defaultdict
from inspect import isfunction
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import pandas as pd
import pint

ureg = pint.get_application_registry()

if TYPE_CHECKING:
    from structlog.stdlib import (
        BoundLogger,
    )


def read_file(file_path: str, logger: 'BoundLogger' = None) -> Dict[str, Any]:
    """
    Main function to figure out which specific file format to read.

    Args:
        file_path (str): The path to the file to be read.
        logger (BoundLogger, optional): A structlog logger. Defaults to None.

    Returns:
        Dict[str, Any]: The transmission data in a Python dictionary.
    """
    if file_path.endswith('.asc'):
        return read_perkin_elmer_asc(file_path, logger)
    raise NotImplementedError('Unknown file type.')


def read_perkin_elmer_asc(
    file_path: str, logger: 'BoundLogger' = None
) -> Dict[str, Any]:
    """
    Function for reading the transmission data from PerkinElmer *.asc.

    Args:
        file_path (str): The path to the transmission data file.
        logger (BoundLogger, optional): A structlog logger. Defaults to None.

    Returns:
        Dict[str, Any]: The transmission data and metadata in a Python dictionary.
    """
    from fairmat_readers_transmission.perkin_elmers_asc import (
        read_attenuation_percentage,
        read_detector_change_wavelength,
        read_detector_integration_time,
        read_detector_module,
        read_detector_nir_gain,
        read_is_common_beam_depolarizer_on,
        read_is_d2_lamp_used,
        read_is_tungsten_lamp_used,
        read_lamp_change_wavelength,
        read_monochromator_change_wavelength,
        read_monochromator_slit_width,
        read_polarizer_angle,
        read_sample_name,
        read_start_datetime,
    )

    metadata_map: Dict[str, Any] = {
        'sample_name': read_sample_name,
        'start_datetime': read_start_datetime,
        'analyst_name': 7,
        'instrument_name': 11,
        'instrument_serial_number': 12,
        'instrument_firmware_version': 13,
        'is_d2_lamp_used': read_is_d2_lamp_used,
        'is_tungsten_lamp_used': read_is_tungsten_lamp_used,
        'sample_beam_position': 44,
        'common_beam_mask_percentage': 45,
        'is_common_beam_depolarizer_on': read_is_common_beam_depolarizer_on,
        'attenuation_percentage': read_attenuation_percentage,
        'detector_integration_time': read_detector_integration_time,
        'detector_NIR_gain': read_detector_nir_gain,
        'detector_change_wavelength': read_detector_change_wavelength,
        'detector_module': read_detector_module,
        'polarizer_angle': read_polarizer_angle,
        'ordinate_type': 80,
        'wavelength_units': 79,
        'monochromator_slit_width': read_monochromator_slit_width,
        'monochromator_change_wavelength': read_monochromator_change_wavelength,
        'lamp_change_wavelength': read_lamp_change_wavelength,
    }

    def restructure_measured_data(data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Builds the data entry dict from the data in a pandas dataframe.

        Args:
            data (pd.DataFrame): The dataframe containing the data.

        Returns:
            Dict[str, np.ndarray]: The dict with the measured data.
        """
        output: Dict[str, Any] = {}
        output['measured_wavelength'] = data.index.values
        output['measured_ordinate'] = data.values[:, 0] * ureg.dimensionless

        return output

    output: Dict[str, Any] = defaultdict(lambda: None)
    data_start_ind = '#DATA'

    with open(file_path, encoding='utf-8') as file_obj:
        metadata = []
        for line in file_obj:
            if line.strip() == data_start_ind:
                break
            metadata.append(line.strip())

        data = pd.read_csv(file_obj, sep='\\s+', header=None, index_col=0)

    for path, val in metadata_map.items():
        # If the dict value is an int just get the data with it's index
        if isinstance(val, int):
            if metadata[val]:
                try:
                    output[path] = float(metadata[val]) * ureg.dimensionless
                except ValueError:
                    output[path] = metadata[val]
        elif isfunction(val):
            output[path] = val(metadata, logger)
        else:
            raise ValueError(
                f'Invalid type value {type(val)} of entry "{path}:{val}" in'
                'metadata_map.'
            )

    output.update(restructure_measured_data(data))
    output['measured_wavelength'] *= ureg(output['wavelength_units'])

    return output
