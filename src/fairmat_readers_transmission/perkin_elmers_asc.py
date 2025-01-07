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
from datetime import datetime
from inspect import isfunction
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import pandas as pd
import pint

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

ureg = pint.get_application_registry()


def read_sample_name(metadata: list, logger: 'BoundLogger') -> str:
    """
    Reads the sample name from the metadata.

    Args:
        metadata (list): The metadata list.
        logger (BoundLogger): A structlog logger.

    Returns:
        str: The sample name.
    """
    if not metadata[2]:
        if logger is not None:
            logger.warning('Sample name not found in the metadata.')
        return None
    return metadata[2].split('.')[0]


def read_start_datetime(metadata: list, logger: 'BoundLogger') -> str:
    """
    Reads the start date from the metadata.

    Args:
        metadata (list): The metadata list.
        logger (BoundLogger): A structlog logger.

    Returns:
        str: The start date.
    """
    if not metadata[3] or not metadata[4]:
        return None
    try:
        century = str(datetime.now().year // 100)
        formated_date = metadata[3].replace('/', '-')
        return f'{century}{formated_date}T{metadata[4]}Z'
    except ValueError as e:
        if logger is not None:
            logger.warning(f'Error in reading the start date.\n{e}')
    return None


def read_is_d2_lamp_used(metadata: list, logger: 'BoundLogger') -> bool:
    """
    Reads whether the D2 lamp was active during the measurement.

    Args:
        metadata (list): The metadata list.
        logger (BoundLogger): A structlog logger.

    Returns:
        bool: Whether the D2 lamp was active during the measurement.
    """
    if not metadata[21]:
        return None
    try:
        return bool(float(metadata[21]))
    except ValueError as e:
        if logger is not None:
            logger.warning(f'Error in reading the D2 lamp data.\n{e}')
    return None


def read_is_tungsten_lamp_used(metadata: list, logger: 'BoundLogger') -> bool:
    """
    Reads whether the tungsten lamp was active during the measurement.

    Args:
        metadata (list): The metadata list.
        logger (BoundLogger): A structlog logger.

    Returns:
        bool: Whether the tungsten lamp was active during the measurement.
    """
    if not metadata[22]:
        return None
    try:
        return bool(float(metadata[22]))
    except ValueError as e:
        if logger is not None:
            logger.warning(f'Error in reading the tungsten lamp data.\n{e}')
    return None


def read_attenuation_percentage(metadata: list, logger) -> Dict[str, int]:
    """
    Reads the sample and reference attenuation percentage from the metadata

    Args:
        metadata (list): The metadata list.
        logger (BoundLogger): A structlog logger.

    Returns:
        Dict[str, int]: The sample and reference attenuation percentage.
    """
    output_dict = {'sample': None, 'reference': None}
    try:
        for attenuation_val in metadata[47].split():
            key, val = attenuation_val.split(':')
            if val == '':
                continue
            if 'S' in key:
                output_dict['sample'] = float(val) * ureg.dimensionless
            elif 'R' in key:
                output_dict['reference'] = float(val) * ureg.dimensionless
    except ValueError as e:
        if logger is not None:
            logger.warning(f'Error in reading the attenuation data.\n{e}')
    return output_dict


def read_is_common_beam_depolarizer_on(metadata: list, logger: 'BoundLogger') -> bool:
    """
    Reads whether the common beam depolarizer was active during the measurement.

    Args:
        metadata (list): The metadata list.
        logger (BoundLogger): A structlog logger.

    Returns:
        bool: Whether the common beam depolarizer was active during the measurement.
    """
    if not metadata[46]:
        return None
    if metadata[46] == 'on':
        return True
    if metadata[46] == 'off':
        return False
    if logger is not None:
        logger.warning('Unexpected value for common beam depolarizer state.')
    return None


def read_long_line(line: str, logger: 'BoundLogger') -> list:
    """
    A long line in the data file contains of a quantity at multiple wavelengths. These
    values are available within one line but separated by whitespaces. The function
    generates a list of wavelength range and value.
    Eg. "3350/2.4 860.8/2.05" will give
    [
        {'wavelength_upper_limit': 860.8, 'wavelength_lower_limit': None,  'value': 2.05},
        {'wavelength_upper_limit': 3350, 'wavelength_lower_limit': 860.8, 'value': 2.4},
    ],

    Args:
        line (str): The line to parse.
        logger (BoundLogger): A structlog logger.

    Returns:
        list: The list of wavelength range and value.
    """

    def try_float(val: str) -> float:
        try:
            return float(val)
        except ValueError:
            return val

    wavelength_value_pairs_list = []
    for key_value_pair in line.split():
        key_value_pair_list = key_value_pair.split('/')
        try:
            if len(key_value_pair_list) == 1:
                wavelength_value_pairs_list.append(
                    {'wavelength': None, 'value': try_float(key_value_pair_list[0])}
                )
            elif len(key_value_pair_list) == 2:  # noqa: PLR2004
                wavelength_value_pairs_list.append(
                    {
                        'wavelength': float(key_value_pair_list[0]) * ureg.nanometer,
                        'value': try_float(key_value_pair_list[1]),
                    }
                )
            else:
                if logger is not None:
                    logger.warning(
                        f'Unexpected value while reading the long line: {line}'
                    )
        except ValueError as e:
            if logger is not None:
                logger.warning(f'Error in reading the long line.\n{e}')

    # convert wavelengths to range of wavelengths
    wavelength_value_pairs_list.sort(key=lambda x: x['wavelength'])
    output_list = []
    for i, wavelength_value_pair in enumerate(wavelength_value_pairs_list):
        range_value_pair = {
            'wavelength_upper_limit': wavelength_value_pair['wavelength'],
            'wavelength_lower_limit': None,
            'value': wavelength_value_pair['value'],
        }
        if i - 1 >= 0:
            range_value_pair['wavelength_lower_limit'] = wavelength_value_pairs_list[
                i - 1
            ]['wavelength']
        output_list.append(range_value_pair)

    return output_list


def read_monochromator_slit_width(metadata: list, logger: 'BoundLogger') -> list:
    """
    Reads the monochromator slit width from the metadata.

    Args:
        metadata (list): The metadata list.
        logger (BoundLogger): A structlog logger.

    Returns:
        list: The monochromator slit width at different wavelengths.
    """
    if not metadata[17]:
        return []
    output_list = read_long_line(metadata[17], logger)
    for i, el in enumerate(output_list):
        if isinstance(el['value'], float):
            output_list[i]['value'] *= ureg.nanometer
    return sorted(output_list, key=lambda x: x['wavelength'])


def read_detector_integration_time(metadata: list, logger: 'BoundLogger') -> list:
    """
    Reads the detector integration time from the metadata.

    Args:
        metadata (list): The metadata list.
        logger (BoundLogger): A structlog logger.

    Returns:
        list: The detector integration time at different wavelengths.
    """
    if not metadata[32]:
        return []
    output_list = read_long_line(metadata[32], logger)
    for i, el in enumerate(output_list):
        if isinstance(el['value'], float):
            output_list[i]['value'] *= ureg.s
    return sorted(output_list, key=lambda x: x['wavelength'])


def read_detector_nir_gain(metadata: list, logger: 'BoundLogger') -> list:
    """
    Reads the detector NIR gain from the metadata.

    Args:
        metadata (list): The metadata list.
        logger (BoundLogger): A structlog logger.

    Returns:
        list: The detector NIR gain at different wavelengths.
    """
    if not metadata[35]:
        return []
    output_list = read_long_line(metadata[35], logger)
    for i, el in enumerate(output_list):
        if isinstance(el['value'], float):
            output_list[i]['value'] *= ureg.dimensionless
    return sorted(output_list, key=lambda x: x['wavelength'])


def read_detector_change_wavelength(metadata: list, logger: 'BoundLogger') -> list:
    """
    Reads the detector change wavelength from the metadata.

    Args:
        metadata (list): The metadata list.
        logger (BoundLogger): A structlog logger.

    Returns:
        list: The detector change wavelengths.
    """
    if not metadata[43]:
        return None
    try:
        return (
            np.array(sorted([float(x) for x in metadata[43].split()])) * ureg.nanometer
        )
    except ValueError as e:
        if logger is not None:
            logger.warning(f'Error in reading the detector change wavelength.\n{e}')
    return None


def read_polarizer_angle(metadata: list, logger: 'BoundLogger') -> float:
    """
    Reads the polarizer angle from the metadata.

    Args:
        metadata (list): The metadata list.
        logger (BoundLogger): A structlog logger.

    Returns:
        list: The polarizer angle.
    """
    if not metadata[48]:
        return None
    try:
        return float(metadata[48]) * ureg.degree
    except ValueError as e:
        if logger is not None:
            logger.warning(f'Error in reading the polarizer angle.\n{e}')
    return None


def read_monochromator_change_wavelength(metadata: list, logger: 'BoundLogger') -> list:
    """
    Reads the monochromator change wavelength from the metadata.

    Args:
        metadata (list): The metadata list.
        logger (BoundLogger): A structlog logger.

    Returns:
        list: The monochromator change wavelengths.
    """
    if not metadata[41]:
        return None
    try:
        return (
            np.array(sorted([float(x) for x in metadata[41].split()])) * ureg.nanometer
        )
    except ValueError as e:
        if logger is not None:
            logger.warning(
                f'Error in reading the monochromator change wavelength.\n{e}'
            )
    return None


def read_lamp_change_wavelength(metadata: list, logger: 'BoundLogger') -> list:
    """
    Reads the lamp change wavelength from the metadata.

    Args:
        metadata (list): The metadata list.
        logger (BoundLogger): A structlog logger.

    Returns:
        list[float]: The lamp change wavelengths.
    """
    if not metadata[42]:
        return None
    try:
        return (
            np.array(sorted([float(x) for x in metadata[42].split()])) * ureg.nanometer
        )
    except ValueError as e:
        if logger is not None:
            logger.warning(f'Error in reading the lamp change wavelength.\n{e}')
    return None


def read_detector_module(metadata: list, logger: 'BoundLogger') -> str:
    """
    Reads the detector module from the metadata

    Args:
        metadata (list): The metadata list.
        logger (BoundLogger): A structlog logger.

    Returns:
        str: The detector module.
    """
    if not metadata[24]:
        return None
    if 'uv/vis/nir detector' in metadata[24].lower():
        return 'uv/vis/nir detector'
    if '150mm sphere' in metadata[24].lower():
        return '150mm sphere'
    if logger is not None:
        logger.warning(
            f'Unexpected detector module found: "{metadata[24]}". Returning None.'
        )
    return None


def read_lamps(metadata: list, logger: 'BoundLogger') -> list:
    """
    Reads the lamps from the metadata.

    Args:
        metadata (list): The metadata list.
        logger (BoundLogger): A structlog logger.

    Returns:
        list: List of dicts containing type, lower and upper wavelength.
    """
    lamps = []
    if read_is_d2_lamp_used(metadata, logger):
        lamps.append(
            {
                'type': 'Deuterium',
                'wavelength_lower_limit': None,
                'wavelength_upper_limit': None,
            }
        )
    if read_is_tungsten_lamp_used(metadata, logger):
        lamps.append(
            {
                'type': 'Tungsten',
                'wavelength_lower_limit': None,
                'wavelength_upper_limit': None,
            }
        )

    lamp_change_points = read_lamp_change_wavelength(metadata, logger)
    if lamp_change_points is not None and len(lamp_change_points) == len(lamps) - 1:
        for idx, lamp_change_point in enumerate(lamp_change_points):
            lamps[idx]['wavelength_upper_limit'] = lamp_change_point
            lamps[idx + 1]['wavelength_lower_limit'] = lamp_change_point


def read_detectors(metadata: list, logger: 'BoundLogger') -> list:
    """
    Reads the detectors from the metadata.

    Args:
        metadata (list): The metadata list.
        logger (BoundLogger): A structlog logger.

    Returns:
        list: List of dicts containing type, lower and upper wavelength.
    """
    detectors = []
    detector_change_points = read_detector_change_wavelength(metadata, logger)
    if detector_change_points is not None:
        for idx, detector_change_point in enumerate(detector_change_points):
            detectors.append(
                {
                    'type': 'UV/VIS',
                    'wavelength_lower_limit': None,
                    'wavelength_upper_limit': None,
                }
            )
            detectors[idx]['wavelength_upper_limit'] = detector_change_point
            if idx + 1 < len(detector_change_points):
                detectors[idx + 1]['wavelength_lower_limit'] = detector_change_point
    return detectors


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

    metadata_map: Dict[str, Any] = {
        'sample_name': read_sample_name,
        'start_datetime': read_start_datetime,
        'analyst_name': 7,
        'instrument_name': 11,
        'instrument_serial_number': 12,
        'instrument_firmware_version': 13,
        'lamps': read_lamps,
        'sample_beam_position': 44,
        'common_beam_mask_percentage': 45,
        'is_common_beam_depolarizer_on': read_is_common_beam_depolarizer_on,
        'attenuation_percentage': read_attenuation_percentage,
        # 'detectors': read_detectors,
        'detector_integration_time': read_detector_integration_time,
        'detector_NIR_gain': read_detector_nir_gain,
        # 'detector_gain': read_detector_gain,
        'detector_change_wavelength': read_detector_change_wavelength,
        'detector_module': read_detector_module,
        'polarizer_angle': read_polarizer_angle,
        'ordinate_type': 80,
        'wavelength_units': 79,
        # 'monochromators': read_monochromators,
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
