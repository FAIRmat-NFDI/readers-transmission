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

from typing import TYPE_CHECKING, Dict
from datetime import datetime
import numpy as np
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
                output_dict['sample'] = int(val) * ureg.dimensionless
            elif 'R' in key:
                output_dict['reference'] = int(val) * ureg.dimensionless
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
    generates a list of wavelength-value pairs.
    Eg. [
            {'wavelength': 3350, 'value': 2.4},
            {'wavelength': 860.8, 'value': 2.05},
        ],

    Args:
        line (str): The line to parse.
        logger (BoundLogger): A structlog logger.

    Returns:
        list: The list of wavelength-value pairs.
    """

    def try_float(val: str) -> float:
        try:
            return float(val)
        except ValueError:
            return val

    output_list = []
    for key_value_pair in line.split():
        key_value_pair_list = key_value_pair.split('/')
        try:
            if len(key_value_pair_list) == 1:
                output_list.append(
                    {'wavelength': None, 'value': try_float(key_value_pair_list[0])}
                )
            elif len(key_value_pair_list) == 2:  # noqa: PLR2004
                output_list.append(
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
    return output_list


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
    return output_list


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
    return output_list


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
        return np.array([float(x) for x in metadata[43].split()]) * ureg.nanometer
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
        return np.array([float(x) for x in metadata[41].split()]) * ureg.nanometer
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
        return np.array([float(x) for x in metadata[42].split()]) * ureg.nanometer
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
