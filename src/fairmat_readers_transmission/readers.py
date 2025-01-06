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

from typing import TYPE_CHECKING, Any, Dict
from fairmat_readers_transmission.perkin_elmers_asc import read_perkin_elmer_asc

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
