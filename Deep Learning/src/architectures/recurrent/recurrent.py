from typing import Generator, Union

from ...core import Tensor
from ...core.utils.data_processing import concat
from ...core.utils.context_manager import no_grad
from ..auto_regressive import SequentialAutoRegressive


class Recurrent(SequentialAutoRegressive):
    pass