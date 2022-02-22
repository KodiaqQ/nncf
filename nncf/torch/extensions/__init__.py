import enum
from pathlib import Path

import torch

from abc import ABC, abstractmethod

from torch.utils.cpp_extension import _get_build_directory

from nncf.common.utils.registry import Registry

EXTENSIONS = Registry('extensions')


class ExtensionsType(enum.Enum):
    CPU = 0
    CUDA = 1


class ExtensionLoader(ABC):
    @classmethod
    @abstractmethod
    def extension_type(cls):
        pass

    @classmethod
    @abstractmethod
    def load(cls):
        pass

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        pass

    @classmethod
    def get_build_dir(cls) -> str:
        build_dir = Path(_get_build_directory(cls.name(), verbose=False)) / 'nncf' / torch.__version__
        return str(build_dir)


def _force_build_extensions(ext_type: ExtensionsType):
    for class_type in EXTENSIONS.registry_dict.values():
        if class_type.extension_type() != ext_type:
            continue
        class_type.load()


def force_build_cpu_extensions():
    _force_build_extensions(ExtensionsType.CPU)


def force_build_cuda_extensions():
    _force_build_extensions(ExtensionsType.CUDA)


class CudaNotAvailableStub:
    def __getattr__(self, item):
        raise RuntimeError("CUDA is not available on this machine. Check that the machine has a GPU and a proper"
                           "driver supporting CUDA {} is installed".format(torch.version.cuda))
