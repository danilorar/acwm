"""Read numeric/pandas metadata without permitting arbitrary pickle globals."""
import importlib
import pickle

ALLOWED = {
    'builtins': {'slice', 'complex'},
    'numpy': {'dtype', 'ndarray'},
    'numpy.core.multiarray': {'_reconstruct', 'scalar'},
    'numpy._core.multiarray': {'_reconstruct', 'scalar'},
    'numpy.core.numeric': {'_frombuffer'},
    'numpy._core.numeric': {'_frombuffer'},
    'pandas.core.frame': {'DataFrame'},
    'pandas.core.internals.managers': {'BlockManager'},
    'pandas._libs.internals': {'_unpickle_block'},
    'pandas.core.indexes.base': {'_new_Index', 'Index'},
    'pandas.core.indexes.range': {'RangeIndex'},
    'pandas.core.indexes.numeric': {'Int64Index', 'Float64Index'},
}

class NumericUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name not in ALLOWED.get(module,set()):
            raise pickle.UnpicklingError(f'Unapproved pickle global: {module}.{name}')
        # Old pandas numeric indexes were consolidated into Index.
        if module=='pandas.core.indexes.numeric':
            from pandas import Index
            return Index
        return getattr(importlib.import_module(module),name)


def load(path):
    with open(path,'rb') as f:return NumericUnpickler(f).load()
