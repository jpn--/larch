
from sharrow import Dataset as _sharrow_Dataset

class Dataset(_sharrow_Dataset):

    __slots__ = ()

    @property
    def n_cases(self):
        return self.dims['_caseid_']

    @property
    def n_alts(self):
        if '_altid_' in self.dims:
            return self.dims['_altid_']
        if 'n_alts' in self.attrs:
            return self.attrs['n_alts']
        raise ValueError('no n_alts set')

    def _repr_html_(self):
        html = super()._repr_html_()
        html = html.replace("sharrow.Dataset", "larch.Dataset")
        return html

    def __repr__(self):
        r = super().__repr__()
        r = r.replace("sharrow.Dataset", "larch.Dataset")
        return r
