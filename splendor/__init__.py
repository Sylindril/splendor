"""Splendor: a fast multi-agent PufferLib environment."""

__all__ = ['Splendor']


def __getattr__(name):
    # Lazy so that `splendor.layout` / `splendor.policy` import without the
    # compiled extension (splendor.binding) being built.
    if name == 'Splendor':
        from splendor.splendor import Splendor
        return Splendor
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
