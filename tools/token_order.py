"""Control the order pyctcdecode visits candidate tokens in.

pyctcdecode collects each frame's candidates in a Python set and iterates it,
so the layout of a CPython hash table decides how near-ties break. On a small
minority of inputs that reaches the output, which makes it a source of
disagreement that is about neither implementation being wrong. Pinning the
order removes it.

The decode settings that used to live here — beam width, hotword weight — are
deliberately gone. They are recorded in each reference file and read back from
it, so a reference made at one beam width cannot be silently compared against
a decode run at another.
"""


def ordered_set_class(key):
    """A `set` subclass that iterates in a chosen order.

    Assigned over the module-global `set` in `pyctcdecode.decoder`, this
    controls the order candidate tokens are visited at each frame — which
    decides the order beams are appended, and so how ties break. Shadowing a
    module global affects only that module; nothing else sees it.
    """

    class OrderedSet(set):
        def __iter__(self):
            return iter(sorted(set.__iter__(self), key=key))

        def __or__(self, other):
            return OrderedSet(set.__or__(self, other))

    return OrderedSet


def ascending():
    return ordered_set_class(lambda i: int(i))


def descending():
    return ordered_set_class(lambda i: -int(i))
