# Global settings for plenoptic rendering


EXTRA_DIMS = ('IR', )
assert len(EXTRA_DIMS) > 0

DIMS = ('R', 'G', 'B') + EXTRA_DIMS  # Ordering of texture, material and light dimensions.
N_DIMS = len(DIMS)
