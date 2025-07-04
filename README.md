# sb3-extra-buffers
Unofficial implementation of extra Stable-Baselines3 buffer classes. Mostly a proof-of-concept in current state.

---
#### Compressed Buffers
Reinforcement Learning is quite memory-hungry due to massive buffer sizes, so let's try to tackle it by not storing raw frame buffers in full `np.float32` directly and find something smaller instead. For any input data that are sparse and containing large contiguous region of repeating values, lossless compression techniques can be applied to reduce memory footprint.

One such kind of input data is Semantic Segmentation (SS) masks (with one color channel representing class labels as integers) and RGB / Color Palette game frames from retro video games should also benefit from compression.

**Some example SS Masks:**

![SS](ss-example.png)

For `4` vectorized [Doom](https://github.com/Farama-Foundation/ViZDoom) environments each storing `4096` SS masks of `256x144` resolution:
| method | dtype | shape | size |
| ------ | ----- | ----- | ---- |
| `Baseline` | `np.float32` | `(4096, 4, 1, 144, 256)` | `2.3 GB` |
| `Cast to uint8` | `np.uint8` | `(4096, 4, 1, 144, 256)` | `576 MB` |
| `RLE-auto-slice` | `np.uint8` for elements, lengths and positions | `(4096, 4, 145) x 3` | `306 MB` |
| `RLE-flatten` | `np.uint8` for elements, `np.uint16` for lengths and positions | `(4096, 4) x 3` | `20.4 MB` |
| `gzip-level-1` | `np.uint8` | `(4096, 4)` | `11.1 MB` |
| `gzip-level-5` | `np.uint8` | `(4096, 4)` | `7.65 MB` |
| `gzip-level-9` | `np.uint8` | `(4096, 4)` | `5.17 MB` |

> Currently implemented compression methods:
> - RLE (Run-Length Encoding)
> - gzip

##### Example Usage:
```python
from sb3_extra_buffers.compressed import CompressedRolloutBuffer

buffer_dtypes = dict(len_type=np.uint16, pos_type=np.uint16, elem_type=np.uint8)
buffer_kwargs = dict(dtypes=buffer_dtypes, normalize_images=normalize_images, compression_method=compression_method, auto_slice=False, compression_kwargs=compression_kwargs)
```
