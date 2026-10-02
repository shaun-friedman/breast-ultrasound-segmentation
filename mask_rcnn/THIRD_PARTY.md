# Third-party code

The following files were copied in December 2024 from the torchvision detection
reference scripts, as directed by the PyTorch object detection fine-tuning
tutorial. They are not my work. Apart from a two-line provenance header, they
are unchanged since the commit that added them (`bde7b9d`). The exact upstream
revision was not recorded.

| File | Purpose |
|---|---|
| `engine.py` | `train_one_epoch` and `evaluate` loops |
| `coco_eval.py` | COCO AP/AR evaluator wrapper around `pycocotools` |
| `coco_utils.py` | Converts a PyTorch dataset to a COCO ground-truth object |
| `transforms.py` | Detection transforms (not used by the current scripts) |
| `utils.py` | Logging helpers and `collate_fn` |

Source: <https://github.com/pytorch/vision/tree/main/references/detection>

These files remain under torchvision's license, reproduced below. They are not
covered by this repository's MIT license.

```
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
