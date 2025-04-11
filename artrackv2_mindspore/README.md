# ARTrackV2

## Evaluation

Make sure you have installed the GPU version of MindSpore according to [link](https://www.mindspore.cn/install/en).

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Some testing examples:

- GOT10K-test

```python
cd tracking
python test.py ostrack 2stage_256_got --dataset got10k_test --thread 0 --num_gpus 1
```
