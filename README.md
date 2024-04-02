# LLMBasics

This is a collection of some separated basic classes or functions related to LLM implementation.

Currently, it contains:

- `transformer.py`: a class for basic encoder-decoder transformer model
- `sgemm_naive.cu`: a naive implementation of SGEMM kernel for GPU, compile it by `nvcc -Xcompiler -fopenmp -arch=sm_70 sgemm_naive.cu -o sgemm`
