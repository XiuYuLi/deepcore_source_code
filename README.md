# deepcore_source_code

  This iterm is subpart source of deepcore v0.7, include fftconv, cellconvand and gemm operation; remove the conv, reduce ,batch-normalization, pooling and activation operation.

  deepcore is based on CUDA and very fast, if want to use it please do with below:
      0 compile and build the device kernels in deepcore_device,copy the fatBinary to the deepcore/include/dev/*/kbin_sm*.h
      1 compile and build the deepcore to generate deepcore.dll or libdeepcore.so

Note:
    The data struct of deepcore is CNHW(Not NCHW), please see the deepcore doc for more info.# deepcore_source_code
