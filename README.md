## Paper

This repository is an implementation for the following paper.
- [**Training with Mixed-Precision Floating-Point Assignments**](https://openreview.net/forum?id=ZoXi7n54OB)</br>
  [Wonyeol Lee](https://wonyeol.github.io/),
  [Rahul Sharma](https://www.microsoft.com/en-us/research/people/rahsha/),
  [Alex Aiken](https://theory.stanford.edu/~aiken/)</br>
  [TMLR 2023]([https://popl23.sigplan.org/](https://openreview.net/group?id=TMLR))

## Scripts

The results on CIFAR datasets in the paper can be reproduced by running the following commands.
- Figures 4, 10, 11, 12: `bash main_1.sh`
- Figures 5, 16        : `bash main_2.sh`
- Figure  17           : `bash main_3.sh`


##  Packages

The following conda packages were used in our experiments.
  
```    
    # packages in environment at /[user]/.conda/envs/test2:
    #
    # Name                    Version                   Build  Channel
    _libgcc_mutex             0.1                 conda_forge    conda-forge
    _openmp_mutex             4.5                       1_gnu    conda-forge
    astunparse                1.6.3              pyhd8ed1ab_0    conda-forge
    brotli                    1.0.9                hb283c62_7    conda-forge
    brotli-bin                1.0.9                hb283c62_7    conda-forge
    brotlipy                  0.7.0           py38hf25ed59_1004    conda-forge
    bzip2                     1.0.8                h4e0d66e_4    conda-forge
    c-ares                    1.18.1               h4e0d66e_0    conda-forge
    ca-certificates           2021.10.8            h1084571_0    conda-forge
    certifi                   2021.10.8        py38h8328f6c_2    conda-forge
    cffi                      1.15.0           py38hae6c81b_0    conda-forge
    charset-normalizer        2.0.12             pyhd8ed1ab_0    conda-forge
    cmake                     3.23.1               h4f7acd8_0    conda-forge
    cryptography              36.0.0           py38h179485c_0  
    cudatoolkit               10.2.89             h455192d_10    conda-forge
    cudnn                     7.6.5_10.2         667.g338a052    https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda-early-access
    cycler                    0.11.0             pyhd8ed1ab_0    conda-forge
    dataclasses               0.8                pyhc8e2a94_3    conda-forge
    expat                     2.4.8                hbbae597_0    conda-forge
    fonttools                 4.32.0           py38hf25ed59_0    conda-forge
    freetype                  2.10.4               h8a70690_1    conda-forge
    fribidi                   1.0.10               h339bb43_0    conda-forge
    future                    0.18.2           py38hf8b3453_5    conda-forge
    giflib                    5.2.1                h339bb43_2    conda-forge
    htop                      3.1.2                h5c45dff_0    conda-forge
    idna                      3.3                pyhd8ed1ab_0    conda-forge
    jbig                      2.1               h4e0d66e_2003    conda-forge
    jpeg                      9e                   hb283c62_1    conda-forge
    keyutils                  1.6.1                hb283c62_0    conda-forge
    kiwisolver                1.4.2            py38hc53772b_1    conda-forge
    krb5                      1.19.3               ha6b4ebd_0    conda-forge
    lcms2                     2.11                 h6d9531b_1    conda-forge
    ld_impl_linux-ppc64le     2.36.1               ha35d02b_2    conda-forge
    lerc                      3.0                  h3b9df90_0    conda-forge
    libblas                   3.9.0           14_linuxppc64le_openblas    conda-forge
    libbrotlicommon           1.0.9                hb283c62_7    conda-forge
    libbrotlidec              1.0.9                hb283c62_7    conda-forge
    libbrotlienc              1.0.9                hb283c62_7    conda-forge
    libcblas                  3.9.0           14_linuxppc64le_openblas    conda-forge
    libcurl                   7.82.0               h1ac174b_0    conda-forge
    libdeflate                1.10                 h4e0d66e_0    conda-forge
    libedit                   3.1.20191231         h41a240f_2    conda-forge
    libev                     4.33                 h6eb9509_1    conda-forge
    libffi                    3.4.2                h4e0d66e_5    conda-forge
    libgcc-ng                 11.2.0              h7698a5e_15    conda-forge
    libgfortran-ng            11.2.0              hfdc3801_15    conda-forge
    libgfortran5              11.2.0              he58fbb4_15    conda-forge
    libgomp                   11.2.0              h7698a5e_15    conda-forge
    liblapack                 3.9.0           14_linuxppc64le_openblas    conda-forge
    libnghttp2                1.47.0               h350ef5c_0    conda-forge
    libnl                     3.5.0                h4e0d66e_0    conda-forge
    libopenblas               0.3.20          pthreads_h60f2977_0    conda-forge
    libpng                    1.6.37               h38e1d09_2    conda-forge
    libssh2                   1.10.0               he881182_2    conda-forge
    libstdcxx-ng              11.2.0              habdf983_15    conda-forge
    libtiff                   4.3.0                hecb0ed6_3    conda-forge
    libuv                     1.43.0               h4e0d66e_0    conda-forge
    libwebp                   1.2.2                h740ddc3_0    conda-forge
    libwebp-base              1.2.2                h4e0d66e_1    conda-forge
    libxcb                    1.13              h4e0d66e_1004    conda-forge
    libzlib                   1.2.11            hb283c62_1014    conda-forge
    lz4-c                     1.9.3                h3b9df90_1    conda-forge
    matplotlib                3.5.1            py38hf8b3453_0    conda-forge
    matplotlib-base           3.5.1            py38hb0e4686_0    conda-forge
    munkres                   1.1.4              pyh9f0ad1d_0    conda-forge
    mypy                      0.942            py38hf25ed59_0    conda-forge
    mypy_extensions           0.4.3            py38h8328f6c_5    conda-forge
    ncurses                   6.3                  h140841e_2  
    ninja                     1.10.2               h2acdbc0_1    conda-forge
    numpy                     1.22.3           py38h0d5f677_2    conda-forge
    openjpeg                  2.4.0                h29f4549_1    conda-forge
    openssl                   3.0.2                hb283c62_1    conda-forge
    packaging                 21.3               pyhd8ed1ab_0    conda-forge
    pillow                    9.0.1            py38hccce32e_1    conda-forge
    pip                       21.2.4           py38h6ffa863_0  
    psutil                    5.9.0            py38hf25ed59_1    conda-forge
    pthread-stubs             0.4               h339bb43_1001    conda-forge
    pycparser                 2.21               pyhd8ed1ab_0    conda-forge
    pyopenssl                 22.0.0             pyhd8ed1ab_0    conda-forge
    pyparsing                 3.0.8              pyhd8ed1ab_0    conda-forge
    pysocks                   1.7.1            py38hf8b3453_5    conda-forge
    python                    3.8.12          h60499bd_1_cpython    conda-forge
    python-dateutil           2.8.2              pyhd8ed1ab_0    conda-forge
    python_abi                3.8                      2_cp38    conda-forge
    pyyaml                    6.0              py38hf25ed59_4    conda-forge
    readline                  8.1.2                h140841e_1  
    requests                  2.27.1             pyhd8ed1ab_0    conda-forge
    rhash                     1.4.1                h4e0d66e_0    conda-forge
    setuptools                59.5.0           py38hf8b3453_0    conda-forge
    six                       1.16.0             pyh6c4a22f_0    conda-forge
    sqlite                    3.38.2               hd7247d8_0  
    tk                        8.6.11               h7e00dab_0  
    tomli                     2.0.1              pyhd8ed1ab_0    conda-forge
    torch                     1.9.0a0+gitdfbd030          pypi_0    pypi
    torchvision               0.10.0a0+ca1a620          pypi_0    pypi
    tornado                   6.1              py38h6e87771_3    conda-forge
    typing_extensions         4.1.1              pyha770c72_0    conda-forge
    unicodedata2              14.0.0           py38hf25ed59_1    conda-forge
    urllib3                   1.26.9             pyhd8ed1ab_0    conda-forge
    wheel                     0.37.1             pyhd3eb1b0_0  
    xorg-libxau               1.0.9                h4e0d66e_0    conda-forge
    xorg-libxdmcp             1.1.3                h4e0d66e_0    conda-forge
    xz                        5.2.5                h140841e_0  
    yaml                      0.2.5                h4e0d66e_2    conda-forge
    zlib                      1.2.11            hb283c62_1014    conda-forge
    zstd                      1.5.1                h65c4b1a_0    conda-forge
```
