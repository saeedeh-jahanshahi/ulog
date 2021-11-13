# ulog
This repository contains the files for ulog data type library. ulog is a logarithmic base data type that is implemented in software. The purpose of ulog is reduction of read and write to/from memory to decrease the time and power consumption of error tolerance computations, significantly. This library uses the vectorization capability of intel AVX512 CPUs. therefore use avx512 CPUs to run this library. in bellow the different files of ulog library is defined.
## convert_avx512 file
this file contains all functions that need to convert from/to double or float to/from ulog.
## log_operation_avx512 file
this file contains all operation such as mulitplication, addition, reduction by addition, exponentiation, square root and division in ulog scheme.
## macro_avx512.h
this file contains the difinition of all macros that uses on whole project
