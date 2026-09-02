/*
 * Compatibility shim for the Triton CUDA driver helper on CentOS 7/GCC 4.8.
 * Triton's driver.c includes <stdatomic.h> but does not use any C11 atomics.
 */
