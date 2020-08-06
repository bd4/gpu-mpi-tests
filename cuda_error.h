/** Error handling macros for CUDA. All cuda routines should be wrapped
 * in either a CHECK or WARN call; CHECK will print the error and exit on
 * failure, while WARN will just print the error on failure. PTRINFO is
 * a convenience routine for debugging data that needs to be moved to
 * reg storage.
 *
 * By default, no checking is done, for maximum performance on production
 * runs. define "GPU_CHECK_CALLS" to enable checks.
 *
 * TODO: add fortran interface
 * */
#include "cuda_runtime_api.h"
#include "cublas_v2.h"


#ifndef GPU_NO_CHECK_CALLS
#define CHECK(msg, val) __checkCuda(msg, (val), __FILE__, __LINE__, true)
#define WARN(msg, val) __checkCuda(msg, (val), __FILE__, __LINE__, false)
#define PTRINFO(msg, ptr) __print_cuda_ptr_info(msg, ptr)
#define MEMINFO(msg, ptr, size) __print_cuda_mem_info(msg, ptr, size)
#else
#define CHECK(msg, val) { int __i = (val); }
#define WARN(msg, val)  { int __i = (val); }
#define PTRINFO(msg, ptr) { void *__p = (void *)(ptr); }
#define MEMINFO(msg, ptr, size)  { void *__p = (void *)(ptr); }
#endif


inline int __checkCuda(const char *msg, cudaError_t val, const char *fname,
                        const int line, bool abort=true) {
  if (val != cudaSuccess) {
     fprintf(stderr,
             "%s(%i): CUDA Error (%s) %i: %s\n",
             fname, line, msg, val, cudaGetErrorString(val));
     if (abort) {
       cudaDeviceReset();
         exit(EXIT_FAILURE);
     }
  }
  return (int)val;
}


// overload for cublasStatus_t
inline int __checkCuda(const char *msg, cublasStatus_t val, const char *fname,
                        const int line, bool abort=true) {
  if (val != CUBLAS_STATUS_SUCCESS) {
     const char *err_s = "OTHER";
     if (val == CUBLAS_STATUS_NOT_INITIALIZED) {
         err_s = "NOT_INITIALIZED";
     } else if (val == CUBLAS_STATUS_INVALID_VALUE) {
         err_s = "INVALID_VALUE";
     }
     fprintf(stderr,
             "%s(%i): CUDA Error (%s) %i: %s\n",
             fname, line, msg, val, err_s);
     if (abort) {
       cudaDeviceReset();
       exit(EXIT_FAILURE);
     }
  }
  return (int)val;
}


inline void __print_cuda_ptr_info(const char *label, void *ptr) {
  cudaError_t cu_err;
  cudaPointerAttributes attr;
  const char *type_name = NULL;

  if (ptr == NULL) {
    printf("CUDA pointer %s (%zx): NULL\n", label, ptr);
    return;
  }

  // NB: the 'type' attribute was not added until CUDA 10.0, use memoryType
  // for better compatibility
  cu_err = cudaPointerGetAttributes(&attr, ptr);
  if (cu_err != cudaSuccess) {
    if (cu_err == cudaErrorInvalidValue) {
      type_name = "Invalid (non-unified addressing)";
    } else {
      WARN("get pointer attr", cu_err);
      return;
    }
  } else if (attr.type == cudaMemoryTypeDevice) {
    type_name = "Device";
  } else if (attr.type == cudaMemoryTypeManaged) {
    type_name = "Managed";
  } else if (attr.type == cudaMemoryTypeHost) {
    type_name = "Host";
  } else if (attr.type == cudaMemoryTypeUnregistered) {
    type_name = "Unregistered";
  }
  printf("CUDA pointer %s (%zx): %s\n", label, ptr, type_name);
}


inline void __print_cuda_mem_info(const char *label, void *ptr, size_t size) {
  cudaError_t cu_err;
  cudaPointerAttributes pointer_attr;
  int mem_attr = -123;
  bool is_managed = false;

  cu_err = cudaPointerGetAttributes(&pointer_attr, ptr);
  if (cu_err != cudaSuccess) {
    if (cu_err == cudaErrorInvalidValue) {
      printf("CUDA PreferredLocation of '%s' is NOT CUDA\n", label);
      return;
    } else {
      WARN("get pointer attr", cu_err);
      return;
    }
  } else if (pointer_attr.type == cudaMemoryTypeManaged) {
    is_managed = true;
  }

  if (!is_managed) {
    printf("CUDA PreferredLocation of '%s' is UNMANAGED\n", label);
    return;
  }
 
  WARN("get mem range preferred location",
       cudaMemRangeGetAttribute(&mem_attr, sizeof(mem_attr),
                                cudaMemRangeAttributePreferredLocation,
                                ptr, size));
  if (mem_attr == cudaCpuDeviceId) {
      printf("CUDA PreferredLocation of '%s' is CPU (%d)\n", label, mem_attr);
  } else if (mem_attr == cudaInvalidDeviceId) {
      printf("CUDA PreferredLocation of '%s' is INVALID (%d)\n",
             label, mem_attr);
  } else {
      printf("CUDA PreferredLocation of '%s' is DEVICE (%d)\n",
             label, mem_attr);
  }
}
