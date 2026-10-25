// The configuration libaom's CMake would generate, reduced to what cnn.c and ml.c read: a
// generic target, so every RTCD function is its C version.
#ifndef AOM_CONFIG_H_
#define AOM_CONFIG_H_
#define ARCH_ARM 0
#define ARCH_X86 0
#define ARCH_X86_64 0
#define CONFIG_DEBUG 0
#define CONFIG_MULTITHREAD 0
#define CONFIG_NN_V2 0
#define CONFIG_REALTIME_ONLY 0
#define HAVE_PTHREAD_H 0
#define INLINE inline
#endif
