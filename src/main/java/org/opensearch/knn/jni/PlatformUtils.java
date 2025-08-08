/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.jni;

import com.sun.jna.Platform;
import org.apache.commons.lang.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import oshi.util.platform.mac.SysctlUtil;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.AccessController;
import java.security.PrivilegedExceptionAction;
import java.util.Locale;

public class PlatformUtils {
    private static final Logger logger = LogManager.getLogger(PlatformUtils.class);

    private static volatile Boolean isAVX2Supported;
    private static volatile Boolean isAVX512Supported;
    private static volatile Boolean isAVX512SPRSupported;
    private static volatile Boolean isF16CSupported;
    private static volatile String linuxCpuFlags = null;

    /**
     * Reads and caches the CPU flags from /proc/cpuinfo on Linux systems.
     * This method ensures the flags are only read once and reused for all feature checks
     * (AVX2, AVX512, AVX512_SPR, F16C) to avoid stream consumption issues.
     * If reading fails, logs the error and returns an empty string.
     *
     * @return The cached "flags" string from /proc/cpuinfo, or an empty string if unavailable.
     */
    private static String getLinuxCpuFlags() {
        if (linuxCpuFlags != null) {
            return linuxCpuFlags;
        }

        // The "/proc/cpuinfo" is a virtual file which identifies and provides the processor details used
        // by system. This info contains "flags" for each processor which determines the qualities of that processor
        // and it's ability to process different instruction sets like mmx, avx, avx2, avx512, f16c and so on.
        // https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/deployment_guide/s2-proc-cpuinfo
        String fileName = "/proc/cpuinfo";
        try {
            linuxCpuFlags = java.security.AccessController.doPrivileged((java.security.PrivilegedExceptionAction<String>) () -> {
                try (java.util.stream.Stream<String> lines = Files.lines(Paths.get(fileName))) {
                    return lines.filter(s -> s.startsWith("flags")).findFirst().orElse("");
                }
            });
        } catch (Exception e) {
            linuxCpuFlags = "";
            logger.error("[KNN] Error reading file [{}]. [{}]", fileName, e.getMessage(), e);
        }
        return linuxCpuFlags;
    }

    static void reset() {
        isAVX2Supported = null;
        isAVX512Supported = null;
        isAVX512SPRSupported = null;
        isF16CSupported = null;
        linuxCpuFlags = null;
    }

    /**
     * Verify if the underlying system supports AVX2 SIMD Optimization or not
     * 1. If the architecture is not x86 return false.
     * 2. If the operating system is not Mac or Linux return false(for example Windows).
     * 3. If the operating system is macOS, use oshi library to verify if the cpu flags
     *    contains 'avx2' and return true if it exists else false.
     * 4. If the operating system is linux, read the '/proc/cpuinfo' file path and verify if
     *    the flags contains 'avx2' and return true if it exists else false.
     */
    public static boolean isAVX2SupportedBySystem() {
        if (!Platform.isIntel() || Platform.isWindows()) {
            isAVX2Supported = false;
        }

        if (isAVX2Supported != null) {
            return isAVX2Supported;
        }

        if (Platform.isMac()) {
            // sysctl or system control retrieves system info and allows processes with appropriate privileges
            // to set system info. This system info contains the machine dependent cpu features that are supported by it.
            // On MacOS, if the underlying processor supports AVX2 instruction set, it will be listed under the "leaf7"
            // subset of instructions ("sysctl -a | grep machdep.cpu.leaf7_features").
            // https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/sysctl.3.html
            try {
                isAVX2Supported = AccessController.doPrivileged((PrivilegedExceptionAction<Boolean>) () -> {
                    String flags = SysctlUtil.sysctl("machdep.cpu.leaf7_features", "empty");
                    return (flags.toLowerCase(Locale.ROOT)).contains("avx2");
                });
            } catch (Exception e) {
                isAVX2Supported = false;
                logger.error("[KNN] Error fetching cpu flags info. [{}]", e.getMessage(), e);
            }

        } else if (Platform.isLinux()) {
            // Here, we are trying to read the details of all processors used by system and find if any of the processor
            // supports AVX2 instructions. Pentium and Celeron are a couple of examples which doesn't support AVX2
            // https://ark.intel.com/content/www/us/en/ark/products/199285/intel-pentium-gold-g6600-processor-4m-cache-4-20-ghz.html
            String flags = getLinuxCpuFlags();
            isAVX2Supported = StringUtils.containsIgnoreCase(flags, "avx2");
        }
        return isAVX2Supported;
    }

    public static boolean isAVX512SupportedBySystem() {
        if (isAVX512Supported == null) {
            isAVX512Supported = areAVX512FlagsAvailable(new String[] { "avx512f", "avx512cd", "avx512vl", "avx512dq", "avx512bw" });
        }
        return isAVX512Supported;
    }

    public static boolean isAVX512SPRSupportedBySystem() {
        if (isAVX512SPRSupported == null) {
            isAVX512SPRSupported = areAVX512FlagsAvailable(new String[] { "avx512_fp16", "avx512_bf16", "avx512_vpopcntdq" });
        }
        return isAVX512SPRSupported;
    }

    private static boolean areAVX512FlagsAvailable(String[] avx512) {
        // AVX512 has multiple flags, which control various features. k-nn requires the same set of flags as faiss to compile
        // using avx512. Please update these if faiss updates their compilation instructions in the future.
        // https://github.com/facebookresearch/faiss/blob/main/faiss/CMakeLists.txt
        if (!Platform.isIntel() || Platform.isMac() || Platform.isWindows()) {
            return false;
        }
        if (Platform.isLinux()) {
            // Here, we are trying to read the details of all processors used by system and find if any of the processor
            // supports AVX512 instructions supported by faiss.
            String flags = getLinuxCpuFlags();
            return java.util.Arrays.stream(avx512).allMatch(flags::contains);
        }
        return false;
    }

    public static boolean isF16CSupportedBySystem() {
        if (!Platform.isIntel() || Platform.isWindows()) {
            isF16CSupported = false;
        }
        if (isF16CSupported != null) {
            return isF16CSupported;
        }
        if (Platform.isMac()) {
            // sysctl or system control retrieves system info and allows processes with appropriate privileges
            // to set system info. This system info contains the machine dependent cpu features that are supported by it.
            // On MacOS, if the underlying processor supports F16C instruction set, it will be listed under the "machdep.cpu.features"
            // instruction subset ("sysctl -a | grep machdep.cpu.features").
            // https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/sysctl.3.html
            try {
                isF16CSupported = AccessController.doPrivileged((PrivilegedExceptionAction<Boolean>) () -> {
                    String flags = SysctlUtil.sysctl("machdep.cpu.features", "empty");
                    String leaf7Flags = SysctlUtil.sysctl("machdep.cpu.leaf7_features", "empty");

                    return (flags != null && flags.toLowerCase(Locale.ROOT).contains("f16c"))
                        || (leaf7Flags != null && leaf7Flags.toLowerCase(Locale.ROOT).contains("f16c"));
                });
            } catch (Exception e) {
                isF16CSupported = false;
                logger.error("[KNN] Error fetching F16C cpu flags on macOS. [{}]", e.getMessage(), e);
            }

        } else if (Platform.isLinux()) {
            // Here, we are trying to read the details of all processors used by system and find if any of the processor
            // supports F16C instructions. Pentium and Celeron are a couple of examples which doesn't support F16C
            // https://ark.intel.com/content/www/us/en/ark/products/199285/intel-pentium-gold-g6600-processor-4m-cache-4-20-ghz.html
            String flags = getLinuxCpuFlags();
            isF16CSupported = StringUtils.containsIgnoreCase(flags, "f16c");
        }
        return isF16CSupported;
    }

    public static boolean isSIMDAVX2SupportedBySystem() {
        return isAVX2SupportedBySystem() && isF16CSupportedBySystem();
    }

    public static boolean isSIMDAVX512SupportedBySystem() {
        return isAVX512SupportedBySystem() && isF16CSupportedBySystem();
    }

    public static boolean isSIMDAVX512SPRSpecSupportedBySystem() {
        return isAVX512SPRSupportedBySystem() && isF16CSupportedBySystem();
    }

}
