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

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import oshi.util.platform.mac.SysctlUtil;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.AccessController;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import java.util.Arrays;
import java.util.Locale;
import java.util.stream.Stream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class PlatformUtils {

    private static final Logger logger = LogManager.getLogger(PlatformUtils.class);

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
            return false;
        }

        if (Platform.isMac()) {

            // sysctl or system control retrieves system info and allows processes with appropriate privileges
            // to set system info. This system info contains the machine dependent cpu features that are supported by it.
            // On MacOS, if the underlying processor supports AVX2 instruction set, it will be listed under the "leaf7"
            // subset of instructions ("sysctl -a | grep machdep.cpu.leaf7_features").
            // https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/sysctl.3.html
            try {
                return AccessController.doPrivileged((PrivilegedExceptionAction<Boolean>) () -> {
                    String flags = SysctlUtil.sysctl("machdep.cpu.leaf7_features", "empty");
                    return (flags.toLowerCase(Locale.ROOT)).contains("avx2");
                });
            } catch (Exception e) {
                logger.error("[KNN] Error fetching cpu flags info. [{}]", e.getMessage(), e);
            }

        } else if (Platform.isLinux()) {
            try {
                // The "/proc/cpuinfo" is a virtual file which identifies and provides the processor details used
                // by system. This info contains "flags" for each processor which determines the qualities of that processor
                // and it's ability to process different instruction sets like mmx, avx, avx2 and so on.
                // https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/deployment_guide/s2-proc-cpuinfo
                // Here, we are trying to read the details of all processors used by system and find if any of the processor
                // supports AVX2 instructions. Pentium and Celeron are a couple of examples which doesn't support AVX2
                // https://ark.intel.com/content/www/us/en/ark/products/199285/intel-pentium-gold-g6600-processor-4m-cache-4-20-ghz.html
                // String fileName = "/proc/cpuinfo";
                // try {
                // return AccessController.doPrivileged(
                // (PrivilegedExceptionAction<Boolean>) () -> (Boolean) Files.lines(Paths.get(fileName))
                // .filter(s -> s.startsWith("flags"))
                // .anyMatch(s -> StringUtils.containsIgnoreCase(s, "avx2"))
                // );

                // } catch (Exception e) {
                // logger.error("[KNN] Error reading file [{}]. [{}]", fileName, e.getMessage(), e);
                // }

                Process p = Runtime.getRuntime().exec("lscpu");
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                    return reader.lines().anyMatch(line -> line.toLowerCase().contains("avx512"));
                } catch (Exception e) {
                    logger.error("[KNN] Exception: ", e);
                }
            } catch (IOException ex) {}
        }
        return false;
    }

    public static boolean isAVX512SupportedBySystem() {
        return areAVX512FlagsAvailable(new String[] { "avx512f", "avx512cd", "avx512vl", "avx512dq", "avx512bw" });
    }

    public static boolean isAVX512SPRSupportedBySystem() {
        return areAVX512FlagsAvailable(new String[] { "avx512_fp16", "avx512_bf16", "avx512_vpopcntdq" });
    }

    private static boolean areAVX512FlagsAvailable(String[] avx512) {
        // AVX512 has multiple flags, which control various features. k-nn requires the same set of flags as faiss to compile
        // using avx512. Please update these if faiss updates their compilation instructions in the future.
        // https://github.com/facebookresearch/faiss/blob/main/faiss/CMakeLists.txt

        if (!Platform.isIntel() || Platform.isMac() || Platform.isWindows()) {
            return false;
        }

        if (Platform.isLinux()) {
            // The "/proc/cpuinfo" is a virtual file which identifies and provides the processor details used
            // by system. This info contains "flags" for each processor which determines the qualities of that processor
            // and it's ability to process different instruction sets like mmx, avx, avx2, avx512 and so on.
            // https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/deployment_guide/s2-proc-cpuinfo
            // Here, we are trying to read the details of all processors used by system and find if any of the processor
            // supports AVX512 instructions supported by faiss.
            String fileName = "/proc/cpuinfo";

            try {
                return AccessController.doPrivileged((PrivilegedExceptionAction<Boolean>) () -> {
                    Stream<String> linestream = Files.lines(Paths.get(fileName));
                    String flags = linestream.filter(line -> line.startsWith("flags")).findFirst().orElse("");
                    return Arrays.stream(avx512).allMatch(flags::contains);
                });

            } catch (PrivilegedActionException e) {
                logger.error("[KNN] Error reading file [{}]. [{}]", fileName, e.getMessage(), e);
            }
        }
        return false;
    }
}
