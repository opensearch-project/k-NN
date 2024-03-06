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

public class JNIUtils {

    private static final Logger logger = LogManager.getLogger(JNIUtils.class);

    /**
     * Verify if the underlying system supports AVX2 SIMD Optimization or not
     * 1. If the architecture is aarch64 return false.
     * 2. If the operating system is windows return false.
     * 3. If the operating system is macOS, use oshi-core library to verify if the cpu flags
     *    contains 'avx2' and return true if it exists else false.
     * 4. If the operating system is linux, read the '/proc/cpuinfo' file path and verify if
     *    the flags contains 'avx2' and return true if it exists else false.
     */
    public static boolean isAVX2SupportedBySystem() {
        if (Platform.isARM() || Platform.isWindows()) {
            return false;
        }

        if (Platform.isMac()) {
            try {
                return AccessController.doPrivileged((PrivilegedExceptionAction<Boolean>) () -> {
                    String flags = SysctlUtil.sysctl("machdep.cpu.leaf7_features", "empty");
                    return (flags.toLowerCase(Locale.ROOT)).contains("avx2");
                });
            } catch (Exception e) {
                logger.error("[KNN] Error fetching cpu flags info. [{}]", e.getMessage(), e);
            }

        } else if (Platform.isLinux()) {
            String fileName = "/proc/cpuinfo";
            try {
                return AccessController.doPrivileged(
                    (PrivilegedExceptionAction<Boolean>) () -> (Boolean) Files.lines(Paths.get(fileName))
                        .filter(s -> s.startsWith("flags"))
                        .anyMatch(s -> StringUtils.containsIgnoreCase(s, "avx2"))
                );

            } catch (Exception e) {
                logger.error("[KNN] Error reading file [{}]. [{}]", fileName, e.getMessage(), e);
            }
        }
        return false;
    }
}
