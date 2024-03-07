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
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import oshi.util.platform.mac.SysctlUtil;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Stream;

import static org.mockito.Mockito.mockStatic;
import static org.opensearch.knn.jni.JNIUtils.isAVX2SupportedBySystem;

public class JNIUtilTests extends KNNTestCase {
    public static final String MAC_CPU_FEATURES = "machdep.cpu.leaf7_features";
    public static final String LINUX_PROC_CPU_INFO = "/proc/cpuinfo";

    public void testIsAVX2SupportedBySystem_platformIsARM_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isARM).thenReturn(true);
            assertFalse(isAVX2SupportedBySystem());
        }
    }

    public void testIsAVX2SupportedBySystem_platformIsNotARMWithOSAsWindows_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isARM).thenReturn(false);
            mockedPlatform.when(Platform::isWindows).thenReturn(true);
            assertFalse(isAVX2SupportedBySystem());
        }
    }

    public void testIsAVX2SupportedBySystem_platformIsMac_returnsTrue() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isARM).thenReturn(false);
            mockedPlatform.when(Platform::isWindows).thenReturn(false);
            mockedPlatform.when(Platform::isMac).thenReturn(true);

            try (MockedStatic<SysctlUtil> mockedSysctlUtil = mockStatic(SysctlUtil.class)) {
                mockedSysctlUtil.when(() -> SysctlUtil.sysctl(MAC_CPU_FEATURES, "empty"))
                    .thenReturn(
                        "RDWRFSGS TSC_THREAD_OFFSET SGX BMI1 AVX2 SMEP BMI2 ERMS INVPCID FPU_CSDS MPX RDSEED ADX SMAP CLFSOPT IPT SGXLC MDCLEAR TSXFA IBRS STIBP L1DF ACAPMSR SSBD"
                    );
                assertTrue(isAVX2SupportedBySystem());
            }
        }
    }

    public void testIsAVX2SupportedBySystem_platformIsMac_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isARM).thenReturn(false);
            mockedPlatform.when(Platform::isWindows).thenReturn(false);
            mockedPlatform.when(Platform::isMac).thenReturn(true);

            try (MockedStatic<SysctlUtil> mockedSysctlUtil = mockStatic(SysctlUtil.class)) {
                mockedSysctlUtil.when(() -> SysctlUtil.sysctl(MAC_CPU_FEATURES, "empty")).thenReturn("NO Flags");
                assertFalse(isAVX2SupportedBySystem());
            }
        }

    }

    public void testIsAVX2SupportedBySystem_platformIsMac_throwsExceptionReturnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isARM).thenReturn(false);
            mockedPlatform.when(Platform::isWindows).thenReturn(false);
            mockedPlatform.when(Platform::isMac).thenReturn(true);

            try (MockedStatic<SysctlUtil> mockedSysctlUtil = mockStatic(SysctlUtil.class)) {
                mockedSysctlUtil.when(() -> SysctlUtil.sysctl(MAC_CPU_FEATURES, "empty")).thenThrow(RuntimeException.class);
                assertFalse(isAVX2SupportedBySystem());
            }
        }

    }

    public void testIsAVX2SupportedBySystem_platformIsLinux_returnsTrue() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isARM).thenReturn(false);
            mockedPlatform.when(Platform::isWindows).thenReturn(false);
            mockedPlatform.when(Platform::isMac).thenReturn(false);
            mockedPlatform.when(Platform::isLinux).thenReturn(true);

            try (MockedStatic<Files> mockedFiles = mockStatic(Files.class)) {
                mockedFiles.when(() -> Files.lines(Paths.get(LINUX_PROC_CPU_INFO))).thenReturn(Stream.of("flags: AVX2", "dummy string"));
                assertTrue(isAVX2SupportedBySystem());
            }
        }
    }

    public void testIsAVX2SupportedBySystem_platformIsLinux_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isARM).thenReturn(false);
            mockedPlatform.when(Platform::isWindows).thenReturn(false);
            mockedPlatform.when(Platform::isMac).thenReturn(false);
            mockedPlatform.when(Platform::isLinux).thenReturn(true);

            try (MockedStatic<Files> mockedFiles = mockStatic(Files.class)) {
                mockedFiles.when(() -> Files.lines(Paths.get(LINUX_PROC_CPU_INFO))).thenReturn(Stream.of("flags: ", "dummy string"));
                assertFalse(isAVX2SupportedBySystem());
            }
        }

    }

    public void testIsAVX2SupportedBySystem_platformIsLinux_throwsExceptionReturnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isARM).thenReturn(false);
            mockedPlatform.when(Platform::isWindows).thenReturn(false);
            mockedPlatform.when(Platform::isMac).thenReturn(false);
            mockedPlatform.when(Platform::isLinux).thenReturn(true);

            try (MockedStatic<Paths> mockedPaths = mockStatic(Paths.class)) {
                mockedPaths.when(() -> Paths.get(LINUX_PROC_CPU_INFO)).thenThrow(RuntimeException.class);
                assertFalse(isAVX2SupportedBySystem());
            }
        }

    }

}
