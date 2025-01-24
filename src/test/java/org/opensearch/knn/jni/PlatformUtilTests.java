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
import static org.opensearch.knn.jni.PlatformUtils.isAVX2SupportedBySystem;
import static org.opensearch.knn.jni.PlatformUtils.isAVX512SupportedBySystem;
import static org.opensearch.knn.jni.PlatformUtils.isAVX512SPRSupportedBySystem;

public class PlatformUtilTests extends KNNTestCase {
    public static final String MAC_CPU_FEATURES = "machdep.cpu.leaf7_features";
    public static final String LINUX_PROC_CPU_INFO = "/proc/cpuinfo";

    public void testIsAVX2SupportedBySystem_platformIsNotIntel_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(false);
            assertFalse(isAVX2SupportedBySystem());
        }
    }

    public void testIsAVX2SupportedBySystem_platformIsIntelWithOSAsWindows_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isWindows).thenReturn(true);
            assertFalse(isAVX2SupportedBySystem());
        }
    }

    public void testIsAVX2SupportedBySystem_platformIsMac_returnsTrue() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
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
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isMac).thenReturn(true);

            try (MockedStatic<SysctlUtil> mockedSysctlUtil = mockStatic(SysctlUtil.class)) {
                mockedSysctlUtil.when(() -> SysctlUtil.sysctl(MAC_CPU_FEATURES, "empty")).thenReturn("NO Flags");
                assertFalse(isAVX2SupportedBySystem());
            }
        }

    }

    public void testIsAVX2SupportedBySystem_platformIsMac_throwsExceptionReturnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isMac).thenReturn(true);

            try (MockedStatic<SysctlUtil> mockedSysctlUtil = mockStatic(SysctlUtil.class)) {
                mockedSysctlUtil.when(() -> SysctlUtil.sysctl(MAC_CPU_FEATURES, "empty")).thenThrow(RuntimeException.class);
                assertFalse(isAVX2SupportedBySystem());
            }
        }

    }

    public void testIsAVX2SupportedBySystem_platformIsLinux_returnsTrue() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
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
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
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
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isMac).thenReturn(false);
            mockedPlatform.when(Platform::isLinux).thenReturn(true);

            try (MockedStatic<Paths> mockedPaths = mockStatic(Paths.class)) {
                mockedPaths.when(() -> Paths.get(LINUX_PROC_CPU_INFO)).thenThrow(RuntimeException.class);
                assertFalse(isAVX2SupportedBySystem());
            }
        }

    }

    // AVX512 tests

    public void testIsAVX512SupportedBySystem_platformIsNotIntel_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(false);
            assertFalse(isAVX512SupportedBySystem());
        }
    }

    public void testIsAVX512SupportedBySystem_platformIsMac_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isMac).thenReturn(false);
            assertFalse(isAVX512SupportedBySystem());
        }
    }

    public void testIsAVX512SupportedBySystem_platformIsIntelMac_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isMac).thenReturn(true);
            assertFalse(isAVX512SupportedBySystem());
        }
    }

    public void testIsAVX512SupportedBySystem_platformIsIntelWithOSAsWindows_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isWindows).thenReturn(true);
            assertFalse(isAVX512SupportedBySystem());
        }
    }

    public void testIsAVX512SupportedBySystem_platformIsLinuxAllAVX512FlagsPresent_returnsTrue() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isLinux).thenReturn(true);

            try (MockedStatic<Files> mockedFiles = mockStatic(Files.class)) {
                mockedFiles.when(() -> Files.lines(Paths.get(LINUX_PROC_CPU_INFO)))
                    .thenReturn(Stream.of("flags: AVX2 avx512f avx512cd avx512vl avx512dq avx512bw", "dummy string"));
                assertTrue(isAVX512SupportedBySystem());
            }
        }
    }

    public void testIsAVX512SupportedBySystem_platformIsLinuxSomeAVX512FlagsPresent_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isLinux).thenReturn(true);

            try (MockedStatic<Files> mockedFiles = mockStatic(Files.class)) {
                mockedFiles.when(() -> Files.lines(Paths.get(LINUX_PROC_CPU_INFO)))
                    .thenReturn(Stream.of("flags: AVX2 avx512vl avx512dq avx512bw avx512vbmi umip pku ospke avx512_vbmi2", "dummy string"));
                assertFalse(isAVX512SupportedBySystem());
            }
        }
    }

    // Tests AVX512 instructions available since Intel(R) Sapphire Rapids.

    public void testIsAVX512SPRSupportedBySystem_platformIsNotIntel_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(false);
            assertFalse(isAVX512SPRSupportedBySystem());
        }
    }

    public void testIsAVX512SPRSupportedBySystem_platformIsMac_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isMac).thenReturn(false);
            assertFalse(isAVX512SPRSupportedBySystem());
        }
    }

    public void testIsAVX512SPRSupportedBySystem_platformIsIntelMac_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isMac).thenReturn(true);
            assertFalse(isAVX512SPRSupportedBySystem());
        }
    }

    public void testIsAVX512SPRSupportedBySystem_platformIsIntelWithOSAsWindows_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isWindows).thenReturn(true);
            assertFalse(isAVX512SPRSupportedBySystem());
        }
    }

    public void testIsAVX512SPRSupportedBySystem_platformIsLinuxAllAVX512SPRFlagsPresent_returnsTrue() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isLinux).thenReturn(true);

            try (MockedStatic<Files> mockedFiles = mockStatic(Files.class)) {
                mockedFiles.when(() -> Files.lines(Paths.get(LINUX_PROC_CPU_INFO)))
                    .thenReturn(Stream.of("flags: avx512_fp16 avx512_bf16 avx512_vpopcntdq", "dummy string"));
                assertTrue(isAVX512SPRSupportedBySystem());
            }
        }
    }

    public void testIsAVX512SPRSupportedBySystem_platformIsLinuxSomeAVX512SPRFlagsPresent_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isLinux).thenReturn(true);

            try (MockedStatic<Files> mockedFiles = mockStatic(Files.class)) {
                mockedFiles.when(() -> Files.lines(Paths.get(LINUX_PROC_CPU_INFO)))
                    .thenReturn(Stream.of("flags: avx512_fp16 avx512_vpopcntdq", "dummy string"));
                assertFalse(isAVX512SPRSupportedBySystem());
            }
        }
    }
}
