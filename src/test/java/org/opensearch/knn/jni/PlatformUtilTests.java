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

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.mockito.MockedStatic;
import oshi.util.platform.mac.SysctlUtil;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Stream;

import static org.mockito.Mockito.mockStatic;
import static org.opensearch.knn.jni.PlatformUtils.isAVX2SupportedBySystem;
import static org.opensearch.knn.jni.PlatformUtils.isAVX512SupportedBySystem;
import static org.opensearch.knn.jni.PlatformUtils.isAVX512SPRSupportedBySystem;

public class PlatformUtilTests extends Assert {
    public static final String MAC_CPU_FEATURES = "machdep.cpu.leaf7_features";
    public static final String LINUX_PROC_CPU_INFO = "/proc/cpuinfo";

    @Before
    public void setUp() {
        PlatformUtils.reset();
    }

    @Test
    public void testIsAVX2SupportedBySystem_platformIsNotIntel_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(false);
            assertFalse(isAVX2SupportedBySystem());
        }
    }

    @Test
    public void testIsAVX2SupportedBySystem_platformIsIntelWithOSAsWindows_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isWindows).thenReturn(true);
            assertFalse(isAVX2SupportedBySystem());
        }
    }

    @Test
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

    @Test
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

    @Test
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

    @Test
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

    @Test
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

    @Test
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
    @Test
    public void testIsAVX512SupportedBySystem_platformIsNotIntel_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(false);
            assertFalse(isAVX512SupportedBySystem());
        }
    }

    @Test
    public void testIsAVX512SupportedBySystem_platformIsMac_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isMac).thenReturn(false);
            assertFalse(isAVX512SupportedBySystem());
        }
    }

    @Test
    public void testIsAVX512SupportedBySystem_platformIsIntelMac_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isMac).thenReturn(true);
            assertFalse(isAVX512SupportedBySystem());
        }
    }

    @Test
    public void testIsAVX512SupportedBySystem_platformIsIntelWithOSAsWindows_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isWindows).thenReturn(true);
            assertFalse(isAVX512SupportedBySystem());
        }
    }

    @Test
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

    @Test
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
    @Test
    public void testIsAVX512SPRSupportedBySystem_platformIsNotIntel_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(false);
            assertFalse(isAVX512SPRSupportedBySystem());
        }
    }

    @Test
    public void testIsAVX512SPRSupportedBySystem_platformIsMac_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isMac).thenReturn(false);
            assertFalse(isAVX512SPRSupportedBySystem());
        }
    }

    @Test
    public void testIsAVX512SPRSupportedBySystem_platformIsIntelMac_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isMac).thenReturn(true);
            assertFalse(isAVX512SPRSupportedBySystem());
        }
    }

    @Test
    public void testIsAVX512SPRSupportedBySystem_platformIsIntelWithOSAsWindows_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isWindows).thenReturn(true);
            assertFalse(isAVX512SPRSupportedBySystem());
        }
    }

    @Test
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

    @Test
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

    @Test
    public void testIsF16CSupportedBySystem_platformIsNotIntel_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(false);
            assertFalse(PlatformUtils.isF16CSupportedBySystem());
        }
    }

    @Test
    public void testIsF16CSupportedBySystem_platformIsIntelWithOSAsWindows_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isWindows).thenReturn(true);
            assertFalse(PlatformUtils.isF16CSupportedBySystem());
        }
    }

    @Test
    public void testIsF16CSupportedBySystem_platformIsMac_returnsTrue() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isMac).thenReturn(true);

            try (MockedStatic<SysctlUtil> mockedSysctlUtil = mockStatic(SysctlUtil.class)) {
                mockedSysctlUtil.when(() -> SysctlUtil.sysctl("machdep.cpu.features", "empty"))
                    .thenReturn(
                        "FPU VME DE PSE TSC MSR PAE MCE CX8 APIC SEP MTRR PGE MCA CMOV PAT PSE36 CLFSH MMX FXSR SSE SSE2 HTT SSE3 PCLMULQDQ SSSE3 FMA CX16 SSE4.1 SSE4.2 x2APIC MOVBE POPCNT AES F16C RDRAND"
                    );
                assertTrue(PlatformUtils.isF16CSupportedBySystem());
            }
        }
    }

    @Test
    public void testIsF16CSupportedBySystem_platformIsMac_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isMac).thenReturn(true);

            try (MockedStatic<SysctlUtil> mockedSysctlUtil = mockStatic(SysctlUtil.class)) {
                mockedSysctlUtil.when(() -> SysctlUtil.sysctl("machdep.cpu.features", "empty"))
                    .thenReturn("FPU VME DE PSE TSC MSR PAE MCE CX8 APIC SEP MTRR PGE MCA CMOV PAT PSE36 CLFSH MMX FXSR SSE SSE2 HTT SSE3");
                assertFalse(PlatformUtils.isF16CSupportedBySystem());
            }
        }
    }

    @Test
    public void testIsF16CSupportedBySystem_platformIsMac_throwsExceptionReturnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isMac).thenReturn(true);

            try (MockedStatic<SysctlUtil> mockedSysctlUtil = mockStatic(SysctlUtil.class)) {
                mockedSysctlUtil.when(() -> SysctlUtil.sysctl("machdep.cpu.features", "empty")).thenThrow(RuntimeException.class);
                assertFalse(PlatformUtils.isF16CSupportedBySystem());
            }
        }
    }

    @Test
    public void testIsF16CSupportedBySystem_platformIsLinux_returnsTrue() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isMac).thenReturn(false);
            mockedPlatform.when(Platform::isLinux).thenReturn(true);

            try (MockedStatic<Files> mockedFiles = mockStatic(Files.class)) {
                mockedFiles.when(() -> Files.lines(Paths.get(LINUX_PROC_CPU_INFO)))
                    .thenReturn(
                        Stream.of(
                            "flags: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm",
                            "dummy string"
                        )
                    );
                assertTrue(PlatformUtils.isF16CSupportedBySystem());
            }
        }
    }

    @Test
    public void testIsF16CSupportedBySystem_platformIsLinux_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isMac).thenReturn(false);
            mockedPlatform.when(Platform::isLinux).thenReturn(true);

            try (MockedStatic<Files> mockedFiles = mockStatic(Files.class)) {
                mockedFiles.when(() -> Files.lines(Paths.get(LINUX_PROC_CPU_INFO)))
                    .thenReturn(
                        Stream.of(
                            "flags: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx",
                            "dummy string"
                        )
                    );
                assertFalse(PlatformUtils.isF16CSupportedBySystem());
            }
        }
    }

    @Test
    public void testIsF16CSupportedBySystem_platformIsLinux_throwsExceptionReturnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isMac).thenReturn(false);
            mockedPlatform.when(Platform::isLinux).thenReturn(true);

            try (MockedStatic<Files> mockedFiles = mockStatic(Files.class)) {
                mockedFiles.when(() -> Files.lines(Paths.get(LINUX_PROC_CPU_INFO))).thenThrow(RuntimeException.class);
                assertFalse(PlatformUtils.isF16CSupportedBySystem());
            }
        }
    }

    // SIMD AVX2 tests
    @Test
    public void testIsSIMDAVX2SupportedBySystem_bothAVX2AndF16CSupported_returnsTrue() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isMac).thenReturn(true);

            try (MockedStatic<SysctlUtil> mockedSysctlUtil = mockStatic(SysctlUtil.class)) {
                // Mock AVX2 support
                mockedSysctlUtil.when(() -> SysctlUtil.sysctl(MAC_CPU_FEATURES, "empty"))
                    .thenReturn(
                        "RDWRFSGS TSC_THREAD_OFFSET SGX BMI1 AVX2 SMEP BMI2 ERMS INVPCID FPU_CSDS MPX RDSEED ADX SMAP CLFSOPT IPT SGXLC MDCLEAR TSXFA IBRS STIBP L1DF ACAPMSR SSBD"
                    );
                // Mock F16C support
                mockedSysctlUtil.when(() -> SysctlUtil.sysctl("machdep.cpu.features", "empty"))
                    .thenReturn(
                        "FPU VME DE PSE TSC MSR PAE MCE CX8 APIC SEP MTRR PGE MCA CMOV PAT PSE36 CLFSH MMX FXSR SSE SSE2 HTT SSE3 PCLMULQDQ SSSE3 FMA CX16 SSE4.1 SSE4.2 x2APIC MOVBE POPCNT AES F16C RDRAND"
                    );

                assertTrue(PlatformUtils.isSIMDAVX2SupportedBySystem());
            }
        }
    }

    @Test
    public void testIsSIMDAVX2SupportedBySystem_AVX2SupportedButF16CNotSupported_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isMac).thenReturn(true);

            try (MockedStatic<SysctlUtil> mockedSysctlUtil = mockStatic(SysctlUtil.class)) {
                // Mock AVX2 support
                mockedSysctlUtil.when(() -> SysctlUtil.sysctl(MAC_CPU_FEATURES, "empty"))
                    .thenReturn(
                        "RDWRFSGS TSC_THREAD_OFFSET SGX BMI1 AVX2 SMEP BMI2 ERMS INVPCID FPU_CSDS MPX RDSEED ADX SMAP CLFSOPT IPT SGXLC MDCLEAR TSXFA IBRS STIBP L1DF ACAPMSR SSBD"
                    );
                // Mock F16C not supported
                mockedSysctlUtil.when(() -> SysctlUtil.sysctl("machdep.cpu.features", "empty"))
                    .thenReturn("FPU VME DE PSE TSC MSR PAE MCE CX8 APIC SEP MTRR PGE MCA CMOV PAT PSE36 CLFSH MMX FXSR SSE SSE2 HTT SSE3");

                assertFalse(PlatformUtils.isSIMDAVX2SupportedBySystem());
            }
        }
    }

    @Test
    public void testIsSIMDAVX2SupportedBySystem_bothAVX2AndF16CSupported_returnsTrue_linux() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isMac).thenReturn(false);
            mockedPlatform.when(Platform::isLinux).thenReturn(true);

            try (MockedStatic<Files> mockedFiles = mockStatic(Files.class)) {
                mockedFiles.when(() -> Files.lines(Paths.get(LINUX_PROC_CPU_INFO)))
                    .thenReturn(Stream.of("flags: AVX2 f16c", "dummy string"));
                assertTrue(PlatformUtils.isSIMDAVX2SupportedBySystem());
            }
        }
    }

    // SIMD AVX512 tests
    @Test
    public void testIsSIMDAVX512SupportedBySystem_bothAVX512AndF16CSupported_returnsTrue() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isLinux).thenReturn(true);

            try (MockedStatic<Files> mockedFiles = mockStatic(Files.class)) {
                mockedFiles.when(() -> Files.lines(Paths.get(LINUX_PROC_CPU_INFO)))
                    .thenReturn(Stream.of("flags: AVX2 avx512f avx512cd avx512vl avx512dq avx512bw f16c", "dummy string"));
                assertTrue(PlatformUtils.isSIMDAVX512SupportedBySystem());
            }
        }
    }

    @Test
    public void testIsSIMDAVX512SupportedBySystem_AVX512SupportedButF16CNotSupported_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isLinux).thenReturn(true);

            try (MockedStatic<Files> mockedFiles = mockStatic(Files.class)) {
                mockedFiles.when(() -> Files.lines(Paths.get(LINUX_PROC_CPU_INFO)))
                    .thenReturn(Stream.of("flags: AVX2 avx512f avx512cd avx512vl avx512dq avx512bw", "dummy string"));
                assertFalse(PlatformUtils.isSIMDAVX512SupportedBySystem());
            }
        }
    }

    // SIMD AVX512 SPR tests
    @Test
    public void testIsSIMDAVX512SPRSpecSupportedBySystem_bothAVX512SPRAndF16CSupported_returnsTrue() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isLinux).thenReturn(true);

            try (MockedStatic<Files> mockedFiles = mockStatic(Files.class)) {
                mockedFiles.when(() -> Files.lines(Paths.get(LINUX_PROC_CPU_INFO)))
                    .thenReturn(Stream.of("flags: avx512_fp16 avx512_bf16 avx512_vpopcntdq f16c", "dummy string"));
                assertTrue(PlatformUtils.isSIMDAVX512SPRSpecSupportedBySystem());
            }
        }
    }

    @Test
    public void testIsSIMDAVX512SPRSpecSupportedBySystem_AVX512SPRSupportedButF16CNotSupported_returnsFalse() {
        try (MockedStatic<Platform> mockedPlatform = mockStatic(Platform.class)) {
            mockedPlatform.when(Platform::isIntel).thenReturn(true);
            mockedPlatform.when(Platform::isLinux).thenReturn(true);

            try (MockedStatic<Files> mockedFiles = mockStatic(Files.class)) {
                mockedFiles.when(() -> Files.lines(Paths.get(LINUX_PROC_CPU_INFO)))
                    .thenReturn(Stream.of("flags: avx512_fp16 avx512_bf16 avx512_vpopcntdq", "dummy string"));
                assertFalse(PlatformUtils.isSIMDAVX512SPRSpecSupportedBySystem());
            }
        }
    }
}
