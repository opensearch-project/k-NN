/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;

import java.util.List;

import static org.mockito.Mockito.mockStatic;

/**
 * Unit tests for {@link KNNLibraryLoader#variantCandidates(String)}. No native libraries are loaded;
 * CPU support and the avx*.disabled settings are stubbed via static mocks.
 */
public class KNNLibraryLoaderTests extends KNNTestCase {

    private static final String BASE = "opensearchknn_example";

    public void testVariantCandidates_whenAllVariantsPermitted_thenOrderedWidestFirstEndingWithPlain() {
        assertEquals(
            List.of(BASE + "_avx512_spr", BASE + "_avx512", BASE + "_avx2", BASE),
            candidates(true, true, true, false, false, false)
        );
    }

    public void testVariantCandidates_whenNoSimdSupport_thenOnlyPlainName() {
        assertEquals(List.of(BASE), candidates(false, false, false, false, false, false));
    }

    public void testVariantCandidates_whenVariantDisabledBySetting_thenExcludedEvenIfSupported() {
        assertEquals(List.of(BASE + "_avx512", BASE + "_avx2", BASE), candidates(true, true, true, true, false, false));
        assertEquals(List.of(BASE), candidates(true, true, true, true, true, true));
    }

    public void testVariantCandidates_whenAnyCombination_thenEndsWithPlainNameAndHasNoDuplicates() {
        for (int mask = 0; mask < 64; mask++) {
            final List<String> candidates = candidates(
                (mask & 1) != 0,
                (mask & 2) != 0,
                (mask & 4) != 0,
                (mask & 8) != 0,
                (mask & 16) != 0,
                (mask & 32) != 0
            );
            assertEquals(BASE, candidates.get(candidates.size() - 1));
            assertEquals(candidates.size(), candidates.stream().distinct().count());
        }
    }

    private static List<String> candidates(
        boolean sprSupported,
        boolean avx512Supported,
        boolean avx2Supported,
        boolean sprDisabled,
        boolean avx512Disabled,
        boolean avx2Disabled
    ) {
        try (
            MockedStatic<PlatformUtils> platform = mockStatic(PlatformUtils.class);
            MockedStatic<KNNSettings> settings = mockStatic(KNNSettings.class)
        ) {
            platform.when(PlatformUtils::isAVX512SPRSupportedBySystem).thenReturn(sprSupported);
            platform.when(PlatformUtils::isAVX512SupportedBySystem).thenReturn(avx512Supported);
            platform.when(PlatformUtils::isAVX2SupportedBySystem).thenReturn(avx2Supported);
            settings.when(KNNSettings::isFaissAVX512SPRDisabled).thenReturn(sprDisabled);
            settings.when(KNNSettings::isFaissAVX512Disabled).thenReturn(avx512Disabled);
            settings.when(KNNSettings::isFaissAVX2Disabled).thenReturn(avx2Disabled);
            return KNNLibraryLoader.variantCandidates(BASE);
        }
    }
}
