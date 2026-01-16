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

package org.opensearch.knn.index.util;

import junit.framework.TestCase;
import org.junit.Assert;
import org.mockito.MockedStatic;
import org.opensearch.Version;
import org.opensearch.knn.index.KNNSettings;

import java.util.Map;

import static org.mockito.Mockito.mockStatic;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;

public class IndexHyperParametersUtilTests extends TestCase {

    public void testLombokNonNull() {
        Assert.assertThrows(NullPointerException.class, () -> IndexHyperParametersUtil.getHNSWEFConstructionValue(null));
        Assert.assertThrows(NullPointerException.class, () -> IndexHyperParametersUtil.getHNSWEFSearchValue(null));
    }

    public void testGetHNSWEFConstructionValue_withDifferentValues_thenSuccess() {
        Assert.assertEquals(512, IndexHyperParametersUtil.getHNSWEFConstructionValue(Version.V_2_11_0));
        Assert.assertEquals(512, IndexHyperParametersUtil.getHNSWEFConstructionValue(Version.V_2_3_0));
        Assert.assertEquals(
            KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION.intValue(),
            IndexHyperParametersUtil.getHNSWEFConstructionValue(Version.CURRENT)
        );

    }

    public void testGetHNSWEFSearchValue_withDifferentValues_thenSuccess() {
        Assert.assertEquals(512, IndexHyperParametersUtil.getHNSWEFConstructionValue(Version.V_2_11_0));
        Assert.assertEquals(512, IndexHyperParametersUtil.getHNSWEFConstructionValue(Version.V_2_3_0));
        Assert.assertEquals(
            KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH.intValue(),
            IndexHyperParametersUtil.getHNSWEFConstructionValue(Version.CURRENT)
        );
    }

    public void testGetBinaryQuantizationEFValues_thenSuccess() {
        // Test for Binary Quantization EF Construction value
        Assert.assertEquals(256, IndexHyperParametersUtil.getBinaryQuantizationEFConstructionValue());

        // Test for Binary Quantization EF Search value
        Assert.assertEquals(256, IndexHyperParametersUtil.getBinaryQuantizationEFSearchValue());
    }

    public void testGetHNSWEFSearchValue_withMethodParameters_thenReturnMethodParameterValue() {
        String indexName = "test-index";
        int efSearchValue = 100;
        Map<String, Object> methodParameters = Map.of(METHOD_PARAMETER_EF_SEARCH, efSearchValue);

        int result = IndexHyperParametersUtil.getHNSWEFSearchValue(methodParameters, indexName);

        Assert.assertEquals(efSearchValue, result);
    }

    public void testGetHNSWEFSearchValue_withNullMethodParameters_thenReturnIndexSetting() {
        String indexName = "test-index";
        int expectedValue = 150;

        try (MockedStatic<KNNSettings> knnSettingsMock = mockStatic(KNNSettings.class)) {
            knnSettingsMock.when(() -> KNNSettings.getEfSearchParam(indexName)).thenReturn(expectedValue);

            int result = IndexHyperParametersUtil.getHNSWEFSearchValue(null, indexName);

            Assert.assertEquals(expectedValue, result);
        }
    }

    public void testGetHNSWEFSearchValue_withEmptyMethodParameters_thenReturnIndexSetting() {
        String indexName = "test-index";
        int expectedValue = 200;
        Map<String, Object> methodParameters = Map.of();

        try (MockedStatic<KNNSettings> knnSettingsMock = mockStatic(KNNSettings.class)) {
            knnSettingsMock.when(() -> KNNSettings.getEfSearchParam(indexName)).thenReturn(expectedValue);

            int result = IndexHyperParametersUtil.getHNSWEFSearchValue(methodParameters, indexName);

            Assert.assertEquals(expectedValue, result);
        }
    }
}
