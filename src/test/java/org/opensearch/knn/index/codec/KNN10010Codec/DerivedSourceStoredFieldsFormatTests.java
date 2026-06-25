/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN10010Codec;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

public class DerivedSourceStoredFieldsFormatTests extends KNNTestCase {
    private static final String NMSLIB_ENGINE_NAME = "nmslib";

    public void testGetFieldInfoReturnsExactMatch() {
        FieldInfo exactFieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("vectorSearch.nameVector").fieldNumber(1).build();
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { exactFieldInfo });

        assertSame(exactFieldInfo, KNN10010DerivedSourceStoredFieldsFormat.getFieldInfo(fieldInfos, "vectorSearch.nameVector"));
    }

    public void testGetFieldInfoFallsBackToUnambiguousCaseInsensitiveMatch() {
        FieldInfo mixedCaseFieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("vectorSearch.nameVector").fieldNumber(1).build();
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { mixedCaseFieldInfo });

        assertSame(mixedCaseFieldInfo, KNN10010DerivedSourceStoredFieldsFormat.getFieldInfo(fieldInfos, "vectorsearch.namevector"));
    }

    public void testGetFieldInfoDoesNotGuessWhenCaseInsensitiveMatchIsAmbiguous() {
        FieldInfo mixedCaseFieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("vectorSearch.nameVector")
            .fieldNumber(1)
            .vectorDimension(16)
            .build();
        FieldInfo lowerCaseFieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("vectorSearch.namevector")
            .fieldNumber(2)
            .vectorDimension(16)
            .build();
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { mixedCaseFieldInfo, lowerCaseFieldInfo });

        assertNull(KNN10010DerivedSourceStoredFieldsFormat.getFieldInfo(fieldInfos, "vectorsearch.namevector"));
    }

    public void testGetFieldInfoFallsBackToFirstCaseInsensitiveMatchWhenNoVectorHints() {
        FieldInfo mixedCaseFieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("vectorSearch.nameVector").fieldNumber(1).build();
        FieldInfo lowerCaseFieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("vectorSearch.namevector").fieldNumber(2).build();
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { mixedCaseFieldInfo, lowerCaseFieldInfo });

        assertSame(mixedCaseFieldInfo, KNN10010DerivedSourceStoredFieldsFormat.getFieldInfo(fieldInfos, "vectorsearch.namevector"));
    }

    public void testGetFieldInfoPrefersVectorFieldWhenCaseInsensitiveMatchHasSingleVectorField() {
        FieldInfo nonVectorFieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("vectorSearch.namevector").fieldNumber(1).build();
        FieldInfo vectorFieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("vectorSearch.nameVector")
            .fieldNumber(2)
            .vectorDimension(16)
            .build();
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { nonVectorFieldInfo, vectorFieldInfo });

        assertSame(vectorFieldInfo, KNN10010DerivedSourceStoredFieldsFormat.getFieldInfo(fieldInfos, "vectorsearch.namevector"));
    }

    public void testGetFieldInfoPrefersNativeFaissVectorFieldWhenCaseInsensitiveMatchHasSingleVectorField() {
        assertNativeVectorFieldPreferred(FAISS_NAME);
    }

    public void testGetFieldInfoPrefersNativeNmslibVectorFieldWhenCaseInsensitiveMatchHasSingleVectorField() {
        assertNativeVectorFieldPreferred(NMSLIB_ENGINE_NAME);
    }

    private void assertNativeVectorFieldPreferred(String engineName) {
        FieldInfo nonVectorFieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("vectorSearch.namevector").fieldNumber(1).build();
        FieldInfo nativeVectorFieldInfo = nativeVectorFieldInfo(engineName);
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { nonVectorFieldInfo, nativeVectorFieldInfo });

        assertTrue(nativeVectorFieldInfo.hasVectorValues());
        assertSame(nativeVectorFieldInfo, KNN10010DerivedSourceStoredFieldsFormat.getFieldInfo(fieldInfos, "vectorsearch.namevector"));
    }

    private FieldInfo nativeVectorFieldInfo(String engineName) {
        return KNNCodecTestUtil.FieldInfoBuilder.builder("vectorSearch.nameVector")
            .fieldNumber(2)
            .addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true")
            .addAttribute(KNN_METHOD, METHOD_HNSW)
            .addAttribute(KNN_ENGINE, engineName)
            .addAttribute(VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue())
            .vectorDimension(16)
            .build();
    }
}
