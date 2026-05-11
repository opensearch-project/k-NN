/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN10010Codec;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;

public class DerivedSourceStoredFieldsFormatTests extends KNNTestCase {

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

    public void testGetFieldInfoDoesNotGuessWhenCaseInsensitiveMatchHasNoVectorHints() {
        FieldInfo mixedCaseFieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("vectorSearch.nameVector").fieldNumber(1).build();
        FieldInfo lowerCaseFieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder("vectorSearch.namevector").fieldNumber(2).build();
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { mixedCaseFieldInfo, lowerCaseFieldInfo });

        assertNull(KNN10010DerivedSourceStoredFieldsFormat.getFieldInfo(fieldInfos, "vectorsearch.namevector"));
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
}
