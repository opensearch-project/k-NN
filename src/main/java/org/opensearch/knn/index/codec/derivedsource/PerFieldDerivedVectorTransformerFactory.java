/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.opensearch.knn.common.KNNVectorUtil;

public class PerFieldDerivedVectorTransformerFactory {

    /**
     * Create a {@link PerFieldDerivedVectorTransformer} instance based on information in field info.
     *
     * @param fieldInfo FieldInfo for the field to create the injector for
     * @param isNested whether the field is nested
     * @param derivedSourceReaders {@link DerivedSourceReaders} instance
     * @param fieldInfos FieldInfos to look up the norm field
     * @return PerFieldDerivedVectorTransformer instance
     */
    public static PerFieldDerivedVectorTransformer create(
        FieldInfo fieldInfo,
        boolean isNested,
        DerivedSourceReaders derivedSourceReaders,
        FieldInfos fieldInfos
    ) {
        FieldInfo normFieldInfo = fieldInfos.fieldInfo(KNNVectorUtil.getNormFieldName(fieldInfo.name));
        DerivedSourceNormSupplier normSupplier = normFieldInfo != null
            ? DerivedSourceNormSupplier.fromDocValues(() -> derivedSourceReaders.getDocValuesProducer().getNumeric(normFieldInfo))
            : DerivedSourceNormSupplier.UNIT;

        if (isNested) {
            return new NestedPerFieldDerivedVectorTransformer(fieldInfo, derivedSourceReaders, normSupplier);
        }
        return new RootPerFieldDerivedVectorTransformer(fieldInfo, derivedSourceReaders, normSupplier);
    }
}
