/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.FieldInfo;
import org.opensearch.knn.index.mapper.VectorTransformer;

public class PerFieldDerivedVectorTransformerFactory {

    /**
     * Create a {@link PerFieldDerivedVectorTransformer} instance based on information in field info.
     *
     * @param fieldInfo FieldInfo for the field to create the injector for
     * @param isNested whether the field is nested
     * @param derivedSourceReaders {@link DerivedSourceReaders} instance
     * @param vectorTransformer VectorTransformer for undoing transformations
     * @return PerFieldDerivedVectorInjector instance
     */
    public static PerFieldDerivedVectorTransformer create(
        FieldInfo fieldInfo,
        boolean isNested,
        DerivedSourceReaders derivedSourceReaders,
        VectorTransformer vectorTransformer
    ) {
        // Nested case
        if (isNested) {
            return new NestedPerFieldDerivedVectorTransformer(fieldInfo, derivedSourceReaders, vectorTransformer);
        }

        // Non-nested case
        return new RootPerFieldDerivedVectorTransformer(fieldInfo, derivedSourceReaders, vectorTransformer);
    }

}
