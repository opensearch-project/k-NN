/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.backward_codecs.KNN9120Codec;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentReadState;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceReaders;

/**
 * Factory for creating {@link PerFieldDerivedVectorInjector} instances.
 */
class PerFieldDerivedVectorInjectorFactory {

    /**
     * Create a {@link PerFieldDerivedVectorInjector} instance based on information in field info.
     *
     * @param fieldInfo FieldInfo for the field to create the injector for
     * @param derivedSourceReaders {@link DerivedSourceReaders} instance
     * @return PerFieldDerivedVectorInjector instance
     */
    public static PerFieldDerivedVectorInjector create(
        FieldInfo fieldInfo,
        KNN9120DerivedSourceReaders derivedSourceReaders,
        SegmentReadState segmentReadState
    ) {
        // Nested case
        if (ParentChildHelper.getParentField(fieldInfo.name) != null) {
            return new NestedPerFieldDerivedVectorInjector(fieldInfo, derivedSourceReaders, segmentReadState);
        }

        // Non-nested case
        return new RootPerFieldDerivedVectorInjector(fieldInfo, derivedSourceReaders);
    }
}
