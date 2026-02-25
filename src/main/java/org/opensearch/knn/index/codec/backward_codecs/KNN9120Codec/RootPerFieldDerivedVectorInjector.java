/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.backward_codecs.KNN9120Codec;

import org.apache.lucene.index.FieldInfo;
import org.opensearch.common.CheckedSupplier;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.io.IOException;
import java.util.Map;

/**
 * {@link PerFieldDerivedVectorInjector} for root fields (i.e. non nested fields).
 */
class RootPerFieldDerivedVectorInjector extends AbstractPerFieldDerivedVectorInjector {

    private final FieldInfo fieldInfo;
    private final CheckedSupplier<KNNVectorValues<?>, IOException> vectorValuesSupplier;

    /**
     * Constructor for RootPerFieldDerivedVectorInjector.
     *
     * @param fieldInfo FieldInfo for the field to create the injector for
     * @param derivedSourceReaders {@link KNN9120DerivedSourceReaders} instance
     */
    public RootPerFieldDerivedVectorInjector(FieldInfo fieldInfo, KNN9120DerivedSourceReaders derivedSourceReaders) {
        this.fieldInfo = fieldInfo;
        this.vectorValuesSupplier = () -> KNNVectorValuesFactory.getVectorValues(
            fieldInfo,
            derivedSourceReaders.getDocValuesProducer(),
            derivedSourceReaders.getKnnVectorsReader()
        );
    }

    @Override
    public void inject(int docId, Map<String, Object> sourceAsMap) throws IOException {
        KNNVectorValues<?> vectorValues = vectorValuesSupplier.get();
        if (vectorValues.docId() == docId || vectorValues.advance(docId) == docId) {
            sourceAsMap.put(fieldInfo.name, formatVector(fieldInfo, vectorValues));
        }
    }
}
