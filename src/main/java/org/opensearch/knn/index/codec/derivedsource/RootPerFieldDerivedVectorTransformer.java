/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.FieldInfo;
import org.opensearch.common.CheckedSupplier;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.io.IOException;

public class RootPerFieldDerivedVectorTransformer extends AbstractPerFieldDerivedVectorTransformer {

    private final FieldInfo fieldInfo;
    private final CheckedSupplier<KNNVectorValues<?>, IOException> vectorValuesSupplier;
    private KNNVectorValues<?> vectorValues;

    /**
     * Constructor for RootPerFieldDerivedVectorTransformer.
     *
     * @param fieldInfo FieldInfo for the field to create the injector for
     * @param derivedSourceReaders {@link DerivedSourceReaders} instance
     */
    public RootPerFieldDerivedVectorTransformer(FieldInfo fieldInfo, DerivedSourceReaders derivedSourceReaders) {
        this.fieldInfo = fieldInfo;
        this.vectorValuesSupplier = () -> KNNVectorValuesFactory.getVectorValues(
            fieldInfo,
            derivedSourceReaders.getDocValuesProducer(),
            derivedSourceReaders.getKnnVectorsReader()
        );
    }

    @Override
    public void setCurrentDoc(int offset, int docId) throws IOException {
        vectorValues = vectorValuesSupplier.get();
        vectorValues.advance(docId);
    }

    @Override
    public Object apply(Object object) {
        if (object == null) {
            return object;
        }

        try {
            return formatVector(fieldInfo, vectorValues::getVector, vectorValues::conditionalCloneVector);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
