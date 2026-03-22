/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.NumericDocValues;
import org.opensearch.common.CheckedSupplier;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.io.IOException;

public class RootPerFieldDerivedVectorTransformer extends AbstractPerFieldDerivedVectorTransformer {

    private final FieldInfo fieldInfo;
    private final CheckedSupplier<KNNVectorValues<?>, IOException> vectorValuesSupplier;
    private final CheckedSupplier<NumericDocValues, IOException> normValuesSupplier;
    private KNNVectorValues<?> vectorValues;
    private NumericDocValues normDocValues;

    /**
     * Constructor for RootPerFieldDerivedVectorTransformer.
     *
     * @param fieldInfo FieldInfo for the field to create the injector for
     * @param derivedSourceReaders {@link DerivedSourceReaders} instance
     * @param normFieldInfo FieldInfo for the norm field, or null if no denormalization is needed
     */
    public RootPerFieldDerivedVectorTransformer(
        FieldInfo fieldInfo,
        DerivedSourceReaders derivedSourceReaders,
        FieldInfo normFieldInfo
    ) {
        this.fieldInfo = fieldInfo;
        this.vectorValuesSupplier = () -> KNNVectorValuesFactory.getVectorValues(
            fieldInfo,
            derivedSourceReaders.getDocValuesProducer(),
            derivedSourceReaders.getKnnVectorsReader()
        );
        this.normValuesSupplier = normFieldInfo != null
            ? () -> derivedSourceReaders.getDocValuesProducer().getNumeric(normFieldInfo)
            : null;
    }

    @Override
    public void setCurrentDoc(int offset, int docId) throws IOException {
        vectorValues = vectorValuesSupplier.get();
        vectorValues.advance(docId);
        if (normValuesSupplier != null) {
            normDocValues = normValuesSupplier.get();
            normDocValues.advance(docId);
        }
    }

    @Override
    public Object apply(Object object) {
        if (object == null) {
            return object;
        }

        try {
            float norm = (normDocValues != null) ? Float.intBitsToFloat((int) normDocValues.longValue()) : 1.0f;
            return formatVector(fieldInfo, vectorValues::getVector, vectorValues::conditionalCloneVector, norm);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
