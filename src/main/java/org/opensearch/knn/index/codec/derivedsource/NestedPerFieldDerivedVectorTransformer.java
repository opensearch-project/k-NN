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

public class NestedPerFieldDerivedVectorTransformer extends AbstractPerFieldDerivedVectorTransformer {

    private final FieldInfo childFieldInfo;
    private final DerivedSourceReaders derivedSourceReaders;
    private final CheckedSupplier<NumericDocValues, IOException> normValuesSupplier;
    private KNNVectorValues<?> vectorValues;
    private NumericDocValues normDocValues;

    /**
     *
     * @param childFieldInfo FieldInfo of the child field
     * @param derivedSourceReaders Readers for access segment info
     * @param normFieldInfo FieldInfo for the norm field, or null if no denormalization is needed
     */
    public NestedPerFieldDerivedVectorTransformer(
        FieldInfo childFieldInfo,
        DerivedSourceReaders derivedSourceReaders,
        FieldInfo normFieldInfo
    ) {
        this.childFieldInfo = childFieldInfo;
        this.derivedSourceReaders = derivedSourceReaders;
        this.normValuesSupplier = normFieldInfo != null
            ? () -> derivedSourceReaders.getDocValuesProducer().getNumeric(normFieldInfo)
            : null;
    }

    @Override
    public Object apply(Object object) {
        if (object == null) {
            return object;
        }

        try {
            float norm = (normDocValues != null) ? Float.intBitsToFloat((int) normDocValues.longValue()) : 1.0f;
            Object vector = formatVector(childFieldInfo, vectorValues::getVector, vectorValues::conditionalCloneVector, norm);
            vectorValues.nextDoc();
            if (normDocValues != null) {
                normDocValues.nextDoc();
            }
            return vector;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void setCurrentDoc(int offset, int docId) throws IOException {
        vectorValues = KNNVectorValuesFactory.getVectorValues(
            childFieldInfo,
            derivedSourceReaders.getDocValuesProducer(),
            derivedSourceReaders.getKnnVectorsReader()
        );
        vectorValues.advance(offset);
        if (normValuesSupplier != null) {
            normDocValues = normValuesSupplier.get();
            normDocValues.advance(offset);
        }
    }
}
