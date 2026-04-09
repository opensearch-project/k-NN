/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.FieldInfo;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.io.IOException;

public class NestedPerFieldDerivedVectorTransformer extends AbstractPerFieldDerivedVectorTransformer {

    private final FieldInfo childFieldInfo;
    private final DerivedSourceReaders derivedSourceReaders;
    private final DerivedSourceNormSupplier normSupplier;
    private KNNVectorValues<?> vectorValues;
    private int currentOffset;

    /**
     *
     * @param childFieldInfo FieldInfo of the child field
     * @param derivedSourceReaders Readers for access segment info
     * @param normSupplier supplier for L2 norm values
     */
    public NestedPerFieldDerivedVectorTransformer(
        FieldInfo childFieldInfo,
        DerivedSourceReaders derivedSourceReaders,
        DerivedSourceNormSupplier normSupplier
    ) {
        this.childFieldInfo = childFieldInfo;
        this.derivedSourceReaders = derivedSourceReaders;
        this.normSupplier = normSupplier;
    }

    @Override
    public Object apply(Object object) {
        if (object == null) {
            return object;
        }

        try {
            Object vector = formatVector(
                childFieldInfo,
                vectorValues::getVector,
                vectorValues::conditionalCloneVector,
                () -> normSupplier.getNorm(currentOffset)
            );
            vectorValues.nextDoc();
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
        currentOffset = offset;
    }
}
