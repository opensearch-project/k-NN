/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.NormsProducer;
import org.apache.lucene.index.SegmentReadState;

import java.io.IOException;

/**
 * Class encapsulates the suppliers to give the {@link DerivedSourceReaders} from particular formats needed to implement
 * derived source. More specifically, given a {@link org.apache.lucene.index.SegmentReadState}, this class will provide
 * the correct format reader for that segment.
 */
@RequiredArgsConstructor
public class DerivedSourceReadersSupplier {
    private final DerivedSourceReaderSupplier<KnnVectorsReader> knnVectorsReaderSupplier;
    private final DerivedSourceReaderSupplier<DocValuesProducer> docValuesProducerSupplier;
    private final DerivedSourceReaderSupplier<FieldsProducer> fieldsProducerSupplier;
    private final DerivedSourceReaderSupplier<NormsProducer> normsProducer;

    /**
     * Get the readers for the segment
     *
     * @param state SegmentReadState
     * @return DerivedSourceReaders
     * @throws IOException in case of I/O error
     */
    public DerivedSourceReaders getReaders(SegmentReadState state) throws IOException {
        return new DerivedSourceReaders(
            knnVectorsReaderSupplier.apply(state),
            docValuesProducerSupplier.apply(state),
            fieldsProducerSupplier.apply(state),
            normsProducer.apply(state)
        );
    }
}
