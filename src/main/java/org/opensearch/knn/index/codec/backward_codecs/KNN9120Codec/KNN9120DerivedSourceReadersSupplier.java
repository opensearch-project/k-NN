/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.backward_codecs.KNN9120Codec;

import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.NormsProducer;
import org.apache.lucene.index.SegmentReadState;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceReaderSupplier;
import org.opensearch.knn.index.codec.derivedsource.DerivedSourceReaders;

import java.io.IOException;

/**
 * Class encapsulates the suppliers to give the {@link DerivedSourceReaders} from particular formats needed to implement
 * derived source. More specifically, given a {@link org.apache.lucene.index.SegmentReadState}, this class will provide
 * the correct format reader for that segment.
 */
@RequiredArgsConstructor
public class KNN9120DerivedSourceReadersSupplier {
    @NonNull
    private final DerivedSourceReaderSupplier<KnnVectorsReader> knnVectorsReaderSupplier;
    @NonNull
    private final DerivedSourceReaderSupplier<DocValuesProducer> docValuesProducerSupplier;
    @NonNull
    private final DerivedSourceReaderSupplier<FieldsProducer> fieldsProducerSupplier;
    @NonNull
    private final DerivedSourceReaderSupplier<NormsProducer> normsProducer;

    /**
     * Get the readers for the segment
     *
     * @param state SegmentReadState
     * @return DerivedSourceReaders
     * @throws IOException in case of I/O error
     */
    public KNN9120DerivedSourceReaders getReaders(SegmentReadState state) throws IOException {
        return new KNN9120DerivedSourceReaders(
            knnVectorsReaderSupplier.apply(state),
            docValuesProducerSupplier.apply(state),
            fieldsProducerSupplier.apply(state),
            normsProducer.apply(state)
        );
    }
}
