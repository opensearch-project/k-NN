/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.NormsProducer;
import org.apache.lucene.util.IOUtils;

import java.io.Closeable;
import java.io.IOException;

/**
 * Class holds the readers necessary to implement derived source. Important to note that if a segment does not have
 * any of these fields, the values will be null. Caller needs to check if these are null before using.
 */
@RequiredArgsConstructor
@Getter
public class DerivedSourceReaders implements Closeable {
    private final KnnVectorsReader knnVectorsReader;
    private final DocValuesProducer docValuesProducer;
    private final FieldsProducer fieldsProducer;
    private final NormsProducer normsProducer;

    @Override
    public void close() throws IOException {
        IOUtils.close(knnVectorsReader, docValuesProducer, fieldsProducer, normsProducer);
    }
}
