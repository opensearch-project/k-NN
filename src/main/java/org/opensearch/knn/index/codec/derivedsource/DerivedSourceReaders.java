/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.NormsProducer;
import org.apache.lucene.util.IOUtils;
import org.opensearch.common.Nullable;

import java.io.Closeable;
import java.io.IOException;

/**
 * Class holds the readers necessary to implement derived source. Important to note that if a segment does not have
 * any of these fields, the values will be null. Caller needs to check if these are null before using.
 */
@RequiredArgsConstructor
@Getter
@Log4j2
public class DerivedSourceReaders implements Closeable {
    @Nullable
    private final KnnVectorsReader knnVectorsReader;
    @Nullable
    private final DocValuesProducer docValuesProducer;
    @Nullable
    private final FieldsProducer fieldsProducer;
    @Nullable
    private final NormsProducer normsProducer;

    @Override
    public void close() throws IOException {
        log.info("Closing derived source readers");
        IOUtils.close(knnVectorsReader, docValuesProducer, fieldsProducer, normsProducer);
    }
}
