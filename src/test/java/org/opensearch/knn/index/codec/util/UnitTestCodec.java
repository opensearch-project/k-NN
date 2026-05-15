/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNNCodecVersion;

import java.util.function.Supplier;

/**
 * This codec is for testing. The reason for putting this codec here is SPI is only scanning the src folder and not
 * able to pick this class if its in test folder. Don't use this codec outside of testing
 */
public class UnitTestCodec extends FilterCodec {
    private final Supplier<KnnVectorsFormat> knnVectorsFormatSupplier;

    public UnitTestCodec() {
        super("UnitTestCodec", KNNCodecVersion.CURRENT_DEFAULT);
        this.knnVectorsFormatSupplier = KNNCodecVersion.CURRENT_DEFAULT::knnVectorsFormat;
    }

    public UnitTestCodec(Supplier<KnnVectorsFormat> knnVectorsFormatSupplier) {
        super("UnitTestCodec", KNNCodecVersion.CURRENT_DEFAULT);
        this.knnVectorsFormatSupplier = knnVectorsFormatSupplier;
    }

    @Override
    public KnnVectorsFormat knnVectorsFormat() {
        return new PerFieldKnnVectorsFormat() {
            @Override
            public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                return knnVectorsFormatSupplier.get();
            }
        };
    }
}
