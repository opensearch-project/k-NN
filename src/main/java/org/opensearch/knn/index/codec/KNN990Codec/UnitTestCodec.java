/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNNCodecVersion;

/**
 * This codec is for testing. The reason for putting this codec here is SPI is only scanning the src folder and not
 * able to pick this class if its in test folder. Don't use this codec outside of testing
 */
public class UnitTestCodec extends FilterCodec {
    public UnitTestCodec() {
        super("UnitTestCodec", KNNCodecVersion.current().getDefaultKnnCodecSupplier().get());
    }

    @Override
    public KnnVectorsFormat knnVectorsFormat() {
        return new PerFieldKnnVectorsFormat() {
            @Override
            public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                return new NativeEngines990KnnVectorsFormat();
            }
        };
    }
}
