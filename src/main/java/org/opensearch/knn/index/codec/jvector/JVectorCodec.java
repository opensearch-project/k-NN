/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene912.Lucene912Codec;

public class JVectorCodec extends FilterCodec {

    public static final String CODEC_NAME = "JVectorCodec";

    public JVectorCodec() {
        super(CODEC_NAME, new Lucene912Codec());
    }

    @Override
    public KnnVectorsFormat knnVectorsFormat() {
        return new JVectorFormat();
    }

    @Override
    public CompoundFormat compoundFormat() {
        return new JVectorCompoundFormat(delegate.compoundFormat());
    }
}
