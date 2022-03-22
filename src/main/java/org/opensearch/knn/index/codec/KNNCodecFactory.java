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

package org.opensearch.knn.index.codec;

import org.apache.lucene.codecs.Codec;
import org.opensearch.knn.index.codec.KNN91Codec.KNN91Codec;

/**
 * Factory abstraction for KNN codes
 */
public class KNNCodecFactory {

    public static Codec createKNN91Codec(Codec userCodec) {
        return new KNN91Codec(userCodec);
    }
}
