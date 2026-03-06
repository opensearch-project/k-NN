/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import org.apache.lucene.codecs.Codec;
import org.opensearch.knn.index.codec.KNN1040Codec.KNN1040Codec;

/**
 * Class contains easy to access information about current default codec.
 */
public class KNNCodecVersion {
    public static final Codec CURRENT_DEFAULT = new KNN1040Codec();
    public static final Codec CURRENT_DEFAULT_DELEGATE = KNN1040Codec.DEFAULT_DELEGATE;
}
