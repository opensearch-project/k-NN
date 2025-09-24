/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import org.apache.lucene.codecs.Codec;
import org.opensearch.knn.index.codec.KNN1030Codec.KNN1030Codec;

/**
 * Class contains easy to access information about current default codec.
 */
public class KNNCodecVersion {
    public static final Codec CURRENT_DEFAULT = new KNN1030Codec();
    public static final Codec CURRENT_DEFAULT_DELEGATE = KNN1030Codec.DEFAULT_DELEGATE;
}
