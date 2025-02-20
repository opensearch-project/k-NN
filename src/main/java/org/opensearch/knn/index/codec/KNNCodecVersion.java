/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import org.apache.lucene.codecs.Codec;
import org.opensearch.knn.index.codec.KNN10010Codec.KNN10010Codec;

/**
 * Class contains easy to access information about current default codec.
 */
public class KNNCodecVersion {
    public static final Codec CURRENT_DEFAULT = new KNN10010Codec();
    public static final Codec CURRENT_DEFAULT_DELEGATE = KNN10010Codec.DEFAULT_DELEGATE;
}
