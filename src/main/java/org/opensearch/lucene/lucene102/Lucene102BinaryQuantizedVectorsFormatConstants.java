/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.lucene.lucene102;

import lombok.experimental.UtilityClass;

@UtilityClass
public class Lucene102BinaryQuantizedVectorsFormatConstants {
    public static final byte QUERY_BITS = 4;

    public static final String NAME = "Lucene102BinaryQuantizedVectorsFormat";

    static final int VERSION_START = 0;
    static final int VERSION_CURRENT = VERSION_START;
    static final String META_CODEC_NAME = "Lucene102BinaryQuantizedVectorsFormatMeta";
    static final String VECTOR_DATA_CODEC_NAME = "Lucene102BinaryQuantizedVectorsFormatData";
    static final String META_EXTENSION = "vemb";
    static final String VECTOR_DATA_EXTENSION = "veb";
}
