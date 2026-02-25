/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.util;

import org.apache.lucene.util.BytesRef;

/**
 * Interface abstracts the vector serializer object that is responsible for serialization and de-serialization of k-NN vector
 */
public interface KNNVectorSerializer {
    /**
     * Serializes array of floats to array of bytes
     * @param input array that will be converted
     * @return array of bytes that contains serialized input array
     */
    byte[] floatToByteArray(float[] input);

    /**
     * Deserializes all bytes from the stream to array of floats
     *
     * @param bytesRef bytes that will be used for deserialization to array of floats
     * @return array of floats deserialized from the stream
     */
    float[] byteToFloatArray(BytesRef bytesRef);
}
