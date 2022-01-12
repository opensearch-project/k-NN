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

package org.opensearch.knn.index.codec.util;

import java.io.ByteArrayInputStream;
import java.io.IOException;

/**
 * Interface abstracts the vector serializer object that is responsible for serialization and de-serialization of k-NN vector
 */
public interface KNNVectorSerializer {
    /**
     * Serializes array of floats to array of bytes
     * @param input array that will be converted
     * @return array of bytes that contains serialized input array
     * @throws Exception
     */
    byte[] floatToByteArray(float[] input) throws Exception;

    /**
     * Deserializes all bytes from the stream to array of floats
     * @param byteStream stream of bytes that will be used for deserialization to array of floats
     * @return array of floats deserialized from the stream
     * @throws IOException
     * @throws ClassNotFoundException
     */
    float[] byteToFloatArray(ByteArrayInputStream byteStream) throws IOException, ClassNotFoundException;
}
