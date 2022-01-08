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

import java.io.ByteArrayInputStream;
import java.io.IOException;

/**
 * Interface abstracts the vector serializer object that is responsible for serialization and de-serialization of k-NN vector
 */
public interface VectorSerializer {
    byte[] floatToByte(float[] floats) throws Exception;

    float[] byteToFloat(ByteArrayInputStream byteStream) throws IOException, ClassNotFoundException;
}
