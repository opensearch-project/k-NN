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

package org.opensearch.knn.index;

import java.io.File;

public class IndexUtil {

    private static Long BYTES_PER_KILOBYTES = 1024L;

    /**
     * Determines the size of a file on disk in kilobytes
     *
     * @param filePath path to the file
     * @return file size in kilobytes
     */
    public static long getFileSizeInKB(String filePath) {
        if (filePath == null || filePath.isEmpty()) {
            return 0;
        }
        File file = new File(filePath);
        if (!file.exists() || !file.isFile()) {
            return 0;
        }

        return (file.length() / BYTES_PER_KILOBYTES) + 1L; // Add one so that integer division rounds up
    }
}
