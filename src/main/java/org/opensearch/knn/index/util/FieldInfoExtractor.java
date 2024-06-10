/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import org.apache.lucene.index.FieldInfo;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.knn.common.KNNConstants;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;

/**
 * Class having methods to extract a value from field info
 */
public class FieldInfoExtractor {
    public static String getIndexDescription(FieldInfo fieldInfo) throws IOException {
        String parameters = fieldInfo.attributes().get(KNNConstants.PARAMETERS);
        if (parameters == null) {
            return null;
        }

        return (String) XContentHelper.createParser(
            NamedXContentRegistry.EMPTY,
            DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
            new BytesArray(parameters),
            MediaTypeRegistry.getDefaultMediaType()
        ).map().getOrDefault(INDEX_DESCRIPTION_PARAMETER, null);
    }
}
