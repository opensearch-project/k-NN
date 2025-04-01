/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.Term;
import org.opensearch.common.collect.Tuple;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.MediaType;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.index.mapper.ParsedDocument;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.DerivedKnnFloatVectorField;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class DerivedSourceIndexOperationListenerTests extends KNNTestCase {

    public void testPreIndex() throws Exception {
        String fieldName = "test-vector";
        int[] userVector = { 1, 2, 3, 4 };
        float[] backendVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        List<Double> expectedOutputAsList = new ArrayList<>(List.of(1.0, 2.0, 3.0, 4.0));

        Map<String, Object> originalSourceMap = Map.of(fieldName, userVector);
        BytesStreamOutput bStream = new BytesStreamOutput();
        XContentBuilder builder = MediaTypeRegistry.contentBuilder(XContentType.JSON, bStream).map(originalSourceMap);
        builder.close();
        BytesReference originalSource = bStream.bytes();

        ParseContext.Document document = new ParseContext.Document();
        document.add(new DerivedKnnFloatVectorField(fieldName, backendVector, true));

        Engine.Index operation = new Engine.Index(
            new Term("test-iud"),
            1,
            new ParsedDocument(null, null, null, null, List.of(document), originalSource, XContentType.JSON, null)
        );

        DerivedSourceIndexOperationListener derivedSourceIndexOperationListener = new DerivedSourceIndexOperationListener();
        operation = derivedSourceIndexOperationListener.preIndex(null, operation);
        Tuple<? extends MediaType, Map<String, Object>> modifiedSource = XContentHelper.convertToMap(
            operation.parsedDoc().source(),
            true,
            operation.parsedDoc().getMediaType()
        );

        assertEquals(expectedOutputAsList, modifiedSource.v2().get(fieldName));
    }
}
