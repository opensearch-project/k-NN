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

import com.google.common.collect.ImmutableMap;
import org.opensearch.core.action.ActionListener;
import org.opensearch.action.admin.indices.create.CreateIndexRequestBuilder;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;

import java.io.IOException;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;

public class KNNCreateIndexFromModelTests extends KNNSingleNodeTestCase {

    public void testCreateIndexFromModel() throws IOException, InterruptedException {
        // This test confirms that we can create an index from cluster metadata

        // Create a model offline
        String modelId = "test-model";
        KNNEngine knnEngine = KNNEngine.FAISS;
        SpaceType spaceType = SpaceType.L2;
        int dimension = 3;

        // "Train" a faiss flat index - this really just creates an empty index that does brute force k-NN
        long vectorsPointer = JNIService.transferVectors(0, new float[0][0]);
        byte[] modelBlob = JNIService.trainIndex(
            ImmutableMap.of(INDEX_DESCRIPTION_PARAMETER, "Flat", SPACE_TYPE, spaceType.getValue()),
            dimension,
            vectorsPointer,
            KNNEngine.FAISS
        );

        // Setup model
        ModelMetadata modelMetadata = new ModelMetadata(
            knnEngine,
            spaceType,
            dimension,
            ModelState.CREATED,
            ZonedDateTime.now(ZoneOffset.UTC).toString(),
            "",
            "",
            "test-node",
            MethodComponentContext.EMPTY,
            VectorDataType.FLOAT
        );

        Model model = new Model(modelMetadata, modelBlob, modelId);

        // Check that an index can be created from the model.
        ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();

        final CountDownLatch inProgressLatch = new CountDownLatch(1);

        String indexName = "test-index";
        String fieldName = "test-field";

        final String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("model_id", modelId)
            .endObject()
            .endObject()
            .endObject()
            .toString();

        modelDao.put(model, ActionListener.wrap(indexResponse -> {
            CreateIndexRequestBuilder createIndexRequestBuilder = client().admin()
                .indices()
                .prepareCreate(indexName)
                .setSettings(Settings.builder().put("number_of_shards", 1).put("number_of_replicas", 0).put("index.knn", true).build())
                .setMapping(mapping);

            client().admin().indices().create(createIndexRequestBuilder.request(), ActionListener.wrap(createIndexResponse -> {
                assertTrue(createIndexResponse.isAcknowledged());
                inProgressLatch.countDown();
            }, e -> fail("Unable to create index: " + e.getMessage())));

        }, e -> fail("Unable to put model: " + e.getMessage())));

        assertTrue(inProgressLatch.await(20, TimeUnit.SECONDS));
    }
}
