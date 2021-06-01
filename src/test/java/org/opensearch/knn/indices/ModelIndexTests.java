/*
 *   Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License").
 *   You may not use this file except in compliance with the License.
 *   A copy of the License is located at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   or in the "license" file accompanying this file. This file is distributed
 *   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *   express or implied. See the License for the specific language governing
 *   permissions and limitations under the License.
 */

package org.opensearch.knn.indices;

import org.opensearch.ExceptionsHelper;
import org.opensearch.ResourceAlreadyExistsException;
import org.opensearch.action.ActionListener;
import org.opensearch.action.admin.indices.create.CreateIndexResponse;
import org.opensearch.action.delete.DeleteResponse;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.index.engine.VersionConflictEngineException;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.rest.RestStatus;

import java.io.IOException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

public class ModelIndexTests extends KNNSingleNodeTestCase {

    public void testCreateModelIndex() throws IOException, InterruptedException {
        int attempts = 50;
        final CountDownLatch inProgressLatch = new CountDownLatch(attempts);

        ActionListener<CreateIndexResponse> indexCreationListener = ActionListener.wrap(response -> {
            assertTrue(response.isAcknowledged());
            inProgressLatch.countDown();
        }, exception -> {
            if (!(ExceptionsHelper.unwrapCause(exception) instanceof ResourceAlreadyExistsException)) {
                fail("Failed for reason other than ResourceAlreadyExistsException: " + exception);
            }
            inProgressLatch.countDown();
        });

        ModelIndex modelIndex = ModelIndex.getInstance();

        for (int i = 0; i < attempts; i++) {
            modelIndex.createModelIndex(indexCreationListener);
        }

        assertTrue(inProgressLatch.await(30, TimeUnit.SECONDS));
    }

    public void testModelIndexExists() throws IOException, InterruptedException {
        ModelIndex modelIndex = ModelIndex.getInstance();

        assertFalse(modelIndex.modelIndexExists());

        final CountDownLatch inProgressLatch = new CountDownLatch(1);

        ActionListener<CreateIndexResponse> indexCreationListener = ActionListener.wrap(response -> {
            assertTrue(response.isAcknowledged());
            inProgressLatch.countDown();
        }, exception -> fail("Model index unable to create"));

        modelIndex.createModelIndex(indexCreationListener);

        assertTrue(inProgressLatch.await(30, TimeUnit.SECONDS));

        assertTrue(modelIndex.modelIndexExists());
    }

    public void testPutModel() throws IOException, InterruptedException {
        ModelIndex modelIndex = ModelIndex.getInstance();
        String modelId = "efbsdhcvbsd";
        byte [] modelBlob = "hello".getBytes();

        // User provided model id
        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);

        ActionListener<CreateIndexResponse> indexCreationListener = ActionListener.wrap(response -> {
            assertTrue(response.isAcknowledged());
            inProgressLatch1.countDown();
        }, exception -> {
            if (!(ExceptionsHelper.unwrapCause(exception) instanceof ResourceAlreadyExistsException)) {
                fail("Failed during create: " + exception);
            }
            inProgressLatch1.countDown();
        });

        modelIndex.createModelIndex(indexCreationListener);
        assertTrue(inProgressLatch1.await(30, TimeUnit.SECONDS));

        final CountDownLatch inProgressLatch2 = new CountDownLatch(1);
        ActionListener<IndexResponse> docCreationListener = ActionListener.wrap(response -> {
            assertEquals(RestStatus.CREATED, response.status());
            assertEquals(modelId, response.getId());
            inProgressLatch2.countDown();
        }, exception -> fail("Unable to put the model: " + exception));

        modelIndex.putModel(modelId, KNNEngine.DEFAULT, modelBlob, docCreationListener);

        assertTrue(inProgressLatch2.await(30, TimeUnit.SECONDS));

        // User provided model id that already exists
        final CountDownLatch inProgressLatch3 = new CountDownLatch(1);
        ActionListener<IndexResponse> docCreationListenerDuplicateId = ActionListener.wrap(
                response -> fail("Model already exists, but creation was successful"),
                exception -> {
                    if (!(ExceptionsHelper.unwrapCause(exception) instanceof VersionConflictEngineException)) {
                        fail("Unable to put the model: " + exception);
                    }
                    inProgressLatch3.countDown();
        });

        modelIndex.putModel(modelId, KNNEngine.DEFAULT, modelBlob, docCreationListenerDuplicateId);
        assertTrue(inProgressLatch3.await(30, TimeUnit.SECONDS));

        // User does not provide model id
        final CountDownLatch inProgressLatch4 = new CountDownLatch(1);
        ActionListener<IndexResponse> docCreationListenerNoModelId = ActionListener.wrap(response -> {
                    assertEquals(RestStatus.CREATED, response.status());
                    inProgressLatch4.countDown();
                },
                exception -> fail("Unable to put the model: " + exception));

        modelIndex.putModel(KNNEngine.DEFAULT, modelBlob, docCreationListenerNoModelId);
        assertTrue(inProgressLatch4.await(30, TimeUnit.SECONDS));
    }

    public void testGetModel() throws IOException, InterruptedException, ExecutionException {
        ModelIndex modelIndex = ModelIndex.getInstance();
        String modelId = "efbsdhcvbsd";
        byte[] modelBlob = "hello".getBytes();

        // model index doesnt exist
        expectThrows(IllegalStateException.class, () -> modelIndex.getModel(modelId));

        // model id doesnt exist
        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);
        ActionListener<CreateIndexResponse> indexCreationListener = ActionListener.wrap(response -> {
            assertTrue(response.isAcknowledged());
            inProgressLatch1.countDown();
        }, exception -> {
            if (!(ExceptionsHelper.unwrapCause(exception) instanceof ResourceAlreadyExistsException)) {
                fail("Failed during index creation: " + exception);
            }
            inProgressLatch1.countDown();
        });

        modelIndex.createModelIndex(indexCreationListener);
        assertTrue(inProgressLatch1.await(30, TimeUnit.SECONDS));

        expectThrows(Exception.class, () -> modelIndex.getModel(modelId));

        // model id exists
        final CountDownLatch inProgressLatch2 = new CountDownLatch(1);
        ActionListener<IndexResponse> docCreationListener = ActionListener.wrap(response -> {
            assertEquals(RestStatus.CREATED, response.status());
            assertEquals(modelId, response.getId());
            inProgressLatch2.countDown();
        }, exception -> fail("Unable to put the model: " + exception));

        modelIndex.putModel(modelId, KNNEngine.DEFAULT, modelBlob, docCreationListener);
        assertTrue(inProgressLatch2.await(30, TimeUnit.SECONDS));

        assertArrayEquals(modelBlob, modelIndex.getModel(modelId));
    }

    public void testDeleteModel() throws IOException, InterruptedException {
        ModelIndex modelIndex = ModelIndex.getInstance();
        String modelId = "efbsdhcvbsd";
        byte[] modelBlob = "hello".getBytes();

        // model index doesnt exist
        expectThrows(IllegalStateException.class, () -> modelIndex.deleteModel(modelId, null));

        // model id doesnt exist
        final CountDownLatch inProgressLatch1 = new CountDownLatch(1);
        ActionListener<CreateIndexResponse> indexCreationListener = ActionListener.wrap(response -> {
            assertTrue(response.isAcknowledged());
            inProgressLatch1.countDown();
        }, exception -> {
            if (!(ExceptionsHelper.unwrapCause(exception) instanceof ResourceAlreadyExistsException)) {
                fail("Failed during index creation: " + exception);
            }
            inProgressLatch1.countDown();
        });

        modelIndex.createModelIndex(indexCreationListener);

        assertTrue(inProgressLatch1.await(30, TimeUnit.SECONDS));

        final CountDownLatch inProgressLatch2 = new CountDownLatch(1);
        ActionListener<DeleteResponse> deleteModelDoesNotExistListener = ActionListener.wrap(response -> {
            assertEquals(RestStatus.NOT_FOUND, response.status());
            inProgressLatch2.countDown();
        }, exception -> fail("Unable to delete the model: " + exception));

        modelIndex.deleteModel(modelId, deleteModelDoesNotExistListener);
        assertTrue(inProgressLatch2.await(30, TimeUnit.SECONDS));

        // model id exists
        final CountDownLatch inProgressLatch3 = new CountDownLatch(1);

        ActionListener<IndexResponse> docCreationListener = ActionListener.wrap(response -> {
            assertEquals(RestStatus.CREATED, response.status());
            assertEquals(modelId, response.getId());
            inProgressLatch3.countDown();
        }, exception -> fail("Unable to put the model: " + exception));


        modelIndex.putModel(modelId, KNNEngine.DEFAULT, modelBlob, docCreationListener);
        assertTrue(inProgressLatch3.await(30, TimeUnit.SECONDS));

        final CountDownLatch inProgressLatch4 = new CountDownLatch(1);
        ActionListener<DeleteResponse> deleteModelExistsListener = ActionListener.wrap(response -> {
            assertEquals(modelId, response.getId());
            inProgressLatch4.countDown();
        }, exception -> fail("Unable to delete model: " + exception));

        modelIndex.deleteModel(modelId, deleteModelExistsListener);
        assertTrue(inProgressLatch4.await(30, TimeUnit.SECONDS));
    }
}
