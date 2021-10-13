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

package org.opensearch.knn.training;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.JNIService;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryEntryContext;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.ExecutionException;

import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.METHOD_IVF;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NLIST;

public class TrainingJobTests extends KNNTestCase {

    public void testGetModelId() {
        String modelId = "test-model-id";
        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.getEngine()).thenReturn(KNNEngine.DEFAULT);
        when(knnMethodContext.getSpaceType()).thenReturn(SpaceType.DEFAULT);

        TrainingJob trainingJob = new TrainingJob(
                modelId,
                knnMethodContext,
                mock(NativeMemoryCacheManager.class),
                mock(NativeMemoryEntryContext.TrainingDataEntryContext.class),
                mock(NativeMemoryEntryContext.AnonymousEntryContext.class),
                10,
                ""
        );

        assertEquals(modelId, trainingJob.getModelId());
    }

    public void testSetModelId() {
        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.getEngine()).thenReturn(KNNEngine.DEFAULT);
        when(knnMethodContext.getSpaceType()).thenReturn(SpaceType.DEFAULT);

        TrainingJob trainingJob = new TrainingJob(
                "test-model-id",
                knnMethodContext,
                mock(NativeMemoryCacheManager.class),
                mock(NativeMemoryEntryContext.TrainingDataEntryContext.class),
                mock(NativeMemoryEntryContext.AnonymousEntryContext.class),
                10,
                ""
        );

        String setModelId = "test-model-id-2";
        trainingJob.setModelId(setModelId);

        assertEquals(setModelId, trainingJob.getModelId());
    }

    public void testGetModel() {
        SpaceType spaceType = SpaceType.INNER_PRODUCT;
        KNNEngine knnEngine = KNNEngine.DEFAULT;
        int dimension = 10;
        String desciption = "test description";
        String error = "";

        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.getEngine()).thenReturn(knnEngine);
        when(knnMethodContext.getSpaceType()).thenReturn(spaceType);

        TrainingJob trainingJob = new TrainingJob(
                "test-model-id",
                knnMethodContext,
                mock(NativeMemoryCacheManager.class),
                mock(NativeMemoryEntryContext.TrainingDataEntryContext.class),
                mock(NativeMemoryEntryContext.AnonymousEntryContext.class),
                dimension,
                desciption
        );

        Model model = new Model(
                new ModelMetadata(
                        knnEngine,
                        spaceType,
                        dimension,
                        ModelState.TRAINING,
                        trainingJob.getModel().getModelMetadata().getTimestamp(),
                        desciption,
                        error
                ),
                null
        );

        assertEquals(model, trainingJob.getModel());
    }

    public void testRun_success() throws IOException, ExecutionException {
        // Successful end to end run case
        String modelId = "test-model-id";

        // Define the method setup for method that requires training
        int nlists = 5;
        int dimension = 16;
        KNNEngine knnEngine = KNNEngine.FAISS;
        KNNMethodContext knnMethodContext = new KNNMethodContext(knnEngine, SpaceType.INNER_PRODUCT,
                new MethodComponentContext(METHOD_IVF, ImmutableMap.of(METHOD_PARAMETER_NLIST, nlists)));

        // Set up training data
        int tdataPoints = 100;
        float[][] trainingData = new float[tdataPoints][dimension];
        fillFloatArrayRandomly(trainingData);
        long memoryAddress = JNIService.transferVectors(0, trainingData);

        // Setup model manager
        NativeMemoryCacheManager nativeMemoryCacheManager = mock(NativeMemoryCacheManager.class);

        // Setup mock allocation for model
        NativeMemoryAllocation modelAllocation = mock(NativeMemoryAllocation.class);
        doAnswer(invocationOnMock -> null).when(modelAllocation).readLock();
        doAnswer(invocationOnMock -> null).when(modelAllocation).readUnlock();
        when(modelAllocation.isClosed()).thenReturn(false);

        String modelKey = "model-test-key";
        NativeMemoryEntryContext.AnonymousEntryContext modelContext = mock(NativeMemoryEntryContext.AnonymousEntryContext.class);
        when(modelContext.getKey()).thenReturn(modelKey);

        when(nativeMemoryCacheManager.get(modelContext, false)).thenReturn(modelAllocation);
        doAnswer(invocationOnMock -> null).when(nativeMemoryCacheManager).invalidate(modelKey);

        // Setup mock allocation for training data
        NativeMemoryAllocation nativeMemoryAllocation = mock(NativeMemoryAllocation.class);
        doAnswer(invocationOnMock -> null).when(nativeMemoryAllocation).readLock();
        doAnswer(invocationOnMock -> null).when(nativeMemoryAllocation).readUnlock();
        when(nativeMemoryAllocation.isClosed()).thenReturn(false);
        when(nativeMemoryAllocation.getMemoryAddress()).thenReturn(memoryAddress);

        String tdataKey = "t-data-key";
        NativeMemoryEntryContext.TrainingDataEntryContext trainingDataEntryContext =
                mock(NativeMemoryEntryContext.TrainingDataEntryContext.class);
        when(trainingDataEntryContext.getKey()).thenReturn(tdataKey);

        when(nativeMemoryCacheManager.get(trainingDataEntryContext, false)).thenReturn(nativeMemoryAllocation);
        doAnswer(invocationOnMock -> {
            JNIService.freeVectors(memoryAddress);
            return null;
        }).when(nativeMemoryCacheManager).invalidate(tdataKey);

        TrainingJob trainingJob = new TrainingJob(
                modelId,
                knnMethodContext,
                nativeMemoryCacheManager,
                trainingDataEntryContext,
                modelContext,
                dimension,
                ""
        );

        trainingJob.run();

        Model model = trainingJob.getModel();
        assertNotNull(model);

        assertEquals(ModelState.CREATED, model.getModelMetadata().getState());

        // Simple test that creates the index from template and doesnt fail
        int[] ids = { 1, 2, 3, 4};
        float[][] vectors = new float[ids.length][dimension];
        fillFloatArrayRandomly(vectors);

        Path indexPath = createTempFile();
        JNIService.createIndexFromTemplate(ids, vectors, indexPath.toString(), model.getModelBlob(), knnEngine.getName());
        assertNotEquals(0, new File(indexPath.toString()).length());
    }

    public void testRun_failure_onGetTrainingDataAllocation() throws ExecutionException {
        // In this test, getting a training data allocation should fail. Then, run should fail and update the error of
        // the model
        String modelId = "test-model-id";

        // Define the method setup for method that requires training
        int nlists = 5;
        int dimension = 16;
        KNNEngine knnEngine = KNNEngine.FAISS;
        KNNMethodContext knnMethodContext = new KNNMethodContext(knnEngine, SpaceType.INNER_PRODUCT,
                new MethodComponentContext(METHOD_IVF, ImmutableMap.of(METHOD_PARAMETER_NLIST, nlists)));

        // Setup model manager
        NativeMemoryCacheManager nativeMemoryCacheManager = mock(NativeMemoryCacheManager.class);

        // Setup mock allocation for model
        NativeMemoryAllocation modelAllocation = mock(NativeMemoryAllocation.class);
        doAnswer(invocationOnMock -> null).when(modelAllocation).readLock();
        doAnswer(invocationOnMock -> null).when(modelAllocation).readUnlock();
        when(modelAllocation.isClosed()).thenReturn(false);

        String modelKey = "model-test-key";
        NativeMemoryEntryContext.AnonymousEntryContext modelContext = mock(NativeMemoryEntryContext.AnonymousEntryContext.class);
        when(modelContext.getKey()).thenReturn(modelKey);

        when(nativeMemoryCacheManager.get(modelContext, false)).thenReturn(modelAllocation);
        doAnswer(invocationOnMock -> null).when(nativeMemoryCacheManager).invalidate(modelKey);

        // Setup mock allocation for training data
        String tdataKey = "t-data-key";
        NativeMemoryEntryContext.TrainingDataEntryContext trainingDataEntryContext =
                mock(NativeMemoryEntryContext.TrainingDataEntryContext.class);
        when(trainingDataEntryContext.getKey()).thenReturn(tdataKey);

        // Throw error on getting data
        String testException = "test exception";
        when(nativeMemoryCacheManager.get(trainingDataEntryContext, false))
                .thenThrow(new RuntimeException(testException));

        TrainingJob trainingJob = new TrainingJob(
                modelId,
                knnMethodContext,
                nativeMemoryCacheManager,
                trainingDataEntryContext,
                modelContext,
                dimension,
                ""
        );

        trainingJob.run();

        Model model = trainingJob.getModel();
        assertEquals(ModelState.FAILED, trainingJob.getModel().getModelMetadata().getState());
        assertNotNull(model);
        assertEquals(testException, model.getModelMetadata().getError());
    }

    public void testRun_failure_onGetModelAnonymousAllocation() throws ExecutionException {
        // In this test, getting a training data allocation should fail. Then, run should fail and update the error of
        // the model
        String modelId = "test-model-id";

        // Define the method setup for method that requires training
        int nlists = 5;
        int dimension = 16;
        KNNEngine knnEngine = KNNEngine.FAISS;
        KNNMethodContext knnMethodContext = new KNNMethodContext(knnEngine, SpaceType.INNER_PRODUCT,
                new MethodComponentContext(METHOD_IVF, ImmutableMap.of(METHOD_PARAMETER_NLIST, nlists)));

        // Setup model manager
        NativeMemoryCacheManager nativeMemoryCacheManager = mock(NativeMemoryCacheManager.class);

        // Setup mock allocation for model
        NativeMemoryAllocation modelAllocation = mock(NativeMemoryAllocation.class);
        doAnswer(invocationOnMock -> null).when(modelAllocation).readLock();
        doAnswer(invocationOnMock -> null).when(modelAllocation).readUnlock();
        when(modelAllocation.isClosed()).thenReturn(false);

        String modelKey = "model-test-key";
        NativeMemoryEntryContext.AnonymousEntryContext modelContext = mock(NativeMemoryEntryContext.AnonymousEntryContext.class);
        when(modelContext.getKey()).thenReturn(modelKey);

        when(nativeMemoryCacheManager.get(modelContext, false)).thenReturn(modelAllocation);
        doAnswer(invocationOnMock -> null).when(nativeMemoryCacheManager).invalidate(modelKey);

        // Setup mock allocation for training data

        // Throw error on getting data
        String testException = "test exception";
        when(nativeMemoryCacheManager.get(modelContext, false))
                .thenThrow(new RuntimeException(testException));

        TrainingJob trainingJob = new TrainingJob(
                modelId,
                knnMethodContext,
                nativeMemoryCacheManager,
                mock(NativeMemoryEntryContext.TrainingDataEntryContext.class),
                modelContext,
                dimension,
                ""
        );

        trainingJob.run();

        Model model = trainingJob.getModel();
        assertEquals(ModelState.FAILED, trainingJob.getModel().getModelMetadata().getState());
        assertNotNull(model);
        assertEquals(testException, model.getModelMetadata().getError());
    }

    public void testRun_failure_closedTrainingDataAllocation() throws ExecutionException {
        // In this test, the training data allocation should be closed. Then, run should fail and update the error of
        // the model
        String modelId = "test-model-id";

        // Define the method setup for method that requires training
        int nlists = 5;
        int dimension = 16;
        KNNEngine knnEngine = KNNEngine.FAISS;
        KNNMethodContext knnMethodContext = new KNNMethodContext(knnEngine, SpaceType.INNER_PRODUCT,
                new MethodComponentContext(METHOD_IVF, ImmutableMap.of(METHOD_PARAMETER_NLIST, nlists)));

        String tdataKey = "t-data-key";
        NativeMemoryEntryContext.TrainingDataEntryContext trainingDataEntryContext =
                mock(NativeMemoryEntryContext.TrainingDataEntryContext.class);
        when(trainingDataEntryContext.getKey()).thenReturn(tdataKey);

        // Setup model manager
        NativeMemoryCacheManager nativeMemoryCacheManager = mock(NativeMemoryCacheManager.class);

        // Setup mock allocation for model
        NativeMemoryAllocation modelAllocation = mock(NativeMemoryAllocation.class);
        doAnswer(invocationOnMock -> null).when(modelAllocation).readLock();
        doAnswer(invocationOnMock -> null).when(modelAllocation).readUnlock();
        when(modelAllocation.isClosed()).thenReturn(false);

        String modelKey = "model-test-key";
        NativeMemoryEntryContext.AnonymousEntryContext modelContext = mock(NativeMemoryEntryContext.AnonymousEntryContext.class);
        when(modelContext.getKey()).thenReturn(modelKey);

        when(nativeMemoryCacheManager.get(modelContext, false)).thenReturn(modelAllocation);
        doAnswer(invocationOnMock -> null).when(nativeMemoryCacheManager).invalidate(modelKey);

        // Setup mock allocation thats closed
        NativeMemoryAllocation nativeMemoryAllocation = mock(NativeMemoryAllocation.class);
        doAnswer(invocationOnMock -> null).when(nativeMemoryAllocation).readLock();
        doAnswer(invocationOnMock -> null).when(nativeMemoryAllocation).readUnlock();
        when(nativeMemoryAllocation.isClosed()).thenReturn(true);
        when(nativeMemoryAllocation.getMemoryAddress()).thenReturn((long) 0);

        // Throw error on getting data
        when(nativeMemoryCacheManager.get(trainingDataEntryContext, false)).thenReturn(nativeMemoryAllocation);

        TrainingJob trainingJob = new TrainingJob(
                modelId,
                knnMethodContext,
                nativeMemoryCacheManager,
                trainingDataEntryContext,
                mock(NativeMemoryEntryContext.AnonymousEntryContext.class),
                dimension,
                ""
        );

        trainingJob.run();

        Model model = trainingJob.getModel();
        assertNotNull(model);
        assertEquals(ModelState.FAILED, trainingJob.getModel().getModelMetadata().getState());
    }

    public void testRun_failure_notEnoughTrainingData() throws ExecutionException {
        // In this test case, we ensure that failure happens gracefully when there isnt enough training data
        String modelId = "test-model-id";

        // Define the method setup for method that requires training
        int nlists = 1024; // setting this to 1024 will cause training to fail when there is only 2 data points
        int dimension = 16;
        KNNEngine knnEngine = KNNEngine.FAISS;
        KNNMethodContext knnMethodContext = new KNNMethodContext(knnEngine, SpaceType.INNER_PRODUCT,
                new MethodComponentContext(METHOD_IVF, ImmutableMap.of(METHOD_PARAMETER_NLIST, nlists)));

        // Set up training data
        int tdataPoints = 2;
        float[][] trainingData = new float[tdataPoints][dimension];
        fillFloatArrayRandomly(trainingData);
        long memoryAddress = JNIService.transferVectors(0, trainingData);

        // Setup model manager
        NativeMemoryCacheManager nativeMemoryCacheManager = mock(NativeMemoryCacheManager.class);

        // Setup mock allocation for model
        NativeMemoryAllocation modelAllocation = mock(NativeMemoryAllocation.class);
        doAnswer(invocationOnMock -> null).when(modelAllocation).readLock();
        doAnswer(invocationOnMock -> null).when(modelAllocation).readUnlock();
        when(modelAllocation.isClosed()).thenReturn(false);

        String modelKey = "model-test-key";
        NativeMemoryEntryContext.AnonymousEntryContext modelContext = mock(NativeMemoryEntryContext.AnonymousEntryContext.class);
        when(modelContext.getKey()).thenReturn(modelKey);

        when(nativeMemoryCacheManager.get(modelContext, false)).thenReturn(modelAllocation);
        doAnswer(invocationOnMock -> null).when(nativeMemoryCacheManager).invalidate(modelKey);

        // Setup mock allocation
        NativeMemoryAllocation nativeMemoryAllocation = mock(NativeMemoryAllocation.class);
        doAnswer(invocationOnMock -> null).when(nativeMemoryAllocation).readLock();
        doAnswer(invocationOnMock -> null).when(nativeMemoryAllocation).readUnlock();
        when(nativeMemoryAllocation.isClosed()).thenReturn(false);
        when(nativeMemoryAllocation.getMemoryAddress()).thenReturn(memoryAddress);

        String tdataKey = "t-data-key";
        NativeMemoryEntryContext.TrainingDataEntryContext trainingDataEntryContext =
                mock(NativeMemoryEntryContext.TrainingDataEntryContext.class);
        when(trainingDataEntryContext.getKey()).thenReturn(tdataKey);

        when(nativeMemoryCacheManager.get(trainingDataEntryContext, false)).thenReturn(nativeMemoryAllocation);
        doAnswer(invocationOnMock -> {
            JNIService.freeVectors(memoryAddress);
            return null;
        }).when(nativeMemoryCacheManager).invalidate(tdataKey);

        TrainingJob trainingJob = new TrainingJob(
                modelId,
                knnMethodContext,
                nativeMemoryCacheManager,
                trainingDataEntryContext,
                modelContext,
                dimension,
                ""
        );

        trainingJob.run();

        Model model = trainingJob.getModel();
        assertNotNull(model);
        assertEquals(ModelState.FAILED, model.getModelMetadata().getState());
        assertFalse(model.getModelMetadata().getError().isEmpty());
    }

    private void fillFloatArrayRandomly(float [][] vectors) {
        for (int i = 0; i < vectors.length; i++) {
            for (int j = 0; j < vectors[i].length; j++) {
                vectors[i][j] = randomFloat();
            }
        }
    }
}
