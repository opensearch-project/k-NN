/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.processor.muvera;

import org.opensearch.ingest.IngestDocument;
import org.opensearch.ingest.Processor;
import org.opensearch.knn.KNNTestCase;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MuveraIngestProcessorTests extends KNNTestCase {

    private static final String SOURCE_FIELD = "colbert_vectors";
    private static final String TARGET_FIELD = "muvera_fde";

    private MuveraIngestProcessor createProcessor(int dim, int kSim, int dimProj, int rReps) {
        MuveraEncoder encoder = new MuveraEncoder(dim, kSim, dimProj, rReps, 42L);
        return new MuveraIngestProcessor(
            "test_tag",
            "test description",
            SOURCE_FIELD,
            TARGET_FIELD,
            encoder,
            dim,
            encoder.getEmbeddingSize()
        );
    }

    public void testExecuteProducesFdeOfCorrectDimension() throws Exception {
        MuveraIngestProcessor processor = createProcessor(4, 1, 2, 2);
        int expectedDim = 2 * 2 * 2; // 8

        Map<String, Object> source = new HashMap<>();
        source.put(SOURCE_FIELD, Arrays.asList(Arrays.asList(1.0, 0.0, 0.0, 0.0), Arrays.asList(0.0, 1.0, 0.0, 0.0)));
        IngestDocument doc = new IngestDocument(source, new HashMap<>());

        processor.execute(doc);

        Object fdeObj = doc.getFieldValue(TARGET_FIELD, Object.class);
        assertNotNull(fdeObj);
        assertTrue(fdeObj instanceof List);
        List<?> fdeList = (List<?>) fdeObj;
        assertEquals(expectedDim, fdeList.size());
    }

    public void testExecutePreservesSourceField() throws Exception {
        MuveraIngestProcessor processor = createProcessor(4, 1, 2, 2);

        Map<String, Object> source = new HashMap<>();
        List<List<Number>> vectors = Arrays.asList(Arrays.asList(1.0, 0.0, 0.0, 0.0));
        source.put(SOURCE_FIELD, vectors);
        IngestDocument doc = new IngestDocument(source, new HashMap<>());

        processor.execute(doc);

        // Source field should still be present
        assertNotNull(doc.getFieldValue(SOURCE_FIELD, Object.class));
        // Target field should also be present
        assertNotNull(doc.getFieldValue(TARGET_FIELD, Object.class));
    }

    public void testExecuteThrowsOnNullSourceField() {
        MuveraIngestProcessor processor = createProcessor(4, 1, 2, 2);

        Map<String, Object> source = new HashMap<>();
        source.put(SOURCE_FIELD, null);
        IngestDocument doc = new IngestDocument(source, new HashMap<>());

        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> processor.execute(doc));
        assertTrue(e.getMessage().contains("is null"));
    }

    public void testExecuteThrowsOnEmptyVectors() {
        MuveraIngestProcessor processor = createProcessor(4, 1, 2, 2);

        Map<String, Object> source = new HashMap<>();
        source.put(SOURCE_FIELD, Arrays.asList());
        IngestDocument doc = new IngestDocument(source, new HashMap<>());

        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> processor.execute(doc));
        assertTrue(e.getMessage().contains("empty"));
    }

    public void testExecuteThrowsOnDimensionMismatch() {
        MuveraIngestProcessor processor = createProcessor(4, 1, 2, 2);

        Map<String, Object> source = new HashMap<>();
        source.put(
            SOURCE_FIELD,
            Arrays.asList(
                Arrays.asList(1.0, 0.0, 0.0) // dim=3, expected dim=4
            )
        );
        IngestDocument doc = new IngestDocument(source, new HashMap<>());

        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> processor.execute(doc));
        assertTrue(e.getMessage().contains("dimension"));
        assertTrue(e.getMessage().contains("3"));
        assertTrue(e.getMessage().contains("4"));
    }

    public void testExecuteThrowsOnNonListInput() {
        MuveraIngestProcessor processor = createProcessor(4, 1, 2, 2);

        Map<String, Object> source = new HashMap<>();
        source.put(SOURCE_FIELD, "not a list");
        IngestDocument doc = new IngestDocument(source, new HashMap<>());

        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> processor.execute(doc));
        assertTrue(e.getMessage().contains("list of vectors"));
    }

    public void testGetType() {
        MuveraIngestProcessor processor = createProcessor(4, 1, 2, 2);
        assertEquals("muvera", processor.getType());
    }

    // Factory tests

    public void testFactoryCreateWithDefaults() throws Exception {
        MuveraIngestProcessor.Factory factory = new MuveraIngestProcessor.Factory();

        Map<String, Object> config = new HashMap<>();
        config.put("source_field", SOURCE_FIELD);
        config.put("target_field", TARGET_FIELD);
        config.put("dim", 128);

        Processor processor = factory.create(Map.of(), "tag", null, config);
        assertNotNull(processor);
        assertEquals("muvera", processor.getType());
    }

    public void testFactoryThrowsOnMissingDim() {
        MuveraIngestProcessor.Factory factory = new MuveraIngestProcessor.Factory();

        Map<String, Object> config = new HashMap<>();
        config.put("source_field", SOURCE_FIELD);
        config.put("target_field", TARGET_FIELD);
        // dim is missing

        Exception e = expectThrows(Exception.class, () -> factory.create(Map.of(), "tag", null, config));
        assertTrue(e.getMessage().contains("dim"));
    }

    public void testFactoryThrowsOnFdeDimensionMismatch() {
        MuveraIngestProcessor.Factory factory = new MuveraIngestProcessor.Factory();

        Map<String, Object> config = new HashMap<>();
        config.put("source_field", SOURCE_FIELD);
        config.put("target_field", TARGET_FIELD);
        config.put("dim", 4);
        config.put("k_sim", 1);
        config.put("dim_proj", 2);
        config.put("r_reps", 2);
        config.put("fde_dimension", 999); // wrong

        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> factory.create(Map.of(), "tag", null, config));
        assertTrue(e.getMessage().contains("fde_dimension"));
        assertTrue(e.getMessage().contains("999"));
    }

    public void testFactoryAcceptsCorrectFdeDimension() throws Exception {
        MuveraIngestProcessor.Factory factory = new MuveraIngestProcessor.Factory();

        Map<String, Object> config = new HashMap<>();
        config.put("source_field", SOURCE_FIELD);
        config.put("target_field", TARGET_FIELD);
        config.put("dim", 4);
        config.put("k_sim", 1);
        config.put("dim_proj", 2);
        config.put("r_reps", 2);
        config.put("fde_dimension", 8); // correct: 2 * 2 * 2

        Processor processor = factory.create(Map.of(), "tag", null, config);
        assertNotNull(processor);
    }
}
