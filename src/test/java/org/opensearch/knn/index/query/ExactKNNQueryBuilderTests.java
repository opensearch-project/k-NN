/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.junit.Before;
import org.opensearch.Version;
import org.opensearch.cluster.ClusterModule;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.core.common.io.stream.NamedWriteableAwareStreamInput;
import org.opensearch.core.common.io.stream.NamedWriteableRegistry;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.index.Index;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.util.KNNClusterUtil;
import org.apache.lucene.search.join.BitSetProducer;

import java.io.IOException;
import java.util.List;

import static org.hamcrest.Matchers.instanceOf;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNClusterTestUtils.mockClusterService;

public class ExactKNNQueryBuilderTests extends KNNTestCase {

    private static final String FIELD_NAME = "myvector";
    private static final float[] QUERY_VECTOR = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
    private static final byte[] BYTE_QUERY_VECTOR = new byte[] { 1, 2, 3, 4 };
    private static final String SPACE_TYPE = "innerproduct";

    @Before
    @Override
    public void setUp() throws Exception {
        super.setUp();
        ClusterSettings clusterSettings = mock(ClusterSettings.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        KNNSettings.state().setClusterService(clusterService);
    }

    public void testEmptyVector() {
        /**
         * null query vector
         */
        expectThrows(
            IllegalArgumentException.class,
            () -> ExactKNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(null).spaceType(SPACE_TYPE).build()
        );
        /**
         * empty query vector
         */
        expectThrows(
            IllegalArgumentException.class,
            () -> ExactKNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(new float[0]).spaceType(SPACE_TYPE).build()
        );
    }

    public void testInvalidSpaceType() {
        expectThrows(
            IllegalArgumentException.class,
            () -> ExactKNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).spaceType("cosinesimilarity").build()
        );
    }

    public void testIgnoreUnmapped() throws IOException {
        ExactKNNQueryBuilder.Builder builder = ExactKNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType(SPACE_TYPE)
            .ignoreUnmapped(true);
        assertTrue(builder.build().isIgnoreUnmapped());

        Query query = builder.build().doToQuery(mock(QueryShardContext.class));
        assertNotNull(query);
        assertThat(query, instanceOf(MatchNoDocsQuery.class));
        builder.ignoreUnmapped(false);
        expectThrows(IllegalArgumentException.class, () -> builder.build().doToQuery(mock(QueryShardContext.class)));
    }

    public void testEmptyFieldName() {
        /**
         * empty field name
         */
        expectThrows(
            IllegalArgumentException.class,
            () -> ExactKNNQueryBuilder.builder().fieldName("").vector(QUERY_VECTOR).spaceType(SPACE_TYPE).build()
        );
        /**
         * null field name
         */
        expectThrows(
            IllegalArgumentException.class,
            () -> ExactKNNQueryBuilder.builder().fieldName(null).vector(QUERY_VECTOR).spaceType(SPACE_TYPE).build()
        );
    }

    public void testValidSpaceTypes() {
        String[] validSpaceTypes = { "l2", "cosinesimil", "innerproduct", "hamming", "l1", "linf" };
        for (String spaceType : validSpaceTypes) {
            ExactKNNQueryBuilder builder = ExactKNNQueryBuilder.builder()
                .fieldName(FIELD_NAME)
                .vector(QUERY_VECTOR)
                .spaceType(spaceType)
                .build();
            assertEquals(spaceType, builder.getSpaceType());
        }
    }

    public void testDoToQuery_Normal() {
        ExactKNNQueryBuilder builder = ExactKNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType(SPACE_TYPE)
            .build();

        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);

        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(getDefaultKNNMethodContext(), 4));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);

        // float
        ExactKNNFloatQuery query = (ExactKNNFloatQuery) builder.doToQuery(mockQueryShardContext);
        assertNotNull(query);
        assertEquals(builder.fieldName(), query.getField());
        assertEquals(builder.getVector(), query.getQueryVector());
        assertEquals(builder.getSpaceType(), query.getSpaceType());

        // byte
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.BYTE);
        ExactKNNFloatQuery byteQuery = (ExactKNNFloatQuery) builder.doToQuery(mockQueryShardContext);
        assertNotNull(byteQuery);
        assertEquals(builder.fieldName(), byteQuery.getField());
        assertEquals(builder.getVector(), byteQuery.getQueryVector());
        assertEquals(builder.getSpaceType(), byteQuery.getSpaceType());

        // binary
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.BINARY);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(getDefaultBinaryKNNMethodContext(), 32));
        ExactKNNByteQuery binaryQuery = (ExactKNNByteQuery) builder.doToQuery(mockQueryShardContext);
        assertNotNull(binaryQuery);
        assertEquals(builder.fieldName(), binaryQuery.getField());
        assertArrayEquals(BYTE_QUERY_VECTOR, binaryQuery.getByteQueryVector());
        assertEquals(builder.getSpaceType(), binaryQuery.getSpaceType());
    }

    public void testDoToQuery_NoSpaceType() {
        ExactKNNQueryBuilder builder = ExactKNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).build();

        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);

        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(getDefaultKNNMethodContext(), 4));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);

        // float
        ExactKNNFloatQuery query = (ExactKNNFloatQuery) builder.doToQuery(mockQueryShardContext);
        assertNotNull(query);
        assertEquals(builder.fieldName(), query.getField());
        assertEquals(builder.getVector(), query.getQueryVector());
        assertEquals(SpaceType.DEFAULT.getValue(), query.getSpaceType());

        // byte
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.BYTE);
        ExactKNNFloatQuery byteQuery = (ExactKNNFloatQuery) builder.doToQuery(mockQueryShardContext);
        assertNotNull(byteQuery);
        assertEquals(builder.fieldName(), byteQuery.getField());
        assertEquals(builder.getVector(), byteQuery.getQueryVector());
        assertEquals(SpaceType.DEFAULT.getValue(), byteQuery.getSpaceType());
    }

    public void testDoToQuery_InvalidFieldType() {
        ExactKNNQueryBuilder builder = ExactKNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType(SPACE_TYPE)
            .build();

        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mock(MappedFieldType.class)); // Not KNNVectorFieldType

        expectThrows(IllegalArgumentException.class, () -> builder.doToQuery(mockQueryShardContext));
    }

    public void testDoToQuery_WrongVectorDimension() {
        ExactKNNQueryBuilder builder = ExactKNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(new float[] { 1.0f, 2.0f })
            .spaceType(SPACE_TYPE)
            .build();

        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);

        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(getDefaultKNNMethodContext(), 4));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);

        expectThrows(IllegalArgumentException.class, () -> builder.doToQuery(mockQueryShardContext));
    }

    public void testBuilderDefaults() {
        ExactKNNQueryBuilder builder = ExactKNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).build();

        assertNull(builder.getSpaceType()); // Should be null by default
        assertFalse(builder.isIgnoreUnmapped()); // Should be false by default
    }

    public void testNestedFields() {
        ExactKNNQueryBuilder builder = ExactKNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType(SPACE_TYPE)
            .build();

        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        BitSetProducer mockParentFilter = mock(BitSetProducer.class);

        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(getDefaultKNNMethodContext(), 4));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        when(mockQueryShardContext.getParentFilter()).thenReturn(mockParentFilter);

        ExactKNNQuery query = (ExactKNNQuery) builder.doToQuery(mockQueryShardContext);
        assertNotNull(query);
        assertEquals(mockParentFilter, query.getParentFilter());
    }

    public void testSerialization() throws Exception {
        assertSerialization(Version.CURRENT);
        assertSerialization(Version.V_2_3_0);
    }

    @Override
    protected NamedWriteableRegistry writableRegistry() {
        final List<NamedWriteableRegistry.Entry> entries = ClusterModule.getNamedWriteables();
        entries.add(new NamedWriteableRegistry.Entry(QueryBuilder.class, ExactKNNQueryBuilder.NAME, ExactKNNQueryBuilder::new));
        return new NamedWriteableRegistry(entries);
    }

    private void assertSerialization(final Version version) throws Exception {
        final ExactKNNQueryBuilder exactKNNQueryBuilder = ExactKNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType(SPACE_TYPE)
            .build();

        final ClusterService clusterService = mockClusterService(version);

        final KNNClusterUtil knnClusterUtil = KNNClusterUtil.instance();
        knnClusterUtil.initialize(clusterService);
        try (BytesStreamOutput output = new BytesStreamOutput()) {
            output.setVersion(version);
            output.writeNamedWriteable(exactKNNQueryBuilder);

            try (StreamInput in = new NamedWriteableAwareStreamInput(output.bytes().streamInput(), writableRegistry())) {
                in.setVersion(version);
                final QueryBuilder deserializedQuery = in.readNamedWriteable(QueryBuilder.class);

                assertNotNull(deserializedQuery);
                assertTrue(deserializedQuery instanceof ExactKNNQueryBuilder);
                final ExactKNNQueryBuilder deserializedExactKNNQueryBuilder = (ExactKNNQueryBuilder) deserializedQuery;
                assertEquals(FIELD_NAME, deserializedExactKNNQueryBuilder.fieldName());
                assertArrayEquals(QUERY_VECTOR, deserializedExactKNNQueryBuilder.getVector(), 0.0f);
                assertEquals(SPACE_TYPE, deserializedExactKNNQueryBuilder.getSpaceType());
                assertFalse(deserializedExactKNNQueryBuilder.isIgnoreUnmapped());
            }
        }
    }
}
