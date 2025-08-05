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
import org.opensearch.knn.index.util.IndexUtil;
import org.opensearch.knn.index.util.KNNClusterUtil;
import org.apache.lucene.search.join.BitSetProducer;

import java.io.IOException;
import java.util.List;

import static org.hamcrest.Matchers.instanceOf;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNClusterTestUtils.mockClusterService;

public class KNNExactQueryBuilderTests extends KNNTestCase {

    private static final String FIELD_NAME = "myvector";
    private static final float[] QUERY_VECTOR = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
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
            () -> KNNExactQueryBuilder.builder().fieldName(FIELD_NAME).vector(null).spaceType(SPACE_TYPE).build()
        );
        /**
         * empty query vector
         */
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNExactQueryBuilder.builder().fieldName(FIELD_NAME).vector(new float[0]).spaceType(SPACE_TYPE).build()
        );
    }

    public void testInvalidSpaceType() {
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNExactQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).spaceType("cosinesimilarity").build()
        );
    }

    public void testIgnoreUnmapped() throws IOException {
        KNNExactQueryBuilder.Builder builder = KNNExactQueryBuilder.builder()
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
            () -> KNNExactQueryBuilder.builder().fieldName("").vector(QUERY_VECTOR).spaceType(SPACE_TYPE).build()
        );
        /**
         * null field name
         */
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNExactQueryBuilder.builder().fieldName(null).vector(QUERY_VECTOR).spaceType(SPACE_TYPE).build()
        );
    }

    public void testValidSpaceTypes() {
        String[] validSpaceTypes = { "l2", "cosinesimil", "innerproduct", "hamming", "l1", "linf" };
        for (String spaceType : validSpaceTypes) {
            KNNExactQueryBuilder builder = KNNExactQueryBuilder.builder()
                .fieldName(FIELD_NAME)
                .vector(QUERY_VECTOR)
                .spaceType(spaceType)
                .build();
            assertEquals(spaceType, builder.getSpaceType());
        }
    }

    public void testExpandNested() {
        KNNExactQueryBuilder builder = KNNExactQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType(SPACE_TYPE)
            .expandNested(true)
            .build();
        assertTrue(builder.getExpandNested());

        builder = KNNExactQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType(SPACE_TYPE)
            .expandNested(false)
            .build();
        assertFalse(builder.getExpandNested());

        builder = KNNExactQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).spaceType(SPACE_TYPE).build();
        assertNull(builder.getExpandNested());
    }

    public void testDoToQuery_Normal() {
        KNNExactQueryBuilder builder = KNNExactQueryBuilder.builder()
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

        KNNExactQuery query = (KNNExactQuery) builder.doToQuery(mockQueryShardContext);
        assertNotNull(query);
        assertEquals(builder.fieldName(), query.getField());
        assertEquals(builder.vector(), query.getQueryVector());
        assertEquals(builder.getSpaceType(), query.getSpaceType());
    }

    public void testDoToQuery_NoSpaceType() {
        KNNExactQueryBuilder builder = KNNExactQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).build();

        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);

        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(getDefaultKNNMethodContext(), 4));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);

        KNNExactQuery query = (KNNExactQuery) builder.doToQuery(mockQueryShardContext);
        assertNotNull(query);
        assertEquals(builder.fieldName(), query.getField());
        assertEquals(builder.vector(), query.getQueryVector());
        assertEquals(SpaceType.DEFAULT.getValue(), query.getSpaceType());
    }

    public void testDoToQuery_InvalidFieldType() {
        KNNExactQueryBuilder builder = KNNExactQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType(SPACE_TYPE)
            .build();

        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mock(MappedFieldType.class)); // Not KNNVectorFieldType

        expectThrows(IllegalArgumentException.class, () -> builder.doToQuery(mockQueryShardContext));
    }

    public void testDoToQuery_WrongVectorDimension() {
        KNNExactQueryBuilder builder = KNNExactQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(new float[] { 1.0f, 2.0f })
            .spaceType(SPACE_TYPE)
            .build();

        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);

        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(getDefaultKNNMethodContext(), 4)); // 4
                                                                                                                                      // dimensions
                                                                                                                                      // expected
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);

        expectThrows(IllegalArgumentException.class, () -> builder.doToQuery(mockQueryShardContext));
    }

    public void testExpandNestedWithoutParentFilter() {
        KNNExactQueryBuilder builder = KNNExactQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType(SPACE_TYPE)
            .expandNested(true)
            .build();

        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);

        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(getDefaultKNNMethodContext(), 4));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        when(mockQueryShardContext.getParentFilter()).thenReturn(null); // No parent filter

        expectThrows(IllegalArgumentException.class, () -> builder.doToQuery(mockQueryShardContext));
    }

    public void testBuilderDefaults() {
        KNNExactQueryBuilder builder = KNNExactQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).build();

        assertNull(builder.getSpaceType()); // Should be null by default
        assertFalse(builder.isIgnoreUnmapped()); // Should be false by default
        assertNull(builder.getExpandNested()); // Should be null by default
    }

    public void testNestedFieldsWithExpandNested() {
        KNNExactQueryBuilder builder = KNNExactQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType(SPACE_TYPE)
            .expandNested(true)
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

        KNNExactQuery query = (KNNExactQuery) builder.doToQuery(mockQueryShardContext);
        assertNotNull(query);
        assertTrue(query.isExpandNested());
        assertEquals(mockParentFilter, query.getParentFilter());
    }

    public void testSerialization() throws Exception {
        assertSerialization(Version.CURRENT);
        assertSerialization(Version.V_2_3_0);
    }

    @Override
    protected NamedWriteableRegistry writableRegistry() {
        final List<NamedWriteableRegistry.Entry> entries = ClusterModule.getNamedWriteables();
        entries.add(new NamedWriteableRegistry.Entry(QueryBuilder.class, KNNExactQueryBuilder.NAME, KNNExactQueryBuilder::new));
        return new NamedWriteableRegistry(entries);
    }

    private void assertSerialization(final Version version) throws Exception {
        final KNNExactQueryBuilder knnExactQueryBuilder = KNNExactQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .spaceType(SPACE_TYPE)
            .expandNested(true)
            .build();

        final ClusterService clusterService = mockClusterService(version);

        final KNNClusterUtil knnClusterUtil = KNNClusterUtil.instance();
        knnClusterUtil.initialize(clusterService);
        try (BytesStreamOutput output = new BytesStreamOutput()) {
            output.setVersion(version);
            output.writeNamedWriteable(knnExactQueryBuilder);

            try (StreamInput in = new NamedWriteableAwareStreamInput(output.bytes().streamInput(), writableRegistry())) {
                in.setVersion(version);
                final QueryBuilder deserializedQuery = in.readNamedWriteable(QueryBuilder.class);

                assertNotNull(deserializedQuery);
                assertTrue(deserializedQuery instanceof KNNExactQueryBuilder);
                final KNNExactQueryBuilder deserializedKnnExactQueryBuilder = (KNNExactQueryBuilder) deserializedQuery;
                assertEquals(FIELD_NAME, deserializedKnnExactQueryBuilder.fieldName());
                assertArrayEquals(QUERY_VECTOR, (float[]) deserializedKnnExactQueryBuilder.vector(), 0.0f);
                assertEquals(SPACE_TYPE, deserializedKnnExactQueryBuilder.getSpaceType());
                assertFalse(deserializedKnnExactQueryBuilder.isIgnoreUnmapped());
                if (version.onOrAfter(IndexUtil.minimalRequiredVersionMap.get("expand_nested_docs"))) {
                    assertTrue(deserializedKnnExactQueryBuilder.getExpandNested());
                } else {
                    assertNull(deserializedKnnExactQueryBuilder.getExpandNested());
                }
            }
        }
    }
}
