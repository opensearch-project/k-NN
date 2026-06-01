/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucenelib;

import junit.framework.TestCase;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.DiversifyingChildrenByteKnnVectorQuery;
import org.apache.lucene.search.join.DiversifyingChildrenFloatKnnVectorQuery;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import static org.mockito.Mockito.mock;

public class NestedKnnVectorQueryFactoryTests extends TestCase {
    public void testCreate_whenCalled_thenCreateQuery() {
        String fieldName = "field";
        byte[] byteVectors = new byte[3];
        float[] floatVectors = new float[3];
        int luceneK = 10;
        int k = 3;
        Query queryFilter = mock(Query.class);
        BitSetProducer parentFilter = mock(BitSetProducer.class);
        boolean expandNestedDocs = true;

        ExpandNestedDocsQuery expectedByteQuery = new ExpandNestedDocsQuery.ExpandNestedDocsQueryBuilder().internalNestedKnnVectorQuery(
            new InternalNestedKnnByteVectorQuery(fieldName, byteVectors, queryFilter, luceneK, parentFilter, k)
        ).queryUtils(null).build();
        assertEquals(
            expectedByteQuery,
            NestedKnnVectorQueryFactory.createNestedKnnVectorQuery(
                fieldName,
                byteVectors,
                luceneK,
                queryFilter,
                parentFilter,
                expandNestedDocs,
                k
            )
        );

        ExpandNestedDocsQuery expectedFloatQuery = new ExpandNestedDocsQuery.ExpandNestedDocsQueryBuilder().internalNestedKnnVectorQuery(
            new InternalNestedKnnFloatVectorQuery(
                fieldName,
                floatVectors,
                queryFilter,
                luceneK,
                parentFilter,
                k,
                RescoreContext.NO_RESCORE_NEEDED
            )
        ).queryUtils(null).build();
        assertEquals(
            expectedFloatQuery,
            NestedKnnVectorQueryFactory.createNestedKnnVectorQuery(
                fieldName,
                floatVectors,
                luceneK,
                queryFilter,
                parentFilter,
                expandNestedDocs,
                k,
                RescoreContext.NO_RESCORE_NEEDED
            )
        );
    }

    public void testCreate_whenNoExpandNestedDocs_thenDiversifyingQuery() {
        String fieldName = "field";
        byte[] byteVectors = new byte[3];
        float[] floatVectors = new float[3];
        int luceneK = 10;
        int k = 3;
        Query queryFilter = mock(Query.class);
        BitSetProducer parentFilter = mock(BitSetProducer.class);
        boolean expandNestedDocs = false;

        Query byteQuery = NestedKnnVectorQueryFactory.createNestedKnnVectorQuery(
            fieldName,
            byteVectors,
            luceneK,
            queryFilter,
            parentFilter,
            expandNestedDocs,
            k
        );
        assertEquals(OSDiversifyingChildrenByteKnnVectorQuery.class, byteQuery.getClass());
        assertTrue(byteQuery instanceof DiversifyingChildrenByteKnnVectorQuery);

        Query floatQuery = NestedKnnVectorQueryFactory.createNestedKnnVectorQuery(
            fieldName,
            floatVectors,
            luceneK,
            queryFilter,
            parentFilter,
            expandNestedDocs,
            k,
            RescoreContext.NO_RESCORE_NEEDED
        );
        assertEquals(OSDiversifyingChildrenFloatKnnVectorQuery.class, floatQuery.getClass());
        assertTrue(floatQuery instanceof DiversifyingChildrenFloatKnnVectorQuery);
    }

    public void testCreate_whenNoExpandNestedDocsWithRescore_thenDiversifyingQuery() {
        String fieldName = "field";
        float[] floatVectors = new float[3];
        int luceneK = 10;
        int k = 3;
        int rescoreK = 6;
        Query queryFilter = mock(Query.class);
        BitSetProducer parentFilter = mock(BitSetProducer.class);
        boolean expandNestedDocs = false;

        Query floatQuery = NestedKnnVectorQueryFactory.createNestedKnnVectorQuery(
            fieldName,
            floatVectors,
            luceneK,
            queryFilter,
            parentFilter,
            expandNestedDocs,
            k,
            rescoreK
        );
        assertEquals(OSDiversifyingChildrenFloatKnnVectorQuery.class, floatQuery.getClass());
        assertTrue(floatQuery instanceof DiversifyingChildrenFloatKnnVectorQuery);
    }
}
