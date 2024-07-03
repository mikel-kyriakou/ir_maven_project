package com.example;

import com.google.common.io.Files;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.*;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Ranking tests evaluating different {@link Similarity} implementations against {@link WordEmbeddingsSimilarity}
 */
public class RankingTest2 {

  @Test
  public void testRankingWithDifferentSimilarities() throws Exception {
    Path path = Paths.get(Files.createTempDir().toURI());
    Directory directory = FSDirectory.open(path);

    try {

      Collection<Similarity> similarities = Arrays.asList(new ClassicSimilarity(), new BM25Similarity(2.5f, 0.2f),
          new LMJelinekMercerSimilarity(0.1f), new MultiSimilarity(new Similarity[] {new BM25Similarity(), new LMJelinekMercerSimilarity(0.1f)}));
      for (Similarity similarity : similarities) {

        IndexWriterConfig config = new IndexWriterConfig(new WhitespaceAnalyzer());
        config.setSimilarity(similarity);

        IndexWriter writer = new IndexWriter(directory, config);
        FieldType ft = new FieldType(TextField.TYPE_STORED);
        ft.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
        ft.setTokenized(true);
        ft.setStored(true);
        ft.setStoreTermVectors(true);
        ft.setStoreTermVectorOffsets(true);
        ft.setStoreTermVectorPositions(true);

        // Document doc1 = new Document();
        // doc1.add(new Field("title", "riemann bernhard - life and works of bernhard riemann", ft));

        // Document doc2 = new Document();
        // doc2.add(new Field("title", "thomas bernhard biography - bio and influence in literature", ft));

        // Document doc3 = new Document();
        // doc3.add(new Field("title", "riemann hypothesis - a deep dive into a mathematical mystery", ft));

        // Document doc4 = new Document();
        // doc4.add(new Field("title", "bernhard bernhard bernhard bernhard bernhard bernhard bernhard bernhard bernhard bernhard", ft));

        // writer.addDocument(doc1);
        // writer.addDocument(doc2);
        // writer.addDocument(doc3);
        // writer.addDocument(doc4);

        String txtfile = "resources//documents.txt";

        List<MyDoc> docs = TXTParsing.parse(txtfile);
        
        for(MyDoc doc : docs){
          Document document = new Document();
          document.add(new Field("id", doc.getId(), ft));
          document.add(new Field("title", doc.getTitle(), ft));
          document.add(new Field("content", doc.getContent(), ft));
          
          writer.addDocument(document);
        }

        writer.commit();

        IndexReader reader = DirectoryReader.open(directory);
        try {
          IndexSearcher searcher = new IndexSearcher(reader);
          Terms fieldTerms = MultiFields.getTerms(reader, "title");

          System.out.println(similarity);
          searcher.setSimilarity(similarity);
          QueryParser parser = new QueryParser("title", new WhitespaceAnalyzer());
          String queryString = "bernhard riemann influence";
          Query query = parser.parse(queryString);
          TopDocs hits = searcher.search(query, 10);
          for (int i = 0; i < hits.scoreDocs.length; i++) {
            ScoreDoc scoreDoc = hits.scoreDocs[i];
            Document doc = searcher.doc(scoreDoc.doc);

            String title = doc.get("title");
            System.out.println(title + " : " + scoreDoc.score);
            if (similarity instanceof ClassicSimilarity) {
              double[] queryVector = new double[(int) fieldTerms.size()];
              Arrays.fill(queryVector, 0d);
              for (String queryTerm : queryString.split(" ")) {
                TermsEnum iterator = fieldTerms.iterator();
                int j = 0;
                BytesRef term;
                while ((term = iterator.next()) != null) {

                  TermsEnum.SeekStatus seekStatus = iterator.seekCeil(term);
                  if (seekStatus.equals(TermsEnum.SeekStatus.END)) {
                    iterator = fieldTerms.iterator();
                  }
                  if (seekStatus.equals(TermsEnum.SeekStatus.FOUND)) {
                    if (term.utf8ToString().equals(queryTerm)) {
                      double tf = iterator.totalTermFreq();
                      double docFreq = iterator.docFreq();
                      queryVector[j] = VectorizeUtils.tfIdf(reader.numDocs(), tf, docFreq);
                    }
                  }
                  j++;
                }
              }

              Terms docTerms = reader.getTermVector(scoreDoc.doc, "title");
              double[] documentVector = VectorizeUtils.toSparseTFIDFDoubleArray(docTerms, fieldTerms, reader.numDocs());

              System.out.println("cosineSimilarity=" + VectorizeUtils.cosineSimilarity(queryVector, documentVector));
            }
          }

        } finally {
          writer.commit();
          writer.close();
          reader.close();
        }
      }

    } finally {
      directory.close();
    }
  }

  @Test
  public void testRankingWithTFIDFAveragedWordEmbeddings() throws Exception {
    Path path = Paths.get(Files.createTempDir().toURI());
    Directory directory = FSDirectory.open(path);

    try {

      IndexWriterConfig config = new IndexWriterConfig(new WhitespaceAnalyzer());

      IndexWriter writer = new IndexWriter(directory, config);
      FieldType ft = new FieldType(TextField.TYPE_STORED);
      ft.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
      ft.setTokenized(true);
      ft.setStored(true);
      ft.setStoreTermVectors(true);
      ft.setStoreTermVectorOffsets(true);
      ft.setStoreTermVectorPositions(true);
/*
      String pathname = "src/test/resources/enwiki-20180820-pages-articles14.xml-p7697599p7744799";
      String languageCode = "en";
      File dump = new File(pathname);
      WikipediaImport wikipediaImport = new WikipediaImport(dump, languageCode, true);
      wikipediaImport.importWikipedia(writer, ft);
*/
      // Document doc1 = new Document();
      // doc1.add(new Field("title", "riemann bernhard - life and works of bernhard riemann", ft));

      // Document doc2 = new Document();
      // doc2.add(new Field("title", "thomas bernhard biography - bio and influence in literature", ft));

      // Document doc3 = new Document();
      // doc3.add(new Field("title", "riemann hypothesis - a deep dive into a mathematical mystery", ft));

      // Document doc4 = new Document();
      // doc4.add(new Field("title", "bernhard bernhard bernhard bernhard bernhard bernhard bernhard bernhard bernhard bernhard", ft));

      // writer.addDocument(doc1);
      // writer.addDocument(doc2);
      // writer.addDocument(doc3);
      // writer.addDocument(doc4);

      String txtfile = "src\\main\\resources\\documents.txt";

      List<MyDoc> docs = TXTParsing.parse(txtfile);
      
      for(MyDoc doc : docs){
        Document document = new Document();
        document.add(new Field("id", doc.getId(), ft));
        document.add(new Field("title", doc.getTitle(), ft));
        document.add(new Field("content", doc.getContent(), ft));
        
        writer.addDocument(document);
      }

      writer.commit();

      IndexReader reader = DirectoryReader.open(writer);
      String fieldName = "content";

      File gloveFile = new File("src/main/resources/model.txt");
      Word2Vec vec = WordVectorSerializer.readWord2VecModel(gloveFile);

      String resultsFileAvVector2 = "src\\main\\resources\\resultsAverageVector2.txt";
      String resultsFileTFIDVecotr2 = "src\\main\\resources\\resultsTFIDVector2.txt";

      BufferedWriter bw1 = new BufferedWriter(new FileWriter(resultsFileAvVector2));
      BufferedWriter bw2 = new BufferedWriter(new FileWriter(resultsFileTFIDVecotr2));

      try {

        System.out.println(0);
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setSimilarity(new WordEmbeddingsSimilarity(vec, fieldName, WordEmbeddingsSimilarity.Smoothing.MEAN));

        System.out.println(1);

        List<MyQuery> queries = TXTParsing.parseQueries("src\\main\\resources\\queries.txt");
        for(MyQuery q:queries){
          String queryString = q.getQueryContent();

          Terms fieldTerms = MultiFields.getTerms(reader, fieldName);  
  
          INDArray denseAverageTFIDFQueryVector = Nd4j.zeros(vec.getLayerSize());
          Map<String, Double> tfIdfs = new HashMap<>();
  
          String[] split = queryString.split(" ");
    
          // for (String queryTerm : split) {
          //   TermsEnum iterator = fieldTerms.iterator();
          //   BytesRef term;
          //   while ((term = iterator.next()) != null) {
          //     TermsEnum.SeekStatus seekStatus = iterator.seekCeil(term);
          //     if (seekStatus.equals(TermsEnum.SeekStatus.END)) {
          //       iterator = fieldTerms.iterator();
          //     }
          //     if (seekStatus.equals(TermsEnum.SeekStatus.FOUND)) {
          //       String string = term.utf8ToString();
          //       if (string.equals(queryTerm)) {
          //         double tf = iterator.totalTermFreq();
          //         double docFreq = iterator.docFreq();
          //         double tfIdf = VectorizeUtils.tfIdf(reader.numDocs(), tf, docFreq);
          //         tfIdfs.put(string, tfIdf);
          //       }
          //     }
          //   }
          //   Double n = tfIdfs.get(queryTerm);

          //   INDArray vector = vec.getLookupTable().vector(queryTerm);
          //   denseAverageTFIDFQueryVector.addi(vector.div(n));
          // }

          List<String> filteredWords = new ArrayList<>();
          for (String word : split) {
              if (vec.hasWord(word)) {
                  filteredWords.add(word);
              }
          }
  

          System.out.println(2);
          
          System.out.println(split == null);

          System.out.println("2a");

          INDArray denseAverageQueryVector = vec.getWordVectorsMean(filteredWords);

          System.out.println(3);
    
          QueryParser parser = new QueryParser(fieldName, new WhitespaceAnalyzer());

          System.out.println(4);
  
          Query query = parser.parse(queryString);

          System.out.println("a");
  
          TopDocs hits = searcher.search(query, 50);

          System.out.println("b");
          
          for (int i = 0; i < hits.scoreDocs.length; i++) {
            ScoreDoc scoreDoc = hits.scoreDocs[i];
            Document doc = searcher.doc(scoreDoc.doc);

  
            System.out.println(i+1 + ": " + doc.get("title") + " : " + scoreDoc.score);

            System.out.println(1);
  
            Terms docTerms = reader.getTermVector(scoreDoc.doc, fieldName);
  
            System.out.println(2);

            INDArray denseAverageDocumentVector = VectorizeUtils.toDenseAverageVector(docTerms, vec);
            INDArray denseAverageTFIDFDocumentVector = VectorizeUtils.toDenseAverageTFIDFVector(docTerms, reader.numDocs(), vec);

            System.out.println(3);

            Double sim1 = Transforms.cosineSim(denseAverageQueryVector, denseAverageDocumentVector);
            Double sim2 = Transforms.cosineSim(denseAverageTFIDFQueryVector, denseAverageTFIDFDocumentVector);

            System.out.println(4);
  
            System.out.println("cosineSimilarityDenseAvg=" + Transforms.cosineSim(denseAverageQueryVector, denseAverageDocumentVector));
            System.out.println("cosineSimilarityDenseAvgTFIDF=" + Transforms.cosineSim(denseAverageTFIDFQueryVector, denseAverageTFIDFDocumentVector));
            System.out.println();

            bw1.write(q.getId() + "\t" + "Q0" + "\t" + doc.get("id") + "\t" + "0" + "\t" + sim1 + "\t" + "VSM" + "\n");
            bw2.write(q.getId() + "\t" + "Q0" + "\t" + doc.get("id") + "\t" + "0" + "\t" + sim2 + "\t" + "VSM" + "\n");

            System.out.println(5);

          }
          System.out.println();

        }

      } catch(Exception e){
        System.err.println(e);
      } finally {
        System.out.println("Finally");
        WordVectorSerializer.writeWord2VecModel(vec, "target/ch5w2v.zip");
        writer.deleteAll();
        writer.commit();
        writer.close();
        reader.close();
        bw1.close();
        bw2.close();
      }

    } finally {
      directory.close();
    }
  }

}