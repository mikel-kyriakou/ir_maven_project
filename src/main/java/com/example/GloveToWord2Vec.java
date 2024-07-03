package com.example;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;

import java.io.File;

public class GloveToWord2Vec {

    public static void main(String[] args) throws Exception {
        File gloveFile = new File("src/main/resources/model.txt");
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(gloveFile);

        // Example usage
        System.out.println(word2Vec.wordsNearest("example", 10));
    }
}

