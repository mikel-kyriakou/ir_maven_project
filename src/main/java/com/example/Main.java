package com.example;

public class Main {
    public static void main(String[] args) {
        System.out.println("Hello world!");

        RankingTest rk = new RankingTest();
        RankingTest1 rk1 = new RankingTest1();
        RankingTest2 rk2 = new RankingTest2();

        try {
            // rk.testRankingWithTFIDFAveragedWordEmbeddings();
            rk1.testRankingWithTFIDFAveragedWordEmbeddings();
            // rk2.testRankingWithTFIDFAveragedWordEmbeddings();
            // mr.fit_model();
        } catch (Exception e) {
            System.out.println(e);
        }
    }
}