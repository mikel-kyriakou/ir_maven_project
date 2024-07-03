package com.example;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import javax.management.Query;

import com.example.IO;

public class TXTParsing {
    public static List<MyDoc> parse(String file) throws FileNotFoundException {
        try{
            //Parse txt file
            String txt_file = IO.ReadEntireFileIntoAString(file);
            String[] docs = txt_file.split("\n /// \n");
            System.out.println("Read: "+docs.length + " docs");

            //Parse each document from the txt file
            List<MyDoc> parsed_docs= new ArrayList<MyDoc>();
            for (String doc:docs){
                String[] adoc = doc.split("\n", 2);
                String id = adoc[0].strip();

                String[] contentSplit = adoc[1].split(":", 2);
                String title = contentSplit[0];
                String content = contentSplit[1];

                MyDoc mydoc = new MyDoc(id, title, content);
                parsed_docs.add(mydoc);
            }

            return parsed_docs;
        } catch (Throwable err) {
            err.printStackTrace();
            return null;
        }
    }

    public static List<MyQuery> parseQueries(String filePath) throws FileNotFoundException {
        try {
            File file = new File(filePath);
            Scanner scanner = new Scanner(file);
            scanner.useDelimiter("///");

            List<MyQuery> myQueryList = new ArrayList<MyQuery>();
            while (scanner.hasNext()) {
                String q = scanner.next().trim(); // Trim to remove leading and trailing whitespace
                String[] q_info = q.trim().split("\n");
                MyQuery query = new MyQuery(q_info[0], q_info[1]);
                myQueryList.add(query);
            }

            // String txt_file = IO.ReadEntireFileIntoAString(file);
            // String[] queries = txt_file.split("///");

            // List<MyQuery> myQueryList = new ArrayList<MyQuery>();
            // for(String q:queries){
            //     System.out.println("q: " + q);
            //     String[] q_info = q.trim().split("\n");
            //     MyQuery query = new MyQuery(q_info[0], q_info[1]);
            //     myQueryList.add(query);
            // }

            return myQueryList;

        } catch (Throwable err) {
            err.printStackTrace();
            return null;
        }
    }
}
