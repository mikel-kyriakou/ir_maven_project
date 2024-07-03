package com.example;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;



public class IO {
    //     public static StringBuffer ReadFileIntoAStringLineByLine(String file) throws IOException {

    //     BufferedReader bufferedReader = new BufferedReader(new FileReader(file));

    //     StringBuffer stringBuffer = new StringBuffer();
    //     String line = null;

    //     while ((line = bufferedReader.readLine()) != null) {
    //         stringBuffer.append(line).append("\n");
    //     }

    //     bufferedReader.close();

    //     return stringBuffer;
    // }

    public static String ReadEntireFileIntoAString(String file) throws FileNotFoundException {
        Scanner scanner = new Scanner(new File(file));
        scanner.useDelimiter("\\A"); //\\A stands for :start of a string
        String entireFileText = scanner.next();
        scanner.close();
        return entireFileText;
    }
}
