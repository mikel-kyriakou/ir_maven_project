package com.example;

public class MyDoc {
    private String id;
    private String title;
    private String content;

    public MyDoc(String id, String title, String content) {
        this.id = id;
        this.title = title;
        this.content = content;
    }

    @Override
    public String toString() {
        String ret = "MyDoc{"
                + "\n\tid: " + id
                + "\n\ttitle: " + title
                + "\n\tcontent: " + content;                
        return ret + "\n}";
    }

    //---- Getters & Setters definition ----
    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }
}
