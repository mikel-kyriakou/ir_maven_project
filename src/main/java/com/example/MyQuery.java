package com.example;

public class MyQuery {
    private String id;
    private String queryContent;

    public MyQuery(String id, String queryContent) {
        this.id = id;
        this.queryContent = queryContent;
    }

    @Override
    public String toString() {
        String ret = "MyQuery{"
                + "\n\tid: " + id
                + "\n\tqueryContent: " + queryContent;               
        return ret + "\n}";
    }

    //---- Getters & Setters definition ----
    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getQueryContent() {
        return queryContent;
    }

    public void setQueryContent(String queryContent) {
        this.queryContent = queryContent;
    }

}
