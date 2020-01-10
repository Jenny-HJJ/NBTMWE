package BTMWE;
 

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import GPUDMM.Document;

public class Biterm {
  
  public String [] words;
  public int id;
  public String word1;
  public String word2;
  
  
  public Biterm(int docid, String word1, String word2){
    this.id = docid;
    this.word1 = word1;
    this.word2 = word2;
  }
  
  public static ArrayList<Biterm> LoadCorpus(String filename){
	    try{
	      FileInputStream fis = new FileInputStream(filename);
	      InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
	      BufferedReader reader = new BufferedReader(isr);
	      String line;
	      ArrayList<Biterm> biterm_list = new ArrayList();
	      int count = 0;
	      while((line = reader.readLine()) != null){
	        line = line.trim();
	        
	        String[] items = line.split(",");
	        if (items.length == 4){        	
	        int docid = count;
	        count += 1;
	        //System.out.println( items[0] + " ------ " + items[1]);
	        Biterm doc = new Biterm(docid, items[0], items[1]);
	        biterm_list.add(doc);
	        }
	        
	      }
	      System.out.println("there are " + biterm_list.size() + " biterms.");
	      return biterm_list;
	    }
	    catch (Exception e){
	      System.out.println("Error while reading other file:" + e.getMessage());
	      e.printStackTrace();
//	      return false;
	  }
	    return null;
	    
	  }
  
  public static ArrayList<HashMap<Integer, Double>>  GetDocBitermPro(ArrayList<Biterm> biterm_list, ArrayList<Document>doc_list){
	  ArrayList<HashMap<Integer, Double>> doc_biterm_pro = new ArrayList<>();
	  for( int d_index = 0; d_index < doc_list.size(); d_index ++){
		  HashMap<Integer, Double> doc_b = new HashMap<Integer, Double>();
		  Map<String, Integer> map = new HashMap<>();
		  String [] content = doc_list.get(d_index).words;
		   for (String str : content) {
		      Integer num = map.get(str);
		      map.put(str, num == null ? 1 : num + 1);
		    }
		   // System.out.println(map);
		    double all_tf = 0.0;
		    for(int b_index = 0; b_index<biterm_list.size(); b_index ++){
		    	String word1 = biterm_list.get(b_index).word1;
		    	String word2 = biterm_list.get(b_index).word2;
		    	int tf = 0;
		    	try{
		    		int tf1 = map.get(word1);
		    		int tf2 = map.get(word2);
		    		tf = (tf1>tf2) ? tf1:tf2;
		    	}   
		    	catch (Exception e){	    
		    	}
		    	if (tf > 0)  {
		    	 doc_b.put(b_index, tf/1.0);
		    	 all_tf += tf/1.0;
		    	} //end if
		    } // end for b_index
		   for(Entry<Integer, Double> entry : doc_b.entrySet())  {
			  doc_b.put(entry.getKey(), entry.getValue()/all_tf);
			  //System.out.print(entry.getKey() + "  " +  entry.getValue() + "   ");
		  } //end for  entry
  
//	      System.out.println("============"); 
		  doc_biterm_pro.add(doc_b);	   
	  } //end for d_index
	  return doc_biterm_pro;	   
  }
 
  

  public static void main(String[] args) {
    // TODO �Զ����ɵķ������
	String path = "D:/黄佳佳/JAVA Code/BTM_WE/data/corpus/Web Snippets/";
	ArrayList<Biterm> biterm_list = Biterm.LoadCorpus(path +"biterms.txt");
	ArrayList<Document> doc_list = Document.LoadCorpus(path +"test.txt");
	System.out.println(doc_list.size());
	ArrayList<HashMap<Integer, Double>> doc_biterm_pro = Biterm.GetDocBitermPro( biterm_list, doc_list);
     
  }
}

