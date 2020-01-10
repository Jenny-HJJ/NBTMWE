package BTMWE;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;







import BTMWE.Biterm;
import BTMWE.TopicalWordComparator;
import GPUDMM.Document;

public class RatioBTMWE {
	public Set<String> wordSet;
	public int numTopic;
	public double alpha, beta;
	public int numIter;
	public int saveStep;
	public ArrayList<Biterm> bitermList;
	public int roundIndex;
	private Random rg;
	public double threshold;
	public double weight;
	public int topWords;
	public int filterSize;
	public String word2idFileName;
	public String similarityFileName;
	public int numDoc;
	public double[][] doc_biterm_pro;

	public Map<String, Integer> word2id;
	public Map<Integer, String> id2word;
	public Map<Integer, Double> wordIDFMap;
	public Map<Integer, Map<Integer, Double>> docUsefulWords;
	public ArrayList<ArrayList<Integer>> topWordIDList;
	public int vocSize;
	public int numBiterm;
	public ArrayList<int[]> bitremToWordIDList;
	public String initialFileName;  // we use the same initial for DMM-based model
	public double[][] phi;
	private double[] pz;
	private double[][] pdz;
	private double[][] topicProbabilityGivenBiterm;

	public ArrayList<Boolean> bitermGPUFlag; // wordGPUFlag.get(wordIndex) 
	public int[] assignmentList; // topic assignment for every biterm
	public  ArrayList<Map<Integer, Double>> bitermGPUInfo;

	private int[] mz; // have no associatiom with word and similar word
	private double[] nz; // [topic]; nums of words in every topic
	private double[][] nzw; // V_{.k}
	private Map<Integer, Map<Integer, Double>> schemaMap;

	public RatioBTMWE(ArrayList<Biterm> biterm_list, int num_topic, int num_iter, int save_step, double beta,
			double alpha, double threshold) {
		bitermList = biterm_list;
		numBiterm = bitermList.size();
		numTopic = num_topic;
		this.alpha = alpha;
		numIter = num_iter;
		saveStep = save_step;
		this.threshold = threshold;
		this.beta = beta;

	}

	public boolean loadWordMap(String filename) {
		try {
			FileInputStream fis = new FileInputStream(filename);
			InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
			BufferedReader reader = new BufferedReader(isr);
			String line;
			
			//construct word2id map
			while ((line = reader.readLine()) != null) {
				line = line.trim();
				String[] items = line.split(" ");
				//System.out.println(items);
				word2id.put(items[0], Integer.parseInt(items[1]));
				id2word.put(Integer.parseInt(items[1]), items[0]);
			}
			System.out.println("finish read wordmap and the num of word is " + word2id.size());
			return true;
		} catch (Exception e) {
			System.out.println("Error while reading other file:" + e.getMessage());
			e.printStackTrace();
			return false;
		}
	}

	/**
	 * Collect the similar words Map, not including the word itself
	 * 
	 * @param filename:
	 *            shcema_similarity filename
	 * @param threshold:
	 *            if the similarity is bigger than threshold, we consider it as
	 *            similar words
	 * @return
	 */
	public Map<Integer, Map<Integer, Double>> loadSchema(String filename, double threshold) {
		// laod the biterm related words
		int word_size = word2id.size();
		Map<Integer, Map<Integer, Double>> schemaMap = new HashMap<Integer, Map<Integer, Double>>();
		try {
			FileInputStream fis = new FileInputStream(filename);
			InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
			BufferedReader reader = new BufferedReader(isr);
			String line;
			int lineIndex = 0;
			double[][] similarity_matrix = new double[word_size][word_size]; 
			while ((line = reader.readLine()) != null) {
				line = line.trim();
				String[] items = line.split(" ");
				for (int i = 0; i < items.length; i++) {
					Double value = Double.parseDouble(items[i]);
					similarity_matrix[lineIndex][i] = value;
				}
				lineIndex++;
			}
			
			double count = 0.0;
			for (int i = 0; i< numBiterm; i++){
				Map<Integer, Double> tmpMap = new HashMap<Integer, Double>();
				int [] IDarray = bitremToWordIDList.get(i);
				for (int j = 0 ;j < word_size; j++)	{
					double v1 = similarity_matrix[IDarray[0]][j];
					double v2 = similarity_matrix[IDarray[1]][j];
					// System.out.println(v1 + "   " + v2);
					if (Double.compare(v1, threshold) > 0) {
						if (Double.compare(v2, threshold) > 0) {
							tmpMap.put(j, weight);
							
						}
					}
					
				}
				count += tmpMap.size();
				schemaMap.put(i, tmpMap);
			}
			 
			System.out.println("finish read schema, the avrage number of value is " + count / schemaMap.size());
			return schemaMap;
		} catch (Exception e) {
			System.out.println("Error while reading other file:" + e.getMessage());
			e.printStackTrace();
			return null;
		}
	}

	/**
	 * 
	 * @param wordID
	 * @param topic
	 * @return word probability given topic 
	 */
	public double getWordProbabilityUnderTopic(int wordID, int topic) {
		return (nzw[topic][wordID] + beta) / (nz[topic] + vocSize * beta);
	}

	public double getMaxTopicProbabilityForWord(int wordID) {
		double max = -1.0;
		for (int t = 0; t < numTopic; t++) {
			double tmp = getWordProbabilityUnderTopic(wordID, t);
			if (Double.compare(tmp, max) > 0) {
				max = tmp;
			}
		}
		return max;
	}

	/**
	 * update the p(z|b) for every iteration
	 */
	public void updateTopicProbabilityGivenBiterm() {
		// TODO we should update pz and phi information before
		compute_pz();  //update p(z)
		compute_phi();  //update p(w|z)
		for (int i = 0; i < numBiterm; i++) {
			int[] termIDArray = bitremToWordIDList.get(i); 
			double row_sum = 0.0;
			for (int j = 0; j < numTopic; j++) {
				topicProbabilityGivenBiterm[i][j] = pz[j] * phi[j][termIDArray[0]]* phi[j][termIDArray[1]]; // p(b,z) =p(z)*p(w1/z)*p(w2/z)
				row_sum += topicProbabilityGivenBiterm[i][j];
			}
			for (int j = 0; j < numTopic; j++) {
				topicProbabilityGivenBiterm[i][j] = topicProbabilityGivenBiterm[i][j] / row_sum;  //This is p(z|b)
			}
		}
	}
	
	
	public double findTopicMaxProbabilityGivenBiterm(int bitermID) {
		double max = -1.0;
		for (int i = 0; i < numTopic; i++) {
			double tmp = topicProbabilityGivenBiterm[bitermID][i];
			if (Double.compare(tmp, max) > 0) {
				max = tmp;
			}
		}
		return max;
	}

	public double getTopicProbabilityGivenBiterm(int topic, int bitermID) {
		return topicProbabilityGivenBiterm[bitermID][topic];
	}
	
	/**
	 * update GPU flag, which decide whether do GPU operation or not
	 * @param docID
	 * @param newTopic
	 */
	public void updateBitermGPUFlag(int bitermID, int newTopic) {
		// we calculate the p(t|b) and p_max(t|b) and use the ratio to decide we
		// use gpu for the word under this topic or not
	
		double maxProbability = findTopicMaxProbabilityGivenBiterm(bitermID);
		double ratio = getTopicProbabilityGivenBiterm(newTopic, bitermID) / maxProbability;
		double a = rg.nextDouble();
		Boolean docWordGPUFlag = (Double.compare(ratio, a) > 0);
		bitermGPUFlag.set(bitermID, docWordGPUFlag);
	}

	/**
	 * 
	 * @param filename for topic assignment for each document
	 */
	public void loadInitialStatus(String filename) {
		try {
			FileInputStream fis = new FileInputStream(filename);
			InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
			BufferedReader reader = new BufferedReader(isr);
			String line;
			while ((line = reader.readLine()) != null) {
				line = line.trim();
				String[] items = line.split(" ");
				assert(items.length == assignmentList.length);
				for (int i = 0; i < items.length; i++) {
					assignmentList[i] = Integer.parseInt(items[i]);
				}
				break;
			}

			System.out.println("finish loading initial status");
		} catch (Exception e) {
			System.out.println("Error while reading other file:" + e.getMessage());
			e.printStackTrace();
		}
	}

	public void ratioCount(Integer topic, Integer bitermID, int[] termIDArray, int flag) {
		mz[topic] += flag;
		for (int t = 0; t < termIDArray.length; t++) {
			int wordID = termIDArray[t];
			nzw[topic][wordID] += flag;
		}
		nz[topic] += flag;
		// we update gpu flag for every biterm before it change the counter
		// when adding numbers
		if (flag > 0) {
			updateBitermGPUFlag(bitermID, topic);
			boolean gpuFlag = bitermGPUFlag.get(bitermID);
			Map<Integer, Double> gpuInfo = new HashMap<>();
			if (gpuFlag) { // do gpu count
				if (schemaMap.containsKey(bitermID)) {
					Map<Integer, Double> valueMap = schemaMap.get(bitermID);
					// update the counter
					for (Map.Entry<Integer, Double> entry : valueMap.entrySet()) {
						Integer k = entry.getKey();
						double v = weight;
						nzw[topic][k] += v;
						nz[topic] += v;
						gpuInfo.put(k, v);
						} // end loop for similar words
				} // end containsKey
				else { // schemaMap don't contain the word
						// the word doesn't have similar words, the infoMap is empty
						// we do nothing
					}
				} else { // the gpuFlag is False
					// it means we don't do gpu, so the gouInfo map is empty
				}
			bitermGPUInfo.set(bitermID, gpuInfo); // we update the gpuinfo map
		} else { // we do subtraction according to last iteration information
			Map<Integer, Double> gpuInfo = bitermGPUInfo.get(bitermID);
			if (gpuInfo.size() > 0) {
				for (int similarWordID : gpuInfo.keySet()) {
					// double v = gpuInfo.get(similarWordID);
					double v = weight;
					nzw[topic][similarWordID] -= v;
					nz[topic] -= v;
						// if(Double.compare(0, nzw[topic][wordID]) > 0){
						// System.out.println( nzw[topic][wordID]);
						// }
					}
				}
			}
	}

	public void normalCount(Integer topic, int[] termIDArray, Integer flag) {
		mz[topic] += flag;
		for (int t = 0; t < termIDArray.length; t++) {
			int wordID = termIDArray[t];
			nzw[topic][wordID] += flag;
			nz[topic] += flag;
		}
	}
	
	
	public void initNewModel() {
		bitermGPUFlag = new ArrayList<>();
		bitremToWordIDList = new ArrayList<int[]>();
		word2id = new HashMap<String, Integer>();
		id2word = new HashMap<Integer, String>();
		wordIDFMap = new HashMap<Integer, Double>();
		docUsefulWords = new HashMap<Integer, Map<Integer, Double>>();
		wordSet = new HashSet<String>();
		topWordIDList = new ArrayList<>();
		assignmentList = new int[numBiterm];
		bitermGPUInfo = new ArrayList<>();
		rg = new Random();
		// construct vocabulary
		loadWordMap(word2idFileName);

		vocSize = word2id.size();
		phi = new double[numTopic][vocSize];
		pz = new double[numTopic];
		pdz = new double[numDoc][numTopic];

		topicProbabilityGivenBiterm = new double[vocSize][numBiterm];

		for (int i = 0; i < bitermList.size(); i++) {
			Biterm biterm = bitermList.get(i);
			int[] termIDArray = new int[2];
			 
			HashMap<Integer, Double> docWordGPUInfo = new HashMap<Integer, Double>();
			termIDArray[0] = word2id.get(biterm.word1);
			termIDArray[1] = word2id.get(biterm.word2);
			bitermGPUFlag.add(false); // initial for False for every biterm
			bitremToWordIDList.add(termIDArray);
			bitermGPUInfo.add(docWordGPUInfo);
		}

		// init the counter
		mz = new int[numTopic];
		nz = new double[numTopic];
		nzw = new double[numTopic][vocSize];
	}

	public void init_BTMWE() {
//		 schemaMap = loadSchema("E:\\pythonWorkspace\\GPUBTM\\data\\qa_word_similarity.txt",threshold);
		schemaMap = loadSchema(similarityFileName, threshold);
		loadInitialStatus(initialFileName);

		for (int d = 0; d < bitremToWordIDList.size(); d++) {
			int[] termIDArray = bitremToWordIDList.get(d);
			int topic = assignmentList[d];
//			 int topic = rg.nextInt(numTopic);
//			 assignmentList[d] = topic;
			normalCount(topic, termIDArray, +1);
		}
		System.out.println("finish init_MU!");
	}

	private static long getCurrTime() {
		return System.currentTimeMillis();
	}

	public void run_iteration() {
		double time_cost =  0.0;

		for (int iteration = 1; iteration <= numIter; iteration++) {
			//System.out.println(iteration + "th iteration begin");

			long _s = getCurrTime();
			// getTopWordsUnderEachTopicGivenCurrentMarkovStatus();
			updateTopicProbabilityGivenBiterm();
			for (int s = 0; s < bitremToWordIDList.size(); s++) {

				int[] termIDArray = bitremToWordIDList.get(s);
				int preTopic = assignmentList[s];

				ratioCount(preTopic, s, termIDArray, -1);

				double[] pzDist = new double[numTopic];
				for (int topic = 0; topic < numTopic; topic++) {
					double pz = 1.0 * (mz[topic] + alpha);
					double value = 1.0;
					for (int t = 0; t < termIDArray.length; t++) {
						int termID = termIDArray[t];
						value *= (nzw[topic][termID] + beta) / (nzw[topic][termID] + vocSize * beta);
					}
					value = pz * value;
					pzDist[topic] = value;
				}

				for (int i = 1; i < numTopic; i++) {
					pzDist[i] += pzDist[i - 1];
				}

				double u = rg.nextDouble() * pzDist[numTopic - 1];
				int newTopic = -1;
				for (int i = 0; i < numTopic; i++) {
					if (Double.compare(pzDist[i], u) >= 0) {
						newTopic = i;
						break;
					}
				}
				// update
				assignmentList[s] = newTopic;
				ratioCount(newTopic, s, termIDArray, +1);

			}
			long _e = getCurrTime();
			//System.out.println(iteration + "th iter finished and every iterration costs " + (_e - _s) + "ms! "
			//		+ numTopic + " topics round " + roundIndex);
			time_cost += (_e - _s);
		}
		System.out.println("average time_costs of " + numTopic + " topics round " + roundIndex + " is " + time_cost/numIter +" ms! ");
				 	 
		
	}

	public void run_BTMWE(ArrayList<HashMap<Integer, Double>> doc_biterm_pro, String flag) {
		initNewModel();
		init_BTMWE();
		run_iteration();
		saveModel(doc_biterm_pro, flag);
	}

	public void saveModel(ArrayList<HashMap<Integer, Double>> doc_biterm_pro, String flag) {

		compute_phi();
		compute_pz();
		compute_pzd(doc_biterm_pro);
		saveModelPz(flag + "_theta.txt");
		saveModelPhi(flag + "_phi.txt");
		saveModelWords(flag + "_words.txt");
		saveModelAssign(flag + "_assign.txt");
		saveModelPdz(flag + "_pdz.txt");
	}

	public void compute_phi() {
		for (int i = 0; i < numTopic; i++) {
			double sum = 0.0;
			for (int j = 0; j < vocSize; j++) {
				sum += nzw[i][j];
			}
			for (int j = 0; j < vocSize; j++) {
				phi[i][j] = (nzw[i][j] + beta) / (sum + vocSize * beta);
			}
		}
	}

	public void compute_pz() {
		double sum = 0.0;
		for (int i = 0; i < numTopic; i++) {
			sum += nz[i];
		}
		for (int i = 0; i < numTopic; i++) {
			pz[i] = 1.0 * (nz[i] + alpha) / (sum + numTopic * alpha);
		}
	}

	public void compute_pzd(ArrayList<HashMap<Integer, Double>> doc_biterm_pro) {
		// p(z/d) = sum(p(z/b)p(b/d)
		double[][] pwz = new double[numBiterm][numTopic]; // pwz[z/b] = p()
		for (int i = 0; i < numBiterm; i++) {
			double row_sum = 0.0;
			int [] IDArray = bitremToWordIDList.get(i);
			for (int j = 0; j < numTopic; j++) {
				pwz[i][j] = pz[j] * phi[j][IDArray[0]]*phi[j][IDArray[1]];
				row_sum += pwz[i][j];
			}
			for (int j = 0; j < numTopic; j++) {
				pwz[i][j] = pwz[i][j] / row_sum;
			}

		}

		for (int i = 0; i < doc_biterm_pro.size(); i++) {
			HashMap<Integer, Double> doc_b_pro = doc_biterm_pro.get(i);
			
			for(Entry<Integer, Double> entry : doc_b_pro.entrySet())
			{
				int b_index = entry.getKey();
				double b_d_pro = entry.getValue();
				for (int j = 0; j < numTopic; j++) {
					pdz[i][j] += pwz[b_index][j]*b_d_pro;
				}
			}
			double row_sum = 0.0;
			for(int j = 0; j < numTopic; j++){
				row_sum += pdz[i][j];
			}
			for (int j = 0; j < numTopic; j++) {
				pdz[i][j] = pdz[i][j] / row_sum;
				
			}
		}
	}

	public boolean saveModelAssign(String filename) {
		try {
			PrintWriter out = new PrintWriter(filename);

			for (int i = 0; i < numBiterm; i++) {
				int topic = assignmentList[i];
				for (int j = 0; j < numTopic; j++) {
					int value = -1;
					if (j == topic) {
						value = 1;
					} else {
						value = 0;
					}
					out.print(value + " ");
				}
				out.println();
			}
			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saving assign list: " + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}

	public boolean saveModelPdz(String filename) {
		try {
			PrintWriter out = new PrintWriter(filename);

			for (int i = 0; i < numBiterm; i++) {
				for (int j = 0; j < numTopic; j++) {
					out.print(pdz[i][j] + " ");
				}
				out.println();
			}

			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saving p(z|d) distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}

	public boolean saveModelPz(String filename) {
		// return false;
		try {
			PrintWriter out = new PrintWriter(filename);

			for (int i = 0; i < numTopic; i++) {
				out.print(pz[i] + " ");
			}
			out.println();

			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saving pz distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}

	public boolean saveModelPhi(String filename) {
		try {
			PrintWriter out = new PrintWriter(filename);

			for (int i = 0; i < numTopic; i++) {
				for (int j = 0; j < vocSize; j++) {
					out.print(phi[i][j] + " ");
				}
				out.println();
			}
			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saving word-topic distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}

	public boolean saveModelWords(String filename) {
		try {
			PrintWriter out = new PrintWriter(filename, "UTF8");
			for (String word : word2id.keySet()) {
				int id = word2id.get(word);
				out.println(word + "," + id);
			}
			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saveing words list: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}

	public static void main(String[] args) {
		String path = "D:/黄佳佳/JAVA Code/BTM_WE/data/corpus/SinaWeiBo/";
		ArrayList<Biterm> biterm_list = Biterm.LoadCorpus(path +"biterm_list-8.txt");
		ArrayList<Document> doc_list = Document.LoadCorpus(path +"documents.txt");
		ArrayList<HashMap<Integer, Double>> doc_biterm_pro = Biterm.GetDocBitermPro(biterm_list, doc_list);
		String write_path = "D:/黄佳佳/JAVA Code/BTM_WE/data/result/SinaWeiBo/BTMWE-tf=8/";
		System.out.println(biterm_list.size());
		//here
		int num_iter = 500, save_step = 200;
		double beta = 0.1;
		String similarityFileName = path + "/word_similarity.txt";
		double weight = 0.4;
		double threshold = 0.4;
		int filterSize = 20;
		
		for (int round = 1; round <= 1; round += 1) {
			for (int num_topic = 20; num_topic <= 60; num_topic += 20) {
				String initialFileName = write_path + "/topic/" + num_topic + "_initial_status.txt";
//				String initialFileName = "../data/topic" + num_topic + "_qa_random_initial_status.txt";
				double alpha = 0.05;
				RatioBTMWE btmwe = new RatioBTMWE(biterm_list, num_topic, num_iter, save_step, beta, alpha, threshold);
				btmwe.word2idFileName = path + "/word_id.txt";
				btmwe.topWords = 100;
				
				//here
				btmwe.filterSize = filterSize;
				btmwe.roundIndex = round;
				btmwe.initialFileName = initialFileName;
				btmwe.similarityFileName = similarityFileName;
				btmwe.weight = weight;
				btmwe.numDoc = doc_biterm_pro.size();
				btmwe.initNewModel();
				btmwe.init_BTMWE();
				btmwe.run_iteration();
				String flag = round+"_round_"+num_topic;
				flag =  write_path + "/topic/" + flag;
				btmwe.saveModel(doc_biterm_pro, flag);
				
			}
		}
	}
}

/**
 * Comparator to rank the words according to their probabilities.
 */
class TopicalWordComparator implements Comparator<Integer> {
	private double[] distribution = null;

	public TopicalWordComparator(double[] distribution2) {
		distribution = distribution2;
	}

	@Override
	public int compare(Integer w1, Integer w2) {
		if (distribution[w1] < distribution[w2]) {
			return -1;
		} else if (distribution[w1] > distribution[w2]) {
			return 1;
		}
		return 0;
	}
}

 