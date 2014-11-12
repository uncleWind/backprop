package backprop;

import feedforward.*;
import learningset.*; 
import logger.*;
import matrixutil.Matrix;
import xmlparser.*;

import java.util.ArrayList; 

import pruning.PruneBySelection;
import pruning.PruneIncremental;

public class BackProp {
	public static void main(String[] args) throws IllegalAccessException {         
		//Testing.
		for(int i = 0; i < 25; ++i) {
		
		String XMLFilePath = "C:\\Users\\Krzysztof\\workspace\\backpropmachine\\xmlsources\\log_complex.xml";
        XMLNetworkFileParser parser = new XMLNetworkFileParser(XMLFilePath); 
        parser.parse(); 
  
        Network network = parser.getNetwork(); 
        LearningSet learningSet = parser.getLearningSet();
        
        /* Use the code below if you have weight matrices included in
         * the XML source file.
         */
        /*
        ArrayList<Matrix> weightMatrices = parser.getWeightMatrices();
        network.setParsedWeightMatricesSet(weightMatrices);
        */
          
        ArrayList<double[]> learningInputs = learningSet.getLearningInputsSet(); 
        ArrayList<double[]> learningOutputs = learningSet.getLearningOutputsSet(); 
        
        String LoggerFilePath = "C:\\Users\\Krzysztof\\workspace\\backpropmachine";
		Logger.setLoggerDirectory(LoggerFilePath);
		Logger.prepareDirectoryForLogger();
		Logger.enableAllLogging();
		Logger.disableMatrixLogging();
		Logger.disableArrayLogging();
		//Logger.disableLogging(LoggerType.LOG_CONV);
		Logger.disableLogging(LoggerType.LOG_NETWORK);
		//Logger.disableLogging(LoggerType.LOG_LEARNING);
        
		printOutputsFor(network, learningInputs);
        
        System.out.println("Learning..."); 
        network.setOneTimeEpochLimit(5000);
        
        //PruneIncremental prunerIncr = new PruneIncremental(learningInputs, 
        //		learningOutputs, network);
        //prunerIncr.prune();
        network.trainWith(learningInputs, learningOutputs);
        
        System.out.println("After incremental pruning: ");
        printOutputsFor(network, learningInputs);
        
        //PruneBySelection prunerSel = new PruneBySelection(learningInputs, 
        //		learningOutputs, network);
        //prunerSel.prune();
        
        //System.out.println("After selective pruning: ");
        //printOutputsFor(network, learningInputs);
        
        /* Use the code below if you have provided another set of learning
         * tuples in another file.
         */
        
        /*
        XMLFilePath = "C:\\Users\\Krzysztof\\workspace\\backpropmachine\\xmlsources\\log_complex_secondls.xml";
        parser = new XMLNetworkFileParser(XMLFilePath);
        parser.addParsingRequest(XMLRequestType.REQ_LEARNING);
        parser.injectNetworkToHandler(network);
        parser.parseWithRequests();
        
        learningSet = parser.getLearningSet();
        learningInputs = learningSet.getLearningInputsSet();
        learningOutputs = learningSet.getLearningOutputsSet();
        
        printOutputsFor(network, learningInputs);
        
        System.out.println("Learning with second set...");
        network.trainWith(learningInputs, learningOutputs);
        
        printOutputsFor(network, learningInputs);
        */
        
        Logger.writeNetworkToXML(network);
        
        /* Testing set for log_complex learning set.
    	double[] testInputPattern = {0.98, 0.98, 0.98, 0.02, 0.98, 0.98, 0.02, 0.02};
    	network.presentPattern(testInputPattern);
    	double[] result = network.recieveOutput();
    	
    	System.out.print("Testing on: ");
    	for(int i = 0; i < testInputPattern.length; ++i) {
        	System.out.print(testInputPattern[i] + " ");
        }
        System.out.print("\n");
        
        System.out.print("Results: ");
        for(int i = 0; i < result.length; ++i) {
        	System.out.print(result[i] + " ");
        }
        System.out.print("\n");
        */
        
        //Testing.
		}
    }
	
	private static void printOutputsFor(Network network, 
			ArrayList<double[]> learningInputs) {
        for(double[] pattern : learningInputs) { 
			network.presentPattern(pattern);
			
			double[] inputs = network.getLayer(0).getRawValueNeurons();
			double[] results = network.recieveOutput(); 
			double[] decoratedResults = network.recieveDecoratedOutput();
			
			ArrayList<double[]> midLayers = new ArrayList<>();
			int middleLayersCount = network.size() - 2;
			for(int i = 0; i < middleLayersCount; ++i) {
				midLayers.add(0, network.getLayer(i + 1).getOutputValueNeurons());
			}
            
            System.out.print("Inputs: ");
            for(int i = 0; i < inputs.length; ++i) {
            	System.out.print(inputs[i] + " ");
            }
            for(int i = 0; i < midLayers.size(); ++i) {
            	System.out.print(", values for middle layer no. " + (i + 1) + ": ");
            	double[] midLayer = midLayers.get(i);
            	for(double neuronValue : midLayer) {
            		System.out.print(neuronValue + " ");
            	}
            }
            System.out.print(", outputs: ");
            for(int i = 0; i < results.length; ++i) {
            	System.out.print(results[i] + " ");
            }
            System.out.println(", decorated: ");
            for(int i = 0; i < decoratedResults.length; ++i) {
            	System.out.print(decoratedResults[i] + " ");
            }
            System.out.print("\n");
        }
	}
}
