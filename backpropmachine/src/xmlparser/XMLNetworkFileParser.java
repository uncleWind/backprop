package xmlparser;

import feedforward.*;
import learningset.*;
import matrixutil.Matrix;

import javax.xml.parsers.*;

import org.xml.sax.*;
import org.xml.sax.helpers.*;

import java.io.*;
import java.util.ArrayList;

public class XMLNetworkFileParser {
    private Network parsedNetwork;
    private LearningSet parsedLearningSet;
    private ArrayList<Matrix> parsedWeightMatrices;
    
    private ArrayList<XMLRequestType> parseRequests = new ArrayList<>();
    
    final private String filePath;
    private SAXParserFactory factory;
    private SAXParser parser;
    private XMLReader reader;
    private DefaultHandler handler;
    
    private boolean isNetworkInjectionRaised = false;
    private Network injectedNetwork;
     
    public XMLNetworkFileParser(String filePath) {
        this.filePath = filePath;
        this.factory = SAXParserFactory.newInstance();
        try {
            this.parser = factory.newSAXParser(); 
            this.reader = parser.getXMLReader();
        }
        catch (ParserConfigurationException ex) {
            System.err.println("...");
        }
        catch (SAXException ex) {
            System.err.println("...");
        }
    }
     
    public Network getNetwork() throws IllegalAccessException {
        if(parsedNetwork == null) {
            throw new IllegalAccessException("File hasn't been parsed yet.");
        }
        
        parsedNetwork.resetNetwork();
        return parsedNetwork;
    }
     
    public LearningSet getLearningSet() throws IllegalAccessException {
        if(parsedLearningSet == null) {
            throw new IllegalAccessException("File hasn't been parsed yet.");
        }
         
        return parsedLearningSet;
    }
    
    public ArrayList<Matrix> getWeightMatrices() throws IllegalAccessException {
    	if(parsedWeightMatrices == null) {
    		throw new IllegalAccessException("File hasn't been parsed yet "
    				+ "or there isn't provided any weight matrices info.");
    	}
    	
    	return parsedWeightMatrices;
    }
    
    /**
     * Parses supplied file. Method expects that network structure and
     * learning set tuples are provided (as file contains all necessary info
     * that will result in creating non-empty objects).
     */
    public void parse() {
        handleParsing();
        getParsedComponents();
    }
    
    /**
     * Parses supplied file. Method does not restrict which components are
     * needed, but user have to specify which ones she wants to retrieve.
     */
    public void parseWithRequests() {
    	handleParsing();
    	getRequestedParsedComponents();
    }
    
    /**
     * Method injects passed network object to handler so that it can act
     * as being parsed by handler instance. Using this method enables injection
     * trigger (not possible to toggle by different means). Limit is raised to
     * ensure that this method will be used only for existent networks.
     */
    public void injectNetworkToHandler(Network network) {
    	this.isNetworkInjectionRaised = true;
    	this.injectedNetwork = network;
    }
    
    private void handleParsing() {        
        handler = new XMLNetworkHandler();
        if(isNetworkInjectionRaised) {
        	((XMLNetworkHandler)handler).injectNetwork(injectedNetwork);
        }
        reader.setContentHandler(handler);
         
        try {
            reader.parse(filePath);
        }
        catch (IOException ex) {
            System.err.println("File cannot be opened.");
        }
        catch (SAXException ex) {
            System.err.println("Parser stopped working due to some error.");
        }
    }
     
    private void getParsedComponents() {
        parsedNetwork = ((XMLNetworkHandler)handler).getParsedNetwork();
        parsedLearningSet = ((XMLNetworkHandler)handler).getParsedLearningSet();
        parsedWeightMatrices = ((XMLNetworkHandler)handler)
        		.getParsedWeightMatrices();
    }
    
    public void addParsingRequest(XMLRequestType requestType) {
    	if(!parseRequests.contains(requestType)) {
    		parseRequests.add(requestType);
    	}
    }
    
    public void removeRequestType(XMLRequestType requestType) {
    	if(parseRequests.contains(requestType)) {
    		parseRequests.remove(requestType);
    	}
    }
    
    private void getRequestedParsedComponents() {
    	if(parseRequests.contains(XMLRequestType.REQ_NETWORK)) {
    		parsedNetwork = ((XMLNetworkHandler)handler).getParsedNetwork();
    	}
    	if(parseRequests.contains(XMLRequestType.REQ_LEARNING)) {
    		parsedLearningSet = ((XMLNetworkHandler)handler).getParsedLearningSet();
    	}
    	if(parseRequests.contains(XMLRequestType.REQ_WEIGHTS)) {
    		parsedWeightMatrices = ((XMLNetworkHandler)handler)
            		.getParsedWeightMatrices();
    	}
    }
}