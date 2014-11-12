package xmlparser;

import java.util.ArrayList;

import error.*;
import feedforward.*;
import learningset.*;
import matrixutil.Matrix;
import training.*;
import weightinit.*;
import activation.*;
import validation.*;

import org.xml.sax.*;
import org.xml.sax.helpers.*;

public class XMLNetworkHandler extends DefaultHandler {
    private Network parsedNetwork;
    private LearningSet parsedLearningSet;
    private ArrayList<Matrix> parsedWeightMatrices = new ArrayList<>();
    
    private String valueBuffer;
    private String startElement;
    
    private boolean charEditingPermission = false;
    private boolean networkResolvingBiasCondition = false;
    
    private int addedLayersCount = 0;
                
    private String[] startElementOverrideTriggers =
    {
        "training", "rate", "momentum", "errormethod", "targeterror", "bias",
        "neurons", "randomizer", "activation",
        "rows", "columns", "row",
        "randminbound", "randmaxbound",
        "position", "multiplier",
        "firstinput", "firstoutput", "input", "output",
        //Triggers below are added to ensure proper decoding sequence break.
        "inputs", "outputs", "values"
    };

    final private String[] startElementSeqStarters = {
        "training", "neurons", "firstinput", "rows", "position"
    };

    final private String[] startElementSeqAppenders = {
        "rate", "momentum", "randomizer", "activation", "input", 
        "output", "targeterror", "columns", "multiplier"
    };

    final private String[] startElementSeqDivider = {
        "randminbound", "randmaxbound", "firstoutput", "errormethod", "bias",
        "row"
    };
    
    public XMLNetworkHandler() {
        valueBuffer = "";
        startElement = "";
    }
    
    public Network getParsedNetwork() {
        Validator.validateParsedNetworkCompletion(parsedNetwork);
        
        return parsedNetwork;
    }
    
    public LearningSet getParsedLearningSet() {
        Validator.validateParsedLearningSetCompletion(parsedLearningSet);
        
        return parsedLearningSet;
    }
    
    public ArrayList<Matrix> getParsedWeightMatrices() {
    	return parsedWeightMatrices;
    }
    
    public void injectNetwork(Network network) {
    	this.parsedNetwork = network;
    }
    
    @Override
    public void startElement(String namespaceURI, String lname,
            String qname, Attributes attrs) throws SAXException {
        
        charEditingPermission = true;
        
        System.out.println("Entered startElement for: " + qname);
        for(String trigger : startElementOverrideTriggers) {
            if(qname.equalsIgnoreCase(trigger)) {
                startElement = qname;
                System.out.println("Set start element to: " + startElement);
            }
        }

        if(qname.equalsIgnoreCase("learningset")) {
            parsedLearningSet = new LearningSet(parsedNetwork);
        }
    }

    @Override
    public void characters(char[] data, int start, int length)
            throws SAXException {
        System.out.println("Entered characters for data of length: " + length);
        
        if(charEditingPermission) {
            String caughtString = "";
            for(int i  = start; i < start + length; ++i) {
                caughtString = caughtString + data[i];
            }
            caughtString = caughtString.replaceAll("\t", "")
                                       .replaceAll("\r", "")
                                       .replaceAll("\n", "");
            String inserted = caughtString;
            System.out.println("Inserted value: " + inserted);
            System.out.println(">Implying suggested se: " + startElement);
        
            for(String trigger : startElementSeqStarters) {
                if(startElement.equals(trigger)) {
                	System.out.println(">Implying seqStarters.");
                    if(!valueBuffer.isEmpty()) {
                        System.out.println("Clearing on SeqStarter.");
                        valueBuffer = "";
                    }
                    valueBuffer += inserted;
                }
            }
            for(String trigger : startElementSeqAppenders) {
                if(startElement.equals(trigger)) {
                	System.out.println(">Implying seqAppenders.");
                    valueBuffer += ";" + inserted;
                }
            }
            for(String trigger : startElementSeqDivider) {
                if(startElement.equals(trigger)) {
                	System.out.println(">Implying seqDividers.");
                    if(!inserted.equalsIgnoreCase("none")) {
                        valueBuffer += ":" + inserted;
                    }
                }
            }
            System.out.println("valueBuffer set to: " + valueBuffer);
        }
        else {
            System.out.println("Entered from endElement, aborted.");
        }
    }

    @Override
    public void endElement(String namespaceURI, String lname,
            String qname) throws SAXException {
        
        charEditingPermission = false;
        
        System.out.println("Entered endElement of " + qname + ".");
        if(qname.equalsIgnoreCase("params")) {
            System.out.println("Resolving network creation.");
            resolveNetworkCreation();
        }
        if(qname.equalsIgnoreCase("layer")) {
            System.out.println("Resolving layer creation.");
            ++addedLayersCount;
            resolveLayerAddition();
        }
        if(qname.equalsIgnoreCase("tuple")) {
            System.out.println("Resolving tuple creation.");
            resolveLearningTupleAddition();
        }
        if(qname.equalsIgnoreCase("weightmatrix")) {
        	System.out.println("Storing weight matrix.");
        	storeWeightMatrix();
        }
        if(qname.equalsIgnoreCase("neuron")) {
        	System.out.println("Adjusting neuron parameters.");
        	adjustNeuronParameters();
        }
    }

    private void resolveNetworkCreation() {
        String[] networkParams = valueBuffer.split(":");
        String[] trainerParams = networkParams[0].split(";");
        String[] errorParams = networkParams[1].split(";");
        String biasCondition = networkParams[2];
        
        String trainerName = trainerParams[0];
        double learningRate = Double.parseDouble(trainerParams[1]);
        double momentum = Double.parseDouble(trainerParams[2]);
        
        String errorMethodName = errorParams[0];
        double targetError = Double.parseDouble(errorParams[1]);
        
        if(biasCondition.equalsIgnoreCase("true")) {
        	networkResolvingBiasCondition = true;
        }
        System.out.println("networkResolvingBiasCondition value: "
        		+ networkResolvingBiasCondition);
        
        System.out.println("Clearing on NetworkCreation.");
        valueBuffer = "";

        createNetwork(trainerName, learningRate, momentum,
        		errorMethodName, targetError); 
    }

    private void createNetwork(String trainerName, double learningRate,
    		double momentum, String errorMethodName, double targetError) {
        System.out.println("Trainer name for network: " + trainerName);
        Trainer trainer = createTrainer(trainerName);
        ErrorCalculation errorMethod = createErrorMethod(errorMethodName);
        
        parsedNetwork = new Network(trainer, learningRate, momentum, 
        		errorMethod, targetError);
        System.out.println("Created network with " + trainerName
        		+ " trainer and " + errorMethodName + " error calculation.");
    }
    
    private Trainer createTrainer(String trainerName) {
    	if(trainerName.equals("Batch")) {
    		return new Batch();
    	}
    	else if(trainerName.equals("Online")) {
    		return new Online();
    	}
    	
    	throw new IllegalArgumentException("'training' record have "
    			+ "invalid value of " + trainerName);
    }
    
    private ErrorCalculation createErrorMethod(String errorMethodName) {
    	if(errorMethodName.equals("MeanSquare")) {
    		return new MeanSquare();
    	}
    	else if(errorMethodName.equals("RootMeanSquare")) {
    		return new RootMeanSquare();
    	}
    	else if(errorMethodName.equals("SumOfSquares")) {
    		return new SumOfSquares();
    	}
    	
    	throw new IllegalArgumentException("'errormethod' record have "
    			+ "invalid value of " + errorMethodName);
    }
    
    private void resolveLayerAddition() {
        String[] layerParams = valueBuffer.split(";");
        int layerSize = Integer.parseInt(layerParams[0]);
        String setterName = layerParams[1];
        String activationName = layerParams[2];
        System.out.println("Clearing on LayerCreation.");
        valueBuffer = "";

        ActivationFunction activation = 
                createActivationFunction(activationName);
        WeightSetter setter = createWeightSetter(setterName);
        
        /* Because the second layer will always be the output layer,
         * we have to ensure that this layer won't have a bias neuron.
         */
        boolean asBias = networkResolvingBiasCondition;
        if(addedLayersCount == 2) {
        	asBias = false;
        }
        
        parsedNetwork.addLayer(layerSize, setter, activation, asBias);
    }

    private WeightSetter createWeightSetter(String setterName) {
        if(setterName.equalsIgnoreCase("nguyenwidrow")) {
            System.out.println("Created NguyenWidrow WD method.");
            return new NguyenWidrow();
        }
        else if(setterName.contains("RangedRandom")) {
            String[] setterParams = setterName.split(":");
            double minBound = Double.parseDouble(setterParams[1]);
            double maxBound = Double.parseDouble(setterParams[2]);

            RangedRandom setter = new RangedRandom();
            ((RangedRandom)setter).setBounds(minBound, maxBound);
            System.out.println("Created RangedRandom WD method w/ bounds: "
                               + minBound + " " + maxBound);
            return setter;
        }

        throw new IllegalArgumentException("One of 'randomizer' "
            + "records have invalid value of " + setterName);
    }

    private ActivationFunction createActivationFunction(
            String activationName) {
        if(activationName.equalsIgnoreCase("hyperbolictangent")) {
            return new HyperbolicTangent();
        }
        else if(activationName.equalsIgnoreCase("linear")) {
            return new Linear();
        }
        else if(activationName.equalsIgnoreCase("sigmoid")) {
            return new Sigmoid();
        }

        throw new IllegalArgumentException("One of 'activation' "
            + "records have invalid value of " + activationName);
    }

    private void resolveLearningTupleAddition() {
        String[] tupleParams = valueBuffer.split(":");
        String[] tupleInputs = tupleParams[0].split(";");
        String[] tupleOutputs = tupleParams[1].split(";");
        
        System.out.println("Inputs length: " + tupleInputs.length);
        System.out.println("Outputs length: " + tupleOutputs.length);

        double[] inputs = getLearningTupleSubset(tupleInputs);
        double[] outputs = getLearningTupleSubset(tupleOutputs);
        
        //TESTING PURPOSES
        System.out.print("New tuple inputs (" + inputs.length + "): ");
        for(double input : inputs) {
            System.out.print(input + " ");
        }
        System.out.println();
        System.out.print("New tuple outputs (" + outputs.length + "): ");
        for(double output : outputs) {
            System.out.print(output + " ");
        }
        System.out.println();

        parsedLearningSet.putNewLearningTuple(inputs, outputs);
    }

    private double[] getLearningTupleSubset(String[] tupleValues) {
        int records = tupleValues.length;
        double[] values = new double[records];
        for(int i = 0; i < records; ++i) {
            values[i] = Double.parseDouble(tupleValues[i]);
        }
        return values;
    }
    
    private void storeWeightMatrix() {
    	String[] matrixParams = valueBuffer.split(":", 2);
    	String[] matrixSize = matrixParams[0].split(";");
    	String[] matrixRows = matrixParams[1].split(":");
    	
    	int rows = Integer.parseInt(matrixSize[0]);
    	int columns = Integer.parseInt(matrixSize[1]);
    	
    	Matrix weightMatrix = new Matrix(rows, columns);
    	for(int i = 0; i < rows; ++i) {
    		double[] rowValues = getWeightMatrixRowValues(matrixRows[i]);
    		for(int j = 0; j < columns; ++j) {
    			double cellValue = rowValues[j];
    			weightMatrix.setCell(i, j, cellValue);
    		}
    	}
    	
    	parsedWeightMatrices.add(weightMatrix);
    }
    
    private double[] getWeightMatrixRowValues(String values) {
    	String[] extractedValues = values.split(";");
    	double[] result = new double[extractedValues.length];
    	for(int i = 0; i < result.length; ++i) {
    		result[i] = Double.parseDouble(extractedValues[i]);
    	}
    	
    	return result;
    }
    
    private void adjustNeuronParameters() {
    	String[] neuronParams = valueBuffer.split(";");
    	int neuronIndex = Integer.parseInt(neuronParams[0]) - 1;
    	double multiplier = Double.parseDouble(neuronParams[1]);
    	
    	if(neuronIndex < parsedNetwork.size()) {
    		adjustNeuronMultiplier(neuronIndex, multiplier);
    	}
    }
    
    private void adjustNeuronMultiplier(int index, double multiplier) {
    	Layer outputLayer = parsedNetwork.getOutputLayer();
    	outputLayer.setNeuronMultiplierValueAt(index, multiplier);
    }
}