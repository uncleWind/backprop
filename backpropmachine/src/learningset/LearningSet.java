package learningset;
 
import feedforward.Network;
import validation.Validator;
 
import java.util.ArrayList;

public class LearningSet {
    private ArrayList<double[]> inputs;
    private ArrayList<double[]> outputs;
    private Network network;
     
    public LearningSet(Network network) {
        this.inputs = new ArrayList<>();
        this.outputs = new ArrayList<>();
        this.network = network;
    }
     
    public void putNewLearningTuple(double[] newInput, double[] newOutput) {
        putNewInputs(newInput);
        putNewOutputs(newOutput);
    }
     
    private void putNewInputs(double[] newInput) {
        Validator.validateLearningInputSampleSize(newInput, network);
        inputs.add(newInput.clone());
    }
     
    private void putNewOutputs(double[] newOutput) {
        Validator.validateLearningOutputSampleSize(newOutput, network);
        outputs.add(newOutput.clone());
    }
     
    public ArrayList<double[]> getLearningInputsSet() {
        return inputs;
    }
     
    public ArrayList<double[]> getLearningOutputsSet() {
        return outputs;
    }
}