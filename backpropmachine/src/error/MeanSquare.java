package error;

public class MeanSquare extends ErrorCalculation {
	public double get() 
    {
        double error =  getGlobalError() / getPatternSize();
        return error;
    }
}