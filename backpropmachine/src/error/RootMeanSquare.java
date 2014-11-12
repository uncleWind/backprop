package error;

public class RootMeanSquare extends ErrorCalculation  {
	@Override
     public double get() 
    {
        double error = Math.sqrt(getGlobalError() / getPatternSize());
        return error;
    }
}