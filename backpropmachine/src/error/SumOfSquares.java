package error;

public class SumOfSquares extends ErrorCalculation {
	@Override
    public double get() 
    {
        double error = getGlobalError() / 2;
        return error;
    }
}