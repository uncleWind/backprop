package activation;

public class Linear implements ActivationFunction {
	@Override
    public double getValue(double x) 
    {
        return x;
    }
     
    @Override
    public double getDerivative(double x) 
    {
        throw new IllegalArgumentException("Derivative of linear activation "
                + "function doesn't exist.");
    }
}