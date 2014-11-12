package activation;

public class Sigmoid implements ActivationFunction {
	@Override
    public double getValue(double x) 
    {
        double result = 1 / (1 + Math.exp(-x));
        return result;
    }
     
    @Override
    public double getDerivative(double x) 
    {
        double result = getValue(x) * (1 - getValue(x));
        return result;
    }
}