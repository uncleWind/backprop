package activation;

public interface ActivationFunction {
	double getValue(double x);
    double getDerivative(double x);
}