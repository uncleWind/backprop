package matrixutil;

import validation.*;

public class MatrixMath {
	public static Matrix add(Matrix firstAddend, Matrix secondAddend) 
    { 
        Validator.validateSizeEqualityOf(firstAddend.toPackedArray(), 
            secondAddend.toPackedArray()); 
          
        int rows = firstAddend.getRows(); 
        int columns = firstAddend.getColumns(); 
          
        double[][] resultPrototype = new double[rows][columns]; 
          
        for(int i = 0; i < rows; ++i) 
        { 
            for(int j = 0; j < columns; ++j) 
            { 
                resultPrototype[i][j] = firstAddend.getCell(i, j) 
                    + secondAddend.getCell(i, j); 
            } 
        } 
          
        return new Matrix(resultPrototype); 
    } 
      
    public static Matrix add(Matrix addedMatrix, double factor) 
    { 
        Validator.validateValue(factor); 
          
        int rows = addedMatrix.getRows(); 
        int columns = addedMatrix.getColumns(); 
          
        double resultPrototype[][] = new double[rows][columns]; 
          
        for(int i = 0; i < rows; ++i) 
        { 
            for(int j = 0; j < columns; ++j) 
            { 
                resultPrototype[i][j] = addedMatrix.getCell(i, j) + factor; 
            }     
        } 
          
        return new Matrix(resultPrototype); 
    } 
      
    public static Matrix subtract(Matrix minuend, Matrix subtrahend) 
    { 
        Validator.validateSizeEqualityOf(minuend.toPackedArray(), 
            subtrahend.toPackedArray()); 
          
        int rows = minuend.getRows(); 
        int columns = minuend.getColumns(); 
          
        double[][] resultPrototype = new double[rows][columns]; 
          
        for(int i = 0; i < rows; ++i) 
        { 
            for(int j = 0; j < columns; ++j) 
            { 
                resultPrototype[i][j] = minuend.getCell(i, j) 
                    - subtrahend.getCell(i, j); 
            } 
        } 
          
        return new Matrix(resultPrototype); 
    } 
      
    public static Matrix multiply(Matrix multiplier, Matrix multiplicand) 
    { 
        Validator.validateMatrixMultiplicationSizeCondition(multiplier,  
            multiplicand); 
          
        int rows = multiplier.getRows(); 
        int columns = multiplicand.getColumns(); 
          
        double[][] resultPrototype = new double[rows][columns]; 
          
        for (int i = 0; i < rows; ++i) 
        { 
            for (int j = 0; j < columns; ++j) 
            { 
                for (int k = 0; k < multiplicand.getRows(); ++k) 
                { 
                    resultPrototype[i][j] += multiplier.getCell(i, k)  
                        * multiplicand.getCell(k, j); 
                } 
            } 
        } 
          
        return new Matrix(resultPrototype); 
    } 
      
    public static Matrix multiply(Matrix multipliedMatrix, double factor) 
    { 
        Validator.validateValue(factor); 
          
        int rows = multipliedMatrix.getRows(); 
        int columns = multipliedMatrix.getColumns(); 
          
        double[][] resultPrototype = new double[rows][columns]; 
          
        for(int i = 0; i < rows; ++i) 
        { 
            for(int j = 0; j < columns; ++j) 
            { 
                resultPrototype[i][j] = multipliedMatrix.getCell(i, j) * factor; 
            } 
        } 
          
        return new Matrix(resultPrototype); 
    } 
      
    public static Matrix divide(Matrix dividedMatrix, double divisor) 
    { 
        Validator.validateValue(divisor); 
          
        int rows = dividedMatrix.getRows(); 
        int columns = dividedMatrix.getColumns(); 
          
        double[][] resultPrototype = new double[rows][columns]; 
          
        for(int i = 0; i < rows; ++i) 
        { 
            for(int j = 0; j < columns; ++j) 
            { 
                resultPrototype[i][j] = dividedMatrix.getCell(i, j) / divisor; 
            } 
        } 
          
        return new Matrix(resultPrototype); 
    } 
      
    public static double dotProduct(Matrix firstMatrix, Matrix secondMatrix) 
    { 
        Validator.validateDotProductConditions(firstMatrix, secondMatrix); 
          
        double[] firstVector = firstMatrix.toPackedArray(); 
        double[] secondVector = secondMatrix.toPackedArray(); 
        int vectorSize = firstVector.length; 
          
        double result = 0.0f; 
          
        for(int i = 0; i < vectorSize; ++i) 
        { 
            result += firstVector[i] * secondVector[i]; 
        } 
          
        return result; 
    }         
      
    public static void copy(Matrix source, Matrix target) 
    { 
        Validator.validateRowsAndColumnsEquality(target, source); 
          
        int rows = target.getRows(); 
        int columns = target.getColumns(); 
          
        for(int i = 0; i < rows; ++i) 
        { 
            for(int j = 0; j < columns; ++j) 
            { 
                double copiedValue = source.getCell(i, j); 
                target.setCell(i, j, copiedValue); 
            } 
        } 
    } 
      
    public static Matrix deleteRow(Matrix source, int rowIndex) 
    { 
        Validator.validateRowCoordinateWith(source, rowIndex); 
          
        int rows = source.getRows(); 
        int columns = source.getColumns(); 
          
        double[][] resultPrototype = new double[rows - 1][columns]; 
          
        for(int i = 0; i < rows; ++i) 
        { 
            if(i != rowIndex) 
            { 
                int actualRow = i; 
                if(actualRow > rowIndex) 
                { 
                    --actualRow; 
                } 
                  
                for(int j = 0; j < columns; ++j) 
                { 
                    resultPrototype[actualRow][j] = source.getCell(i, j); 
                } 
            } 
        } 
          
        return new Matrix(resultPrototype); 
    } 
      
    public static Matrix deleteColumn(Matrix source, int columnIndex) 
    { 
        Validator.validateColumnCoordinateWith(source, columnIndex); 
          
        int rows = source.getRows(); 
        int columns = source.getColumns(); 
          
        double[][] resultPrototype = new double[rows][columns - 1]; 
          
        for(int i = 0; i < rows; ++i) 
        { 
            for(int j = 0; j < columns; ++j) 
            { 
                if(j != columnIndex) 
                { 
                    int actualColumn = j; 
                    if(actualColumn > columnIndex) 
                    { 
                        --actualColumn; 
                    } 
                      
                    resultPrototype[i][actualColumn] = source.getCell(i, j); 
                } 
            } 
        } 
          
        return new Matrix(resultPrototype); 
    } 
    
    /* TODO: check solution to the rounding problem by using BigDecimal
     * class or other means of comparison
     */
    public static boolean compare(Matrix source, Matrix compared) {
    	Validator.validateSizeEqualityOf(source.toPackedArray(),
    			compared.toPackedArray());
    	
    	int rows = source.getRows();
    	int columns = source.getColumns();
    	
    	for(int i = 0; i < rows; ++i) {
    		for(int j = 0; j < columns; ++j) {
    			double sourceValue = source.getCell(i, j);
    			double comparedValue = compared.getCell(i, j);
    			
    			if(Double.compare(sourceValue, comparedValue) != 0) {
    				return false;
    			}
    		}
    	}
    	
    	return true;
    }
    
    public static Matrix transpose(Matrix source)  
    { 
        int rows = source.getColumns(); 
        int columns = source.getRows(); 
          
        double[][] resultPrototype = new double[rows][columns]; 
          
        for(int i = 0; i < rows; ++i) { 
            for(int j = 0; j < columns; ++j) 
            { 
                resultPrototype[i][j] = source.getCell(j, i); 
            } 
        } 
          
        return new Matrix(resultPrototype); 
    } 
      
    public static Matrix createIdentity(int size) 
    { 
        double[][] resultPrototype = new double[size][size]; 
          
        Validator.validateSize(size); 
          
        for(int i = 0; i < size; ++i) 
        { 
            for(int j = 0; j < size; ++j) 
            { 
                if(i == j) 
                { 
                    resultPrototype[i][j] = 1.0f; 
                } 
                else
                { 
                    resultPrototype[i][j] = 0.0f; 
                } 
            } 
        } 
          
        return new Matrix(resultPrototype); 
    } 
}
