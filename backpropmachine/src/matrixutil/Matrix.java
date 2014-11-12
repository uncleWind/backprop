package matrixutil;

import validation.*; 
import java.util.*; 

public class Matrix implements Cloneable { 
      
    private double[][] matrix; 
      
    public double getCell(int row, int column) { 
        Validator.validateCoordinatesWith(this, row, column); 
        return matrix[row][column]; 
    } 
      
    public void setCell(int row, int column, double value) { 
        Validator.validateValue(value); 
        Validator.validateCoordinatesWith(this, row, column); 
        matrix[row][column] = value; 
    } 
      
    public void addValueToCell(double value, int row, int column) { 
        Validator.validateValue(value); 
        Validator.validateCoordinatesWith(this, row, column); 
        matrix[row][column] += value; 
    } 
      
    public int getRows() { 
        return matrix.length; 
    } 
      
    public int getColumns() { 
        return matrix[0].length; 
    } 
      
    public int getCellsQuantity() { 
        return getRows() * getColumns(); 
    } 
      
    public Matrix(int rows, int columns) { 
        matrix = new double[rows][columns]; 
    } 
      
    public Matrix(boolean[][] valuesSet) { 
        int rows = valuesSet.length; 
        int cols = valuesSet[0].length; 
          
        matrix = new double[rows][cols]; 
          
        for(int i = 0; i < rows; ++i) { 
            for(int j = 0; j < cols; ++j) { 
                if(valuesSet[i][j] == true) { 
                    matrix[i][j] = 1.0f; 
                } 
                else { 
                    matrix[i][j] = 0.0f; 
                } 
            } 
        } 
    } 
      
    public Matrix(double[][] valuesSet) { 
        int rows = valuesSet.length; 
        int cols = valuesSet[0].length; 
          
        matrix = new double[rows][cols]; 
          
        for(int i = 0; i < rows; ++i) { 
            for(int j = 0; j < cols; ++j) { 
                matrix[i][j] = new Double(valuesSet[i][j]); 
            } 
        } 
    } 
    
    public Matrix(Matrix sourceMatrix) {
    	int rows = sourceMatrix.getRows();
    	int cols = sourceMatrix.getColumns();
    	
    	matrix = new double[rows][cols];
    	
    	for(int i = 0; i < rows; ++i) {
    		for(int j = 0; j < cols; ++j) {
    			matrix[i][j] = new Double(sourceMatrix.getCell(i, j));
    		}
    	}
    }
    
    public Matrix getColumnAsMatrix(int index) { 
        Validator.validateColumnCoordinateWith(this, index); 
          
        double[][] matrixTemplate = new double[getRows()][1]; 
        for(int i = 0; i < getRows(); ++i) { 
            matrixTemplate[i][0] = new Double(matrix[i][index]); 
        } 
          
        Matrix newMatrix = new Matrix(matrixTemplate); 
        return newMatrix; 
    } 
      
    public Matrix getRowAsMatrix(int index) { 
        Validator.validateRowCoordinateWith(this, index); 
          
        double[][] matrixTemplate = new double[1][getColumns()]; 
        for(int i = 0; i < getColumns(); ++i) { 
            matrixTemplate[0][i] = new Double(matrix[index][i]); 
        } 
          
        Matrix newMatrix = new Matrix(matrixTemplate); 
        return newMatrix; 
    } 
      
    public Matrix createColumnMatrix(double[] valuesSet) { 
        Validator.validateValuesOf(valuesSet); 
           
        int rows = valuesSet.length; 
          
        double[][] matrixTemplate = new double[rows][1]; 
        for(int i = 0; i < rows; ++i) { 
            matrixTemplate[i][0] = new Double(valuesSet[i]); 
        } 
          
        Matrix newMatrix = new Matrix(matrixTemplate); 
        return newMatrix; 
    } 
      
    public Matrix createRowMatrix(double[] valuesSet) { 
        Validator.validateValuesOf(valuesSet); 
          
        int cols = valuesSet.length; 
          
        double[][] matrixTemplate = new double[1][cols]; 
        for(int i = 0; i < cols; ++i) { 
            matrixTemplate[0][i] = new Double(valuesSet[i]); 
        } 
          
        Matrix newMatrix = new Matrix(matrixTemplate); 
        return newMatrix; 
    } 
      
    @Override
    public Matrix clone() throws CloneNotSupportedException { 
        return (Matrix) super.clone(); 
    } 
      
    public void reset() { 
        for(int i = 0; i < getRows(); ++i) { 
            for(int j = 0; j < getColumns(); ++j) { 
                matrix[i][j] = 0.0f; 
            } 
        } 
    } 
      
    public boolean isZeroed() { 
        for(int i = 0; i < getRows(); ++i) { 
            for(int j = 0; j < getColumns(); ++j) { 
                if(matrix[i][j] != 0.0f) { 
                    return false; 
                } 
            } 
        } 
          
        return true; 
    } 
      
    public boolean isVector() { 
        boolean result = (getRows() == 1 || getColumns() == 1); 
        return result; 
    } 
      
    public boolean isNotAVector() { 
        return !isVector(); 
    } 
      
    public void randomizeFromNegativeOneToOne() { 
        Random randomFactory = new Random(); 
          
        for(int i = 0; i < getRows(); ++i) { 
            for(int j = 0; j < getColumns(); ++j) { 
                double newRandom = -randomFactory.nextDouble(); 
                newRandom += randomFactory.nextDouble() * 2; 
                  
                matrix[i][j] = newRandom; 
            }     
        } 
    } 
      
    public void randomizeFromZeroToOne() { 
        Random randomFactory = new Random(); 
          
        for(int i = 0; i < getRows(); ++i) { 
            for(int j = 0; j < getColumns(); ++j) { 
                double newRandom = randomFactory.nextDouble(); 
                  
                matrix[i][j] = newRandom; 
            } 
        } 
    } 
      
    public void randomizeBetween(double lowerBound, double higherBound) { 
        Random randomFactory = new Random(); 
          
        for(int i = 0; i < getRows(); ++i) { 
            for(int j = 0; j < getColumns(); ++j) { 
                double newRandom = lowerBound + higherBound  
                        * randomFactory.nextDouble(); 
                  
                matrix[i][j] = newRandom; 
            } 
        } 
    } 
      
    public double allCellsSum() { 
        double result = 0.0f; 
          
        for(int i = 0; i < getRows(); ++i) { 
            for(int j = 0; j < getColumns(); ++j) { 
                result += matrix[i][j]; 
            } 
        } 
          
        return result; 
    } 
      
    public double[] toPackedArray() { 
        double[] packedMatrix = new double[getCellsQuantity()]; 
          
        for(int i = 0; i < getRows(); ++i) { 
            for(int j = 0; j < getColumns(); ++j) 
            { 
                //int index = (i * getRows()) + j; 
                int index = (i * getColumns()) + j; 
                  
                packedMatrix[index] = matrix[i][j]; 
            } 
        } 
          
        return packedMatrix; 
    } 
      
    public void fromPackedArray(double[] packed) { 
        Validator.validatePackedArrayCellsQuantityEqualityWith(this, packed); 
          
        int index = 0; 
          
        for(int i = 0; i < getRows(); ++i) { 
            for(int j = 0; j < getColumns(); ++j) { 
                matrix[i][j] = packed[index]; 
            } 
        } 
    } 
}