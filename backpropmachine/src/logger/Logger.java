package logger;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

import org.dom4j.Document;
import org.dom4j.io.OutputFormat;
import org.dom4j.io.XMLWriter;

import xmlparser.XMLNetworkExtractor;
import feedforward.Network;
import matrixutil.Matrix;

public class Logger {
	private static String directory;
	private static File learningLog;
	private static File networkLog;
	private static File convDumpLog;
	private static int processedEpoch = 0;
	private static boolean isMatrixLoggingEnabled = false;
	private static boolean isArrayLoggingEnabled = false;
	private static ArrayList<LoggerType> loggingStates = new ArrayList<>();
	
	public static void setLoggerDirectory(String logDirectory) {
		directory = logDirectory;
	}
	
	public static void enableAllLogging() {
		for(LoggerType logType : LoggerType.values()) {
			if(!loggingStates.contains(logType)) {
				loggingStates.add(logType);
			}
		}
	}
	
	public static void enableLogging(LoggerType type) {
		if(!loggingStates.contains(type)) {
			loggingStates.add(type);
		}
	}
	
	public static void disableAllLogging() {
		loggingStates.clear();
	}
	
	public static void disableLogging(LoggerType type) {
		if(loggingStates.contains(type)) {
			loggingStates.remove(type);
		}
	}
	
	public static void enableMatrixLogging() {
		isMatrixLoggingEnabled = true;
	}
	
	public static void disableMatrixLogging() {
		isMatrixLoggingEnabled = false;
	}
	
	public static void enableArrayLogging() {
		isArrayLoggingEnabled = true;
	}
	
	public static void disableArrayLogging() {
		isArrayLoggingEnabled = false;
	}
	
	public static void prepareDirectoryForLogger() {
		createDirectoryForLogger();
		createLoggerFiles();
	}
	
	private static void createDirectoryForLogger() {
		convertDateToDirectoryString();
		new File(directory).mkdir();
	}
	
	private static void createLoggerFiles() {
		learningLog = new File(directory + "//learning.txt");
		networkLog = new File(directory + "//network.xml");
		convDumpLog = new File(directory + "//convdumplog.txt");
		try {
			learningLog.createNewFile();
			networkLog.createNewFile();
			convDumpLog.createNewFile();
		} 
		catch (IOException e) {
			System.err.println("CreateLogger: Couldn't create logger files.");
		}
	}
	
	private static void convertDateToDirectoryString() {
		DateFormat dateFormatter = new SimpleDateFormat("yyyy-MM-dd HH-mm-ss");
		String stringDate = dateFormatter.format(new Date());
		directory = directory + "\\" + stringDate;
	}
	
	public static void addIndent() {
		if(isLearningLoggingEnabled()) {
			try(PrintWriter out = new PrintWriter(new FileWriter(learningLog, 
					true))) {
				out.println();
			}
			catch(IOException ex) {
				System.err.println("AddIndent: Cannot write to file " 
					+ "learning.txt");
			}
		}
	}
	
	public static void writeComment(String comment) {
		if(isLearningLoggingEnabled()) {
			try(PrintWriter out = new PrintWriter(new FileWriter(learningLog, 
					true))) {
				out.println("Comment: " + comment);
			}
			catch(IOException ex) {
				System.err.println("WriteComment: Cannot write to " 
					+ "file learning.txt");
			}
		}
	}
	
	public static void writeErrorInfo(double epochError, double targetError) {
		if(isLearningLoggingEnabled()) {
			String line = "[Epoch: " + processedEpoch + ";Target error: "
					+ targetError + ";Actual error: " + epochError + "]";
			try(PrintWriter out = new PrintWriter(new FileWriter(learningLog, 
					true))) {
				out.println(line);
			}
			catch(IOException ex) {
				System.err.println("ErrorInfo: Cannot write to learning.txt");
			}
		}
	}
	
	public static void registerEpochInfo(int epoch) {
		if(isLoggingEnabled()) {
			processedEpoch = epoch;
			if(isLearningLoggingEnabled()) {
				try(PrintWriter out = new PrintWriter(new FileWriter(learningLog, 
						true))) {
					out.println("Proceeding epoch " + epoch + ".");
				}
				catch(IOException ex) {
					System.err.println("RegisterEpoch: Cannot write to file " 
						+ "learning.txt");
				}
			}
		}
	}
	
	public static void writeDoubleArray(double[] array, String comment) {
		if(isLearningLoggingEnabled() && isArrayLoggingEnabled) {
			String info = "[Epoch: " + processedEpoch + ";Vector: " + comment 
				+ "]";
			String values = "";
			for(int i = 0; i < array.length; ++i) {
				values += array[i] + " ";
			}
			
			try(PrintWriter out = new PrintWriter(new FileWriter(learningLog, 
					true))) {
				out.println(info);
				out.println(values);
			}
			catch(IOException ex) {
				System.err.println("DoubleArray: Cannot write to learning.txt");
			}
		}
	}
	
	public static void writeMatrixAll(Matrix matrix, String comment,
			MatrixLoggerType type, int layerNumber) {
		writeMatrixReadable(matrix, comment);
		writeMatrixConvertable(matrix, type, layerNumber);
	}
	
	public static void writeMatrixReadable(Matrix matrix, String comment) {
		if(isLearningLoggingEnabled() && isMatrixLoggingEnabled) {
			try(PrintWriter out = new PrintWriter(new FileWriter(learningLog, 
					true))) {
				out.println("[Epoch: " + processedEpoch + ";Matrix: " 
					+ comment + "]");
				for(int i = 0; i < matrix.getRows(); ++i) {
					String line = "";
					for(int j = 0; j < matrix.getColumns(); ++j) {
						line += matrix.getCell(i, j) + " ";
					}
					out.println(line);
				}
			}
			catch(IOException ex) {
				System.err.println("MatrixReadable: Cannot write to file " 
					+ "learning.txt");
			}
		}
	}
	
	public static void writeMatrixConvertable(Matrix matrix, MatrixLoggerType type,
			int layerNumber) {
		if(isConvDumpLoggingEnabled()) {
			String line = getConveribleMatrixString(matrix, type, layerNumber);
					
			try(PrintWriter out = new PrintWriter(new FileWriter(convDumpLog, 
					true))) {
				out.println(line);
			}
			catch(IOException ex) {
				System.err.println("MatrixConvertable: Cannot write to " 
					+ "convdumplog.txt");
			}
		}
	}
	
	private static String getConveribleMatrixString(Matrix matrix, 
			MatrixLoggerType type, int layerNumber) {
		String line = "[BEGIN_CONV;";
		line += "EPOCH:" + processedEpoch + ";";
		
		if(type.equals(MatrixLoggerType.LAYER_WEIGHTS)) {
			line += "CONV_WEIGHTS;";
		}
		else if(type.equals(MatrixLoggerType.LAYER_GRADIENTS)) {
			line += "CONV_GRADIENTS;";
		}
		else if(type.equals(MatrixLoggerType.LAYER_WEIGHT_CHANGES)) {
			line += "CONV_WEIGHT_CHANGES;";
		}
		
		line += "LAYER:" + layerNumber + ";";
		line += "ROWS:" + matrix.getRows() + ";";
		line += "COLUMNS:" + matrix.getColumns() + ";";
		
		line += "VALUES:";
		for(int i = 0; i < matrix.getRows(); ++i) {
			for(int j = 0; j < matrix.getColumns(); ++j) {
				line += matrix.getCell(i, j);
				if(j != matrix.getColumns() - 1) {
					line += ",";
				}
			}
			if(i != matrix.getRows() - 1) {
				line += "#";
			}
		}
		line += ";END_CONV]";
		
		return line;
	}
	
	public static void writeNetworkToXML(Network network) {
		if(isNetworkLoggingEnabled()) {
			XMLNetworkExtractor extractor = new XMLNetworkExtractor(network);
			Document document = extractor.getXMLDocument();
			OutputFormat format = OutputFormat.createPrettyPrint();
			
			try(PrintWriter out = new PrintWriter(new FileWriter(networkLog))) {
				XMLWriter writer = new XMLWriter(out, format);
				writer.write(document);
			}
			catch(IOException ex) {
				System.err.println("NetworkToXML: Cannot write to network.txt");
			}
		}
	}
	
	private static boolean isLoggingEnabled() {
		return !loggingStates.isEmpty();
	}
	
	private static boolean isLearningLoggingEnabled() {
		return loggingStates.contains(LoggerType.LOG_LEARNING);
	}
	
	private static boolean isNetworkLoggingEnabled() {
		return loggingStates.contains(LoggerType.LOG_NETWORK);
	}
	
	private static boolean isConvDumpLoggingEnabled() {
		return loggingStates.contains(LoggerType.LOG_CONV);
	}
}