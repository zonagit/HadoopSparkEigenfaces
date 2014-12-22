package eigenfaces;

import java.io.File;
import java.util.Collection;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public class GenerateCovarianceMatrix 
{
	private static double[][] convertImagesToMatrix(
			Collection<String> imageFileNames,
			int width, int height) throws Exception 
	{
		int columnIndex = 0;
		double[][] pixelMatrix = new double[width * height][imageFileNames.size()];
		for(String fileName: imageFileNames) 
		{
			System.out.println("Reading file " + fileName);
			double[] pixels = Helper.readImagePixels(fileName, width, height);
			for(int i = 0 ; i < pixels.length ; i++) 
			{
				pixelMatrix[i][columnIndex] = pixels[i];
			}
			columnIndex++;
		}
		return pixelMatrix;
	}

	private static double[] computeMeanColumn(double[][] pixelMatrix) 
	{
		int pixelCount = pixelMatrix.length;
		double[] meanColumn = new double[pixelCount];
		int columnCount = pixelMatrix[0].length;
		for(int i = 0 ; i < pixelCount ; i++) 
		{
			int sum = 0;
			for(int j = 0 ; j < columnCount ; j++) 
			{
				sum += pixelMatrix[i][j];
			}
			meanColumn[i] = sum / columnCount;
		}
		return meanColumn;
	}

	private static double[][] computeCovarianceMatrix(double[][] diffMatrixPixels) 
	{
		int rowCount = diffMatrixPixels.length;
		int columnCount = diffMatrixPixels[0].length;
		double[][] covarianceMatrix = new double[columnCount][columnCount];
		for(int i = 0 ; i < columnCount ; i++) 
		{
			for(int j = 0 ; j < columnCount ; j++) 
			{
				int sum = 0;
				for(int k = 0 ; k < rowCount ; k++) 
				{
					sum += diffMatrixPixels[k][i] * diffMatrixPixels[k][j];
				}	
				covarianceMatrix[i][j] = sum;
			}
		}
		return covarianceMatrix;
	}
	
	public static double[][] getDiffMatrix(String[] args) throws Exception
	{
		int width = Integer.parseInt(args[0]);
		int height = Integer.parseInt(args[1]);
		String imageDirectory = args[2];
		List<String> imageFileNames = Helper.listImageFileNames(imageDirectory);
		System.out.println("Reading " + imageFileNames.size() + " images...");
		double[][] pixelMatrix = convertImagesToMatrix(imageFileNames, width, height);
		double[] meanColumn = computeMeanColumn(pixelMatrix);
		return Helper.computeDifferenceMatrixPixels(pixelMatrix, meanColumn);
	
	}
	
	public static double[][] getCovarianceMatrix(String[] args) throws Exception
	{
		int width = Integer.parseInt(args[0]);
		int height = Integer.parseInt(args[1]);
		String imageDirectory = args[2];
		List<String> imageFileNames = Helper.listImageFileNames(imageDirectory);
		System.out.println("Reading " + imageFileNames.size() + " images...");
		double[][] pixelMatrix = convertImagesToMatrix(imageFileNames, width, height);
		double[] meanColumn = computeMeanColumn(pixelMatrix);
		double[][] diffMatrixPixels = Helper.computeDifferenceMatrixPixels(pixelMatrix, meanColumn);
	//	Helper.writeMatrixSequenceFile(outputDirectory + "/diffmatrix.seq", diffMatrixPixels);
		double[][] covarianceMatrix = computeCovarianceMatrix(diffMatrixPixels);
		return covarianceMatrix;
	}
	
	
	
	public static void main(String args[]) throws Exception 
	{
		if (args.length != 4) 	
		{
			System.out.println("Arguments: width height trainingDirectory outputDirectory");
			System.exit(1);
		}
		int width = Integer.parseInt(args[0]);
		int height = Integer.parseInt(args[1]);
		String imageDirectory = args[2];
		String outputDirectory = args[3];
		File outputDirectoryFile = new File(outputDirectory);
		if (!outputDirectoryFile.exists()) 
		{
			outputDirectoryFile.mkdir();
		}
		
		List<String> imageFileNames = Helper.listImageFileNames(imageDirectory);
		System.out.println("Reading " + imageFileNames.size() + " images...");
		double[][] pixelMatrix = convertImagesToMatrix(imageFileNames, width, height);
		double[] meanColumn = computeMeanColumn(pixelMatrix);
		System.out.println("Writing mean image to " + imageDirectory + "/mean-image.gif");
		Helper.writeImage(imageDirectory + "/mean-image.gif", meanColumn, width, height);
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(new Path(imageDirectory+"/mean-image.gif").toUri(),conf);
		fs.copyFromLocalFile(new Path(imageDirectory + "/mean-image.gif"), new Path(outputDirectory + "/mean-image.gif"));
		double[][] diffMatrixPixels = Helper.computeDifferenceMatrixPixels(pixelMatrix, meanColumn);
		System.out.println("Writing difference matrix to " + outputDirectory + "/diffmatrix.seq");
		Helper.writeMatrixSequenceFile(outputDirectory + "/diffmatrix.seq", diffMatrixPixels);
		double[][] covarianceMatrix = computeCovarianceMatrix(diffMatrixPixels);
		System.out.println("Writing covariance matrix to " + outputDirectory + "/covariance.seq");
		Helper.writeMatrixSequenceFile(outputDirectory + "/covariance.seq", covarianceMatrix);
	}
}