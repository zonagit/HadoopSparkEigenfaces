package eigenfaces;

import java.util.List;

public class ComputeDistance 
{
	public static Object[] findClosestImage(double[] weights, double[][] weightMatrix) 
	{
		int imageCount = weightMatrix.length;
		int eigenFaceCount = weightMatrix[0].length;
		int closestImageIndex = -1;
		double minWeightSquareDistance = Double.MAX_VALUE;
		for(int i = 0 ; i < imageCount ; i++) 
		{
			double distance = 0;
			for(int j = 0 ; j < eigenFaceCount ; j++) 
			{
				distance += (weightMatrix[i][j] - weights[j]) * (weightMatrix[i][j] - weights[j]);
			}
			if (distance < minWeightSquareDistance) 
			{
				minWeightSquareDistance = distance;
				closestImageIndex = i;
			}
		}
		return new Object[]{closestImageIndex, Math.sqrt(minWeightSquareDistance / eigenFaceCount)};
	}

	public static void main(String args[]) throws Exception 
	{
		if (args.length != 8) 
		{
			System.out.println("Arguments: eigenFacesFileName meanImageFileName weightSeqFilename width height trainingDirectory testImageDirectory outputDirectory");
			System.exit(1);
		}
		String eigenFacesFilename = args[0];
		String meanImageFilename = args[1];
		String weightSeqFileName = args[2];
		int width = Integer.parseInt(args[3]);
		int height = Integer.parseInt(args[4]);
		String trainImageDirectory = args[5];
		String testImageDirectory = args[6];
		String outputDirectory = args[7];
		double[] meanPixels = Helper.readImagePixels(meanImageFilename, width, height);
		double[][] eigenFaces = Helper.readMatrixSequenceFile(eigenFacesFilename);
		double[][] weightMatrix = Helper.readMatrixSequenceFile(weightSeqFileName);
		List<String> testImageFileNames = Helper.listImageFileNames(testImageDirectory);
		List<String> trainImageFileNames = Helper.listImageFileNames(trainImageDirectory);
		for(String testImageFileName: testImageFileNames) 
		{
			System.out.println("Computing distance for test image "+testImageFileName);
			double[] imagePixels = Helper.readImagePixels(testImageFileName, width, height);
			double[] diffImagePixels = Helper.computeDifferencePixels(imagePixels, meanPixels);
			double[] weights = Helper.computeWeights(diffImagePixels, eigenFaces);
			double[] reconstructedImagePixels = Helper.reconstructImageWithEigenFaces(
					weights, eigenFaces, meanPixels);
			double distance = Helper.computeImageDistance(imagePixels, reconstructedImagePixels);
			String shortTestFileName = Helper.getShortFileName(testImageFileName);
			System.out.printf("Reconstructed Image distance for %1$s: %2$f\n", shortTestFileName, distance);
			Helper.writeImage(outputDirectory + "/test-ef-" + shortTestFileName, reconstructedImagePixels, width, height);
			Object[] closestImageInfo = findClosestImage(weights, weightMatrix);
			int closestImageIndex = (Integer)closestImageInfo[0];
			double closestImageSimilarity = (Double)closestImageInfo[1];
			System.out.printf("Image %1$s is most similar to %2$s: %3$f\n",
					shortTestFileName,
					Helper.getShortFileName(trainImageFileNames.get(closestImageIndex)),
					closestImageSimilarity);
		}
	}
}
