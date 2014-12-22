package eigenfaces;

import java.io.File;
import java.net.URL;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterable;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.decomposer.DistributedLanczosSolver;
import org.apache.mahout.math.hadoop.decomposer.EigenVector;
import org.apache.mahout.math.hadoop.decomposer.EigenVerificationJob;

/**
 * Adaptation of 
 * Algorithm "Generating EingenFaces with Mahout SVD to recognize persons
 * faces"
 * 
 */
public class EigenFacesMain 
{
	public static void main(String args[]) throws Exception
	{
		int rank = 50;
		if (args != null && args.length>0)
		{
			rank = Integer.parseInt(args[0]);
		}
		URL purl = EigenFacesMain.class.getProtectionDomain().getCodeSource().getLocation();
		File file = new File(purl.toURI());
		String input_location = file.getParentFile().toString();
		if (args!= null && args.length>1)
		{
			input_location = args[1];
		}
		String output_location = input_location;
		if (args!= null && args.length>2)
		{
			output_location = args[2];
		}
		//1. Train the model
		//1.a Read all n images from the training dir
		//1.b Convert each image to greyscale and scale it down
		//1.c Create a matrix M with each column representing an
		//image (so length wxh) and each entry is a value between 0 and 255
		//1.d Average the images pixels to create mean-image.gif
		//1.e Compute the diff matrix DM by subtracting the mean image
		//from M
		//1.f Write the diff matrix to diffmatrix.seq
		//1.g Write the covarianc50e matrix to covariance.seq
		String args1[] = new String[4];
		args1[0] = 80 + "";
		args1[1] = 60 + "";
		args1[2] = input_location + "/training-set";//training set location
		args1[3] = output_location + "/output";//output dir
		Configuration conf = new Configuration();
		Path output = new Path(output_location,"output");
		FileSystem fs = FileSystem.get(output.toUri(), conf);
		fs.delete(output, true);
		
		GenerateCovarianceMatrix gcm = new GenerateCovarianceMatrix();
		gcm.main(args1);
		Path covMatrix = new Path(output_location,"output/covariance.seq");
		//2. Compute the eigenvectors of the covariance matrix
		//using Mahout SVD
		Path tmp = new Path(output_location + "/temp" );
		Path tmpCov = new Path(output_location,"temp/covariance.seq" );
		fs = FileSystem.get(tmp.toUri(), conf);
		fs.delete(tmp, true);
		if (input_location.equals(output_location))
		{
			System.out.println("Moving from local to hdfs");
			fs.copyFromLocalFile(covMatrix, tmpCov);
		}
		else
		{
			System.out.println("Renaming hdfs folders from " + covMatrix.toString() + " " + tmpCov.toString());
			boolean successfulRen = fs.rename(covMatrix, tmpCov);
			if (!successfulRen)
				System.out.println("Unable to rename ");
		}
		String[] args2 = {
		        "--input", covMatrix.toString(),
		        "--output", output.toString(),
		        "--tempDir", tmp.toString(),
		        "--numRows", "151",
		        "--numCols", "151",
		        "--rank", rank + "",
		        "--symmetric", "false",
		        "--cleansvd", "true"
		};
		System.out.println("Starting covariance eigenvector computation ");
		
		ToolRunner.run(conf, new DistributedLanczosSolver().new DistributedLanczosSolverJob(), args2);
		//Print eigenvalues
		System.out.println("Printing eigenvalues ");
		int i = 0;
		Path cleanEigenvectors = new Path(output, EigenVerificationJob.CLEAN_EIGENVECTORS);
		Matrix eigenVectors = new DenseMatrix(rank, 151);
		 
		for (VectorWritable value : new SequenceFileValueIterable<VectorWritable>(
				cleanEigenvectors, conf)) 
		{			
		      NamedVector v = (NamedVector) value.get();
		      eigenVectors.assignRow(i, v);
		      if (EigenVector.getCosAngleError(v.getName()) < 1.0e-3) 
		      {
		    	  System.out.println("Small error (<10e-3) " + EigenVector.getEigenValue(v.getName()));
		      }
		      else
		      {
		    	  System.out.println("Error > 10e-3" + EigenVector.getEigenValue(v.getName()));
		      }
		      i++;
		}
		System.out.println("Computing eigenfaces");
		//3. Compute the eingenfaces from the clean eigenvectors
		ComputeEigenFaces cef = new ComputeEigenFaces();
		String[] args3 = {
				output_location + "/output/cleanEigenvectors",
				output_location + "/output/diffmatrix.seq",
				input_location + "/training-set/mean-image.gif",
				80 + "",
				60 + "",
				input_location + "/training-set",
				output_location + "/output",
				input_location + "/images"
		};
		
		cef.main(args3);
		//4. Now test the model
		System.out.println("Testing stage");
		ComputeDistance cd = new ComputeDistance();
		String[] args4 = {
				output_location + "/output/eigenfaces.seq",
				input_location + "/training-set/mean-image.gif",
				output_location + "/output/weights.seq",
				80 + "",
				60 + "",
				input_location + "/training-set",
				input_location + "/testing-set",
				output_location + "/output",
				input_location + "/images"
		};
		cd.main(args4);		
	}
}
