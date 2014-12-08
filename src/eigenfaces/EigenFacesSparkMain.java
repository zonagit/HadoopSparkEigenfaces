package eigenfaces;

import java.io.File;
import java.net.URL;
import java.util.LinkedList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.math.hadoop.stochasticsvd.SSVDCli;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.SingularValueDecomposition;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

/**
 * Adaptation of 
 * Algorithm "Generating EingenFaces with Mahout SVD to recognize persons
 * faces"
 * using stochastic svd and Spark
 */
public class EigenFacesSparkMain 
{
	public static void main(String args[]) throws Exception
	{
		//1. Train the model
		//1.a Read all n images from the training dir
		//1.b Convert each image to greyscale and scale it down
		//1.c Create a matrix M with each column representing an
		//image (so length wxh) and each entry is a value between 0 and 255
		//1.d Average the images pixels to create mean-image.gif
		//1.e Compute the diff matrix DM by subtracting the mean image
		//from M
		//1.f Write the diff matrix to diffmatrix.seq
		//1.g Write the covariance matrix to covariance.seq
		String args1[] = new String[4];
		args1[0] = 80 + "";
		args1[1] = 60 + "";
		URL purl = EigenFacesMain.class.getProtectionDomain().getCodeSource().getLocation();
		File file = new File(purl.toURI());
		args1[2] = file.getParentFile() + "/training-set";//training set location
		args1[3] = file.getParentFile() + "/output";//output dir
		Configuration conf = new Configuration();
		Path output = new Path(file.getParentFile().toString(),"output");
		FileSystem fs = FileSystem.get(output.toUri(), conf);
		fs.delete(output, true);
		
		GenerateCovarianceMatrix gcm = new GenerateCovarianceMatrix();
		//gcm.main(args1);
		Path covMatrix = new Path(file.getParentFile().toString(),"output/covariance.seq");
		//2. Compute the eigenvectors of the covariance matrix
		//using Spark SVD
		SparkConf sconf = new SparkConf().setAppName("Spark SVD").setMaster("local[1]");
		JavaSparkContext jsc = new JavaSparkContext(sconf);
		LinkedList<Vector> rowsList = new LinkedList<Vector>();
		double [][] diffMatrix = gcm.getDiffMatrix(args1);
		for (int i=0;i< diffMatrix.length;i++)
		{
			Vector currentRow = Vectors.dense(diffMatrix[i]);
			rowsList.add(currentRow);
		}
		JavaRDD<Vector> rows = jsc.parallelize(rowsList);
		
		//Create a RowMatrix from JavaRDD<Vector>
		RowMatrix mat = new RowMatrix(rows.rdd());
		
		//Compute the top singular values and corresponding singular vectors
		SingularValueDecomposition<RowMatrix,Matrix> svd = mat.computeSVD(4, true, 1.0E-9d);
		RowMatrix U = svd.U();
		Vector s = svd.s();
		Matrix V = svd.V();
/*		Path tmp = new Path(file.getParentFile().toString() + "/temp" );
		Path tmpCov = new Path(file.getParentFile().toString(),"temp/covariance.seq" );
		fs = FileSystem.get(tmp.toUri(), conf);
		fs.delete(tmp, true);
		fs.copyFromLocalFile(covMatrix, tmpCov);
		String[] args2 = {
		        "--input", tmpCov.toString(),
		        "--output", output.toString(),
		        "--tempDir", tmp.toString(),
		        "--numRows", "151",
		        "--numCols", "151",
		        "--rank", "50",
		        "--symmetric", "false",
		        "--cleansvd", "true"
		};
	//	ToolRunner.run(conf, new DistributedLanczosSolver().new DistributedLanczosSolverJob(), args2);
		//ssvd
		Path difMatrix = new Path(file.getParentFile().toString(),"output/diffmatrix.seq");
		
		Path tmpDif = new Path(file.getParentFile().toString(),"temp/diffmatrix.seq" );
		fs.copyFromLocalFile(covMatrix, tmpDif);
		String[] args22 = {
		        "--input", tmpDif.toString(),
		        "--output", output.toString(),
		        "--tempDir", tmp.toString(),
		        "--computeU", "true",
		        "--computeV", "true",
		        "--rank", "50",
		        "--reduceTasks", "2",
		        "--powerIter", "0"
		};
		SSVDCli ssvdcli = new SSVDCli();
		ssvdcli.main(args22);
		//3. Compute the eingenfaces from the clean eigenvectors
		ComputeEigenFaces cef = new ComputeEigenFaces();
		String[] args3 = {
				file.getParentFile() + "/output/V/v-m-00001",
				file.getParentFile() + "/output/diffmatrix.seq",
				file.getParentFile() + "/output/mean-image.gif",
				80 + "",
				60 + "",
				file.getParentFile() + "/training-set",
				file.getParentFile() + "/output"
		};
		
		cef.main(args3);
		//4. Now test the model
		ComputeDistance cd = new ComputeDistance();
		String[] args4 = {
				file.getParentFile() + "/output/eigenfaces.seq",
				file.getParentFile() + "/output/mean-image.gif",
				file.getParentFile() + "/output/weights.seq",
				80 + "",
				60 + "",
				file.getParentFile() + "/training-set",
				file.getParentFile() + "/testing-set",
				file.getParentFile() + "/output"
		};
		cd.main(args4);
	*/	
	}
}
