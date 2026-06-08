import data_structure.*;
import index.Walias;

import java.util.List;
import java.util.logging.LogManager;

import static data_structure.Parameters.*;
import static data_structure.Parameters.initialParameter;
import static utils.FileUtils.readDataset;
import static utils.FileUtils.readQuery;

public class Main {

    public static void main(String[] args) throws Exception {


        String datasetPath = "./src/datasets/dataset/OSM_1M_6Dim.csv";
        String workloadPath = "./src/datasets/workload/GAU_5.0E-5_Query_OSM_100M_6Dim_test.csv";
//        String queryPath = "./src/datasets/workload/UNI_5.0E-5_Query_TCP-100M_test.csv";
        String queryPath = "./src/datasets/workload/qGAU_5.0E-5_Query_OSM_100M_6Dim_test.csv";

//        Parameters.rate = 0.001;
//        Test(datasetPath, workloadPath, queryPath);
//
//        Parameters.rate = 0.01;
//        Test(datasetPath, workloadPath, queryPath);

        STRATEGY_II = true;
        Test(datasetPath, workloadPath, queryPath);
    }

    static int roundTime = 0;

    public static void Test(String datasetPath, String workloadPath, String testWorkloadPath) throws Exception {
        LogManager.getLogManager().reset();

        DataSpace dataSpace = readDataset(datasetPath);

        dataSpace.dataset = dataSpace.dataset.subList(0, 1000000);

        List<Query> workload = readQuery(workloadPath);
//        workload = workload.subList(0, 100);

        List<Query> testWorkload = readQuery(testWorkloadPath);
        long s, e;

        initialParameter(workload, dataSpace, false, false, false);
        System.out.println("Dataset size: " + dataSpace.getDatasetSize() + ", Workload size: " + workload.size());
        s = System.currentTimeMillis();
        Walias walias = new Walias(dataSpace, workload, testWorkload);
        e = System.currentTimeMillis();
        System.out.println("index construct time: " + (e - s) / 1e3 + "s");
        walias.printIndexPerformance();

        walias.drawPartition(roundTime++);

        System.out.println("Once Time End!");
        System.out.println("Once Time End!");
    }
}