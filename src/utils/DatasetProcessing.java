package utils;

import java.io.*;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

public class DatasetProcessing {

    public static void saveDatasetToFile(int[][] data, String filename, int n) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            for (int i = 0; i < n; i++) {
//                writer.write(String.valueOf(i));
                writer.write(String.valueOf(data[i][0]));
                for (int dim = 1; dim < data[i].length; dim++) {
                    writer.write(" " + data[i][dim]);
                }
//                for (int value : data[i]) {
//                    writer.write(" " + value);
//                }
                writer.newLine();
            }
        } catch (IOException e) {
            System.out.println("failed!" + e.getMessage());
        }
    }

    public static void saveQueryWorkloadToFile(int[][][] queries, String filename) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            for (int[][] query : queries) {
                int d = query[0].length;
                for (int i = 0; i < d; i++) {
                    writer.write(String.valueOf(query[0][i]));
                    if (i != d - 1) {
                        writer.write(" ");
                    } else {
                        writer.write(",");
                    }
                }
                for (int i = 0; i < d; i++) {
                    writer.write(String.valueOf(query[1][i]));
                    if (i != d - 1) {
                        writer.write(" ");
                    }
                }
                writer.newLine();
            }
        } catch (IOException e) {
            System.out.println("failed!" + e.getMessage());
        }
    }


    public static int[][] generateSkewedSpaceData(int n, int d, int[] minValues, int[] maxValues, double r) {
        int[][] data = new int[n][];
        Random random = new Random();
        HashSet<String> st = new HashSet<>();
        int i = 0;
        while (st.size() < n) {
            int[] point = new int[d];
            for (int j = 0; j < d; j++) {
                double u = random.nextDouble();
                double skewedU = 1 - Math.pow(1 - u, r); //
                point[j] = (int) (minValues[j] + skewedU * (maxValues[j] - minValues[j]));
            }
            String str = Arrays.toString(point);
            if (!st.contains(str)) {
                data[i] = point;
                st.add(str);
                ++i;
            }
        }
        return data;
    }


    public static int[][][] generateRectangles_uniform(int n, int dimension, int[] minValues, int[] maxValues, double range) {
        int[][][] rectangles = new int[n][2][dimension];
        double rate = Math.pow(range, 1.0 / dimension);
        int[] len = new int[dimension];
        for (int i = 0; i < dimension; ++i) {
            len[i] = (int) ((maxValues[i] - minValues[i]) *  rate/ 2);
        }

        Random random = new Random();
        for (int i = 0; i < n; i++) {
            int[][] query = new int[2][dimension];
            for (int j = 0; j < dimension; ++j) {
                double u = random.nextDouble();
                int center = (int) (minValues[j] + u * (maxValues[j] - minValues[j]));
                query[0][j] = Math.max(minValues[j], center - len[j]);
                query[1][j] = Math.min(maxValues[j], center + len[j]);
            }
            rectangles[i] = query;
        }
        return rectangles;
    }

    public static int[][][] generateRectangles_gaussian(int n, int dimension, int[] minValues, int[] maxValues, double range, int dataSize) {
        int[][][] rectangles = new int[n][2][dimension];

        int[] len = new int[dimension];
        for (int i = 0; i < dimension; ++i) {
            len[i] = (int) ((maxValues[i] - minValues[i]) * Math.pow(range, 1.0 / dimension) / 2);
        }

        Random random = new Random();

        double[] mu = new double[dimension];
        for (int j = 0; j < dimension; j++) {
            mu[j] = (minValues[j] + maxValues[j]) / 2.0;
        }
        double sigma = dataSize / 1000; //

        for (int i = 0; i < n; i++) {
            int[][] query = new int[2][dimension];
            for (int j = 0; j < dimension; j++) {
                double u1 = random.nextDouble();
                double u2 = random.nextDouble();
                double z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); //
                double offset = z0 * 0.1 * mu[j];

                int center = (int) Math.round(mu[j] + offset);
                center = Math.max(minValues[j], Math.min(maxValues[j], center));

                query[0][j] = Math.max(minValues[j], center - len[j]);
                query[1][j] = Math.min(maxValues[j], center + len[j]);
            }
            rectangles[i] = query;
        }
        return rectangles;
    }

    public static int[][][] generateRectangles_laplace(int n, int dimension, int[] minValues, int[] maxValues, double range, int dataSize) {
        int[][][] rectangles = new int[n][2][dimension];

        int[] len = new int[dimension];
        for (int i = 0; i < dimension; ++i) {
            len[i] = (int) ((maxValues[i] - minValues[i]) * Math.pow(range, 1.0 / dimension) / 2);
        }

        Random random = new Random();

        double[] mu = new double[dimension];
        for (int j = 0; j < dimension; j++) {
            mu[j] = (minValues[j] + maxValues[j]) / 2.0;
        }

        //dataset size / 10
        double b = dataSize / 400.0; //

        for (int i = 0; i < n; i++) {
            int[][] query = new int[2][dimension];
            for (int j = 0; j < dimension; j++) {

                double u = random.nextDouble();
                double offset = -b * Math.signum(u - 0.5) * Math.log(1 - 2 * Math.abs(u - 0.5));

                int center = (int) Math.round(mu[j] + offset);
                center = Math.max(minValues[j], Math.min(maxValues[j], center));

                query[0][j] = Math.max(minValues[j], center - len[j]);
                query[1][j] = Math.min(maxValues[j], center + len[j]);
            }
            rectangles[i] = query;
        }
        return rectangles;
    }



//    public static int[][][] generateRectangles(int n, int dimension, int[] minValues, int[] maxValues, double minRate, double maxRate, double concentration) {
//        int[][][] rectangles = new int[n][2][dimension];
//        Random random = new Random();
//
//        // 计算整个空间的体积
//        double totalVolume = 1.0;
//        for (int i = 0; i < dimension; i++) {
//            totalVolume *= (maxValues[i] - minValues[i]);
//        }
//
//        // 空间中心
//        int[] spaceCenter = new int[dimension];
//        for (int i = 0; i < dimension; i++) {
//            spaceCenter[i] = (minValues[i] + maxValues[i]) / 2;
//        }
//
//        // 标准差（concentration 控制分布的集中度）
//        double[] stdDevs = new double[dimension];
//        for (int i = 0; i < dimension; i++) {
//            // Concentration 越小，分布越集中；越大，分布越分散
//            stdDevs[i] = (maxValues[i] - minValues[i]) * concentration;
//        }
//
//        for (int i = 0; i < n; i++) {
//            int[] lowerBound = new int[dimension];
//            int[] upperBound = new int[dimension];
//            double[] lengths = new double[dimension];
//
//            // 生成符合正态分布的中心点
//            int[] centerPoint = new int[dimension];
//            for (int j = 0; j < dimension; j++) {
//                centerPoint[j] = (int) Math.round(spaceCenter[j] + random.nextGaussian() * stdDevs[j]);
//                centerPoint[j] = Math.max(minValues[j], Math.min(centerPoint[j], maxValues[j]));
//            }
//
//            // 随机确定当前矩形的目标体积（在 minRate 和 maxRate 之间）
//            double targetRate = minRate + random.nextDouble() * (maxRate - minRate);
//            double targetVolume = totalVolume * targetRate;
//
//            // 计算边长，确保总体积接近目标体积，并且各维度边长近似
//            double volume = 1.0;
//            for (int j = 0; j < dimension; j++) {
//                if (j == 0) {
//                    // 第一个维度，根据目标体积和维度数量估算初始边长
//                    lengths[j] = Math.pow(targetVolume, 1.0 / dimension);
//                } else {
//                    // 保持边长比例（不超过±50%变化）
//                    double minAllowed = lengths[j - 1] * 0.5;
//                    double maxAllowed = lengths[j - 1] * 1.5;
//                    int maxPossible = maxValues[j] - minValues[j];
//                    lengths[j] = Math.min(maxAllowed, maxPossible);
//                    lengths[j] = Math.max(minAllowed, lengths[j]);
//                }
//
//                // 确保边长合理
//                lengths[j] = Math.min(maxValues[j] - minValues[j], Math.max(1, (int) lengths[j]));
//                volume *= lengths[j];
//            }
//
//            // 调整所有边长，确保总体积接近目标体积
//            if (volume != 0) {
//                double scaleFactor = Math.pow(targetVolume / volume, 1.0 / dimension);
//                for (int j = 0; j < dimension; j++) {
//                    lengths[j] *= scaleFactor;
//                    lengths[j] = Math.min(maxValues[j] - minValues[j], Math.max(1, (int) lengths[j]));
//                }
//            }
//
//            // 确定每个维度的左下角和右上角坐标
//            for (int j = 0; j < dimension; j++) {
//                // 计算中心点附近的边长，确保不超出空间范围
//                int halfLength = (int) (lengths[j] / 2);
//                lowerBound[j] = centerPoint[j] - halfLength;
//                upperBound[j] = centerPoint[j] + halfLength;
//
//                // 确保边界在空间范围内
//                if (lowerBound[j] < minValues[j]) {
//                    lowerBound[j] = minValues[j];
//                    upperBound[j] = Math.min(maxValues[j], lowerBound[j] + (int) lengths[j]);
//                } else if (upperBound[j] > maxValues[j]) {
//                    upperBound[j] = maxValues[j];
//                    lowerBound[j] = Math.max(minValues[j], upperBound[j] - (int) lengths[j]);
//                }
//            }
//
//            rectangles[i][0] = lowerBound;
//            rectangles[i][1] = upperBound;
//        }
//
//        return rectangles;
//    }


    public static int[][] readBDPDataset(String filePath, int length, int[][] range) {
        HashSet<int[]> st = new HashSet<>();
        range[0][0] = range[0][1] = Integer.MAX_VALUE;
        range[1][0] = range[1][1] = Integer.MIN_VALUE;
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;

            while (st.size() < length && (line = br.readLine()) != null) {
                String[] values = line.split(",");
                int[] data = new int[2];
                for (int i = 0; i < 2; i++) {
                    data[i] = (int) (Double.parseDouble(values[i]) * 800);
                    range[1][i] = Math.max(range[1][i], data[i]);
                    range[0][i] = Math.min(range[0][i], data[i]);
                }
                st.add(data);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        int id = 0;
        int[][] data = new int[length][];
        for (int[] p : st) {
            p[0] -= range[0][0];
            p[1] -= range[0][1];
            data[id++] = p;
        }

        return data;
    }

    public static void generateDatasetQueryWorkload(String datasetPath, String workloadPath, int dimension, double rangeRate, String type, int dataSize) {
        int[][] range = new int[2][dimension];
        readDataset(datasetPath, dataSize, dimension, range);
        int[][][] workloads = null;
        switch (type) {
            case "UNI":
                workloads = generateRectangles_uniform(1000, dimension, range[0], range[1], rangeRate);
                saveQueryWorkloadToFile(workloads, workloadPath);
                break;
            case "GAU":
                workloads = generateRectangles_gaussian(1000, dimension, range[0], range[1], rangeRate, dataSize);
                saveQueryWorkloadToFile(workloads, workloadPath);
                break;
            case "LAP":
                workloads = generateRectangles_laplace(1000, dimension, range[0], range[1], rangeRate, dataSize);
                saveQueryWorkloadToFile(workloads, workloadPath);
                break;
            case "MIX":
                int[][][] workloads1 = generateRectangles_uniform(500, dimension, range[0], range[1], rangeRate);
                int[][][] workloads2 = generateRectangles_gaussian(500, dimension, range[0], range[1], rangeRate, dataSize);
                workloads = Arrays.copyOf(workloads1, workloads1.length + workloads2.length);
                System.arraycopy(workloads2, 0, workloads, workloads1.length, workloads2.length);
                saveQueryWorkloadToFile(workloads, workloadPath);
        }
    }

    public static int[][] readDataset(String filePath, int n, int dimension, int[][] range) {
        int[][] dataset = new int[n][dimension];
        Arrays.fill(range[0], Integer.MAX_VALUE);
        Arrays.fill(range[1], Integer.MIN_VALUE);
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            int id = 0;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(" ");
                int[] data = new int[dimension];
                for (int i = 0; i < dimension; i++) {
                    data[i] = Integer.parseInt(values[i]);
                    range[1][i] = Math.max(range[1][i], data[i]);
                    range[0][i] = Math.min(range[0][i], data[i]);
                }
                dataset[id] = data;
                id++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return dataset;
    }



    public static void main(String[] args) {

//        int n = 1000000, upd_n = 100000;
        int[] minValues = new int[] {0,0, 0, 0, 0, 0};
        int[] maxValues = new int[] {2000000,2000000, 2000000, 2000000, 2000000, 2000000};


             minValues = new int[20]; Arrays.fill(minValues, 0);
             maxValues = new int[20]; Arrays.fill(maxValues, 2000000);
            int[][] dataset = generateSkewedSpaceData(1000000, 20, minValues, maxValues,2);
            saveDatasetToFile(dataset, "D:\\paper_source\\work_8\\dimension_dataset\\Skew_1M_" + 20 + "dim", 1000000);

            for (int i = 0; i <= 20; ++i) {
                generateDatasetQueryWorkload("D:\\paper_source\\work_8\\dimension_dataset\\Skew_1M_20dim", "D:\\paper_source\\work_8\\dimension_dataset\\Skew_1M_query_" + i + "dim", i, 0.00005, "MIX", 1000000);
            }



////
//        int[][] dataset = generateSkewedSpaceData(1000000, 2, minValues, maxValues,2);

//        int[][] dataset = readBDPDataset("D:\\paper_source\\work_6\\dataset\\OSM.csv", 2000000, new int[2][2]);
//        saveDatasetToFile(dataset, "D:\\paper_source\\work_8\\10-11\\src\\dataset\\Skew_1M", 1000000);
//        generateDatasetQueryWorkload("D:\\paper_source\\work_8\\10-11\\src\\dataset\\" + "Skew" + "_1M", "D:\\paper_source\\work_8\\10-11\\src\\dataset\\" + "_Workload_" + "Skew", 2, 0.01 * 0.01, "UNI", 1000000);


//        generateDatasetQueryWorkload("D:\\paper_source\\work_6\\dataset\\Skew_1M", "D:\\paper_source\\work_8\\dataset\\LAP_Query_Skew_1M_test", 2, 0.0001, "LAP", 1000000);

//        String[] datasetNames = new String[] {"OSM_1M"};
/*        String[] typeNames = new String[] {"UNI", "GAU", "LAP", "MIX"};
        for (String datasetName : datasetNames) {
            for (String type : typeNames) {

                for (int i = 3; i == 3; ++i) {
                    double rangeRate;
                    if (i == 3) {
                        rangeRate = 0.6;
                    } else {
                        rangeRate = 0.2 * i;
                    }

                    String datasetPath = "D:\\paper_source\\work_6\\dataset\\OSM_1M";
//                        String datasetPath = "D:\\paper_source\\work_6\\dataset\\" + datasetName;
                    String workloadPath = "D:\\paper_source\\work_6\\dataset\\" + type + "_Workload_OSM_updateTest";
                    generateDatasetQueryWorkload(datasetPath, workloadPath, 2, rangeRate * 0.01, type, 1000000);

                }
            }
        }*/



    }



}
