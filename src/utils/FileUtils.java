package utils;

import data_structure.DataSpace;
import data_structure.Point;
import data_structure.Query;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static data_structure.Parameters.D;

public class FileUtils {
    public static double[] mergeArraysAlternately(double[] array1, double[] array2) {
        if (array1 == null) return array2 == null ? new double[0] : array2.clone();
        if (array2 == null) return array1.clone();

        int len1 = array1.length;
        int len2 = array2.length;
        double[] result = new double[len1 + len2];

        int minLen = Math.min(len1, len2);
        int i = 0;

        // 使用循环展开优化，减少循环次数
        for (; i < minLen; i++) {
            result[2 * i] = array1[i];
            result[2 * i + 1] = array2[i];
        }

        // 处理剩余元素
        if (len1 > len2) {
            System.arraycopy(array1, i, result, 2 * i, len1 - i);
        } else if (len2 > len1) {
            System.arraycopy(array2, i, result, 2 * i, len2 - i);
        }

        return result;
    }
    /**
     * 更新数据范围的最小最大值
     * @param data 当前数据点
     * @param dataMin 当前x最小值
    //     * @param dataMax 当前x最大值
     * @param dataMin 当前y最小值
    //     * @param dataMax 当前y最大值
     * @return 更新后的范围值数组
     */
    public static double[] rangeMinMax(double[] data, double[] dataMin, double[] dataMax) {
        for (int i = 0; i < data.length; i++){
            if (dataMin[i] > data[i]) {
                dataMin[i] = data[i];
            }
            if (dataMax[i] < data[i]) {
                dataMax[i] = data[i];
            }
        }

        return mergeArraysAlternately(dataMin, dataMax);// 合并两个数组交替合并
    }

    public static DataSpace readDataset(String fileName) throws IOException {
        return readDataset(fileName, Integer.MAX_VALUE);
    }
    public static DataSpace readDataset(String fileName, int dataSize) throws IOException {
        List<Point> pointList = new ArrayList<>();

        double[] dataMin = Collections.nCopies(D, Double.MAX_VALUE)
                .stream()
                .mapToDouble(Double::doubleValue)
                .toArray();
        double[] dataMax = Collections.nCopies(D, - Double.MAX_VALUE)
                .stream()
                .mapToDouble(Double::doubleValue)
                .toArray();


        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line;
            int size_i = 0;  // DATA数量计数器
            while ((line = br.readLine()) != null) {

                String[] components = line.trim().split(" "); //按，分割字符串
//                String[] components = line.trim().split("\\s+"); //按空白字符分割字符串
                if (components.length >= 2 && (size_i < dataSize) ) {
                    double[] pointData = new double[D];
                    for(int i = 0; i < D; i++){  // 将字符串转换为数字，控制数据维度
                        double d = Double.parseDouble(components[i]);
                        pointData[i] = d;
                    }
                    // 更新范围值
                    double[] newRange = rangeMinMax(pointData, dataMin, dataMax);
                    for(int i = 0; i < D; i++){
                        dataMin[i] = newRange[2 * i];
                        dataMax[i] = newRange[2 * i + 1];
                    }
                    Point point = new Point(pointData);
                    pointList.add(point);
                }
                size_i++;
            }
        }

        DataSpace dataset = new DataSpace(new Point(dataMin), new Point(dataMax), pointList);

        return dataset;
    }

    public static List<Query> readQuery(String queryFile) {
        List<Query> querySet = new ArrayList<>();  // 存储查询对象的列表

        try (BufferedReader br = new BufferedReader(new FileReader(queryFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                // 处理每一行数据
                double[][] nDimRectangle = parseRectangleComponents(line, D);

                Query query = new Query(new Point(nDimRectangle[0]), new Point(nDimRectangle[1]));
                querySet.add(query);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }


        return querySet;
    }

    public static double[][] parseRectangleComponents(String line, int dimensions) {
        // 清理字符串：去除多余空格和特殊字符
        String cleaned = line.trim()
                .replaceAll("\\s+", " ")  // 多个空格合并为一个
                .replaceAll(",", " ");     // 逗号替换为空格

        String[] components = cleaned.split(" ");

        // 过滤空字符串
        components = Arrays.stream(components)
                .filter(s -> !s.isEmpty())
                .toArray(String[]::new);
//        if (components.length != dimensions * 2) {
//            throw new IllegalArgumentException(String.format(
//                    "数据格式错误! 期望%d个数字(%d维矩形)，实际找到%d个。数据: %s",
//                    dimensions * 2, dimensions, components.length, line));
//        }

        double[][] result = new double[2][dimensions];

        for (int i = 0; i < dimensions; i++) {
            result[0][i] = Double.parseDouble(components[i]);
            result[1][i] = Double.parseDouble(components[components.length / 2 + i]);
        }

        return result;
    }

}
