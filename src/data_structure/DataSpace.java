package data_structure;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static data_structure.Parameters.*;


public class DataSpace {

    public Point minBound;
    public Point maxBound;
    public List<Point> dataset;    // point data set

    public Query getMBR() {
        return new Query(minBound, maxBound);
    }

    public DataSpace(Point minBound, Point maxBound, List<Point> dataset) {
        this.minBound = minBound;
        this.maxBound = maxBound;
        this.dataset = dataset;
    }

    public DataSpace() {
        minBound = new Point(false);
        maxBound = new Point(true);
        this.dataset = new ArrayList<Point>();
    }

    public void add(Point p) {
        dataset.add(p);
        for (int i = 0; i < D; ++i) {
            minBound.data[i] = Math.min(minBound.data[i], p.data[i]);
            maxBound.data[i] = Math.max(maxBound.data[i], p.data[i]);
        }
    }

    public int getDatasetSize() {
        return dataset.size();
    }


    // When the dataset exceeds the threshold, select a dimension for uniform division
//    public Object[] spaceAverageSplit() {
//        List<DataSpace> result = new ArrayList<>();
//        if (dataset.size() <= miniPageSize) {
//            return null;
//        }
//
//        int dim = minBound.data.length;
//        // 1. select the dimension with the largest span
//        int splitDim = 0;
//        double maxRange = 0;
//        for (int i = 0; i < dim; i++) {
//            double range = (maxBound.data[i] - minBound.data[i]) / metaLength[i];
//            if (range > maxRange) {
//                maxRange = range;
//                splitDim = i;
//            }
//        }
//
//        // 1. sort by splitdim dimension
//        int finalSplitDim = splitDim;
//        dataset.sort(Comparator.comparingDouble(p -> p.data[finalSplitDim]));
//
//        int n = dataset.size();
//        int perPart = (int) Math.ceil(n * 1.0 / miniPageSize);
//
//        for (int i = 0; i < miniPageSize; i++) {
//            int from = i * perPart;
//            int to = Math.min((i + 1) * perPart, n);
//            if (from >= to) break;
//
//            List<Point> subList = dataset.subList(from, to);
//
//            // calculate the min max coordinates for this subset
//            Point newMin = new Point();
//            Point newMax = new Point();
//            Arrays.fill(newMin.data, Double.POSITIVE_INFINITY);
//            Arrays.fill(newMax.data, Double.NEGATIVE_INFINITY);
//            for (Point p : subList) {
//                for (int d = 0; d < dim; d++) {
//                    newMin.data[d] = Math.min(newMin.data[d], p.data[d]);
//                    newMax.data[d] = Math.max(newMax.data[d], p.data[d]);
//                }
//            }
//
//            DataSpace subSpace = new DataSpace();
//            subSpace.minBound = newMin;
//            subSpace.maxBound = newMax;
//            subSpace.dataset = new ArrayList<>(subList);
//
//            result.add(subSpace);
//        }
//
//        return new Object[]{dim, result};
//    }

}
