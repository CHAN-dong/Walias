package index;
import data_structure.ModelSample;
import data_structure.Point;
import data_structure.Query;

import java.util.List;
import java.util.Queue;

import static data_structure.Parameters.*;
import static index.Walias.stats;
import static utils.Tools.lowerBound;
import static utils.Tools.upperBound;

public class Node {

    ModelSample[][] samples;

    double cost;

    public List<Point> dataset;
    List<Query> MBRs;
    List<Node> childes;

    int[] splits;

    public int splitDimension;

    public void initSamples() {
        if (this.isLeafNode()) return;
        int n = childes.size();
        samples = new ModelSample[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n && j - i < T; j++) {
                samples[i][j] = new ModelSample();
            }
        }

    }


    public Node(List<Point> dataset, double cost, int splitDimension) {
        this.dataset = dataset;
        this.splitDimension = splitDimension;
    }

    public Node(int splitDimension, List<Query> MBRs, List<Node> childes, double cost) {
        this.MBRs = MBRs;
        this.childes = childes;
        this.splitDimension = splitDimension;
    }

    public boolean isLeafNode() {
        return dataset != null;
    }

    public void leafNodeQuery(Query query, List<Point> res) {
        long s = System.nanoTime();
        int sz = res.size();
        int left = binarySearchLeftPoints(query.pointMin.data[splitDimension]);
        int right = binarySearchRightPoints(query.pointMax.data[splitDimension]);

        for (int i = left; i <= right; i++) {
            Point p = dataset.get(i);
            if (query.hasIntersection(p, splitDimension)) {
                res.add(p);
            }
        }
        long e = System.nanoTime();
        Walias.TMP += (e - s);
        stats.leafNode.add(new Long[]{
                (e - s),  // time
                (long)this.dataset.size(), // objects size
                (long)(right - left + 1), // candidate size
                (long)(res.size() - sz)
        });
    }

    private int binarySearchLeftPoints(double qMin) {
        int left = 0;
        int right = dataset.size() - 1;
        int res = dataset.size();
        while (left <= right) {
            int mid = (left + right) / 2;
            if (dataset.get(mid).data[splitDimension] >= qMin) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    private int binarySearchRightPoints(double qMax) {
        int left = 0;
        int right = dataset.size() - 1;
        int res = -1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (dataset.get(mid).data[splitDimension] <= qMax) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    public void nonLeafNodeQuery(Query query, Queue<Node> queue) {

        long s = System.nanoTime();


        int before = queue.size();

        int left = binarySearchLeft(query.pointMin.data[splitDimension]);
        int right = binarySearchRight(query.pointMax.data[splitDimension]);

        for (int i = left; i <= right; i++) {
            if (query.hasIntersection(MBRs.get(i), splitDimension)) {
                queue.add(childes.get(i));
            }
        }
        long e = System.nanoTime();
        Walias.TMP += (e - s);
        stats.nonLeafNode.add(new Long[]{
                (e - s),  // time
                (long)this.childes.size(), // objects size
                (long)(right - left + 1), // candidate size
                (long)(queue.size() - before)
        });
    }

    private int binarySearchLeft(double qMin) {
        int left = 0;
        int right = MBRs.size() - 1;
        int res = MBRs.size();
        while (left <= right) {
            int mid = (left + right) / 2;
            if (MBRs.get(mid).pointMax.data[splitDimension] >= qMin) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    private int binarySearchRight(double qMax) {
        int left = 0;
        int right = MBRs.size() - 1;
        int res = -1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (MBRs.get(mid).pointMin.data[splitDimension] <= qMax) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }


    public static double getNonLeafNodeCost(int datasetSize, int workloadSize, int candidateSize, int childSize) {
        return workloadSize* (ALFA + BETA * Math.log(datasetSize)) + GAMA * (D-1) * candidateSize + SIGMA * (20 + childSize * (8 * D + 8));
    }

    public static double getLeafNodeCost(int datasetSize, int workloadSize, int candidateSize) {
        return workloadSize* (ALFA + BETA * Math.log(datasetSize)) + GAMA * (D-1) * candidateSize + SIGMA * 20;
    }

    public static double getLeafNodeCost(List<Point> dataset, List<Query> workload, int sortDimension) {

        if (workload == null || workload.size() == 0) return  0;

        int candidateSize = 0;

        for (Query query : workload) {
            double left = query.pointMin.data[sortDimension];
            double right = query.pointMax.data[sortDimension];

            int leftIndex = lowerBound(dataset, left, sortDimension);
            int rightIndex = upperBound(dataset, right, sortDimension); //

            candidateSize += (rightIndex - leftIndex + 1);
        }

        return workload.size() * (ALFA + BETA * Math.log(dataset.size())) + GAMA * candidateSize * (D -1) + SIGMA * 20;
    }


    public Probe leafNodeQueryProbe(Query query) {
        final int n = dataset.size();
        if (n == 0) return new Probe(0, 0L, 0);

        long t0 = System.nanoTime();

        int left = binarySearchLeftPoints(query.pointMin.data[splitDimension]);
        int right = binarySearchRightPoints(query.pointMax.data[splitDimension]);

        int can = 0;
        int hit = 0;
        if (left <= right) {
            can = (right - left + 1);
            for (int i = left; i <= right; i++) {
                Point p = dataset.get(i);
                if (query.hasIntersection(p, splitDimension)) {
                    hit++;
                }
            }
        }

        long t1 = System.nanoTime();
        return new Probe(can, (t1 - t0), hit);
    }

    public static class Probe {
        public final int can;
        public final long timeNs;
        public final int hitCount;
        public Probe(int can, long timeNs, int hitCount) {
            this.can = can;
            this.timeNs = timeNs;
            this.hitCount = hitCount;
        }
    }

}
