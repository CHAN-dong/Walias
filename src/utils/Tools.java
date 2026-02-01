package utils;

import data_structure.DataSpace;
import data_structure.ModelSample;
import data_structure.Point;
import data_structure.Query;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

public class Tools {

    public static int binarySearchInterval(List<Double> points, double value) {
        int left = 0, right = points.size() - 2;
        while (left <= right) {
            int mid = (left + right) >>> 1;
            double low = points.get(mid);
            double high = points.get(mid + 1);

            if (value < low) {
                right = mid - 1;
            } else if (value >= high) {
                left = mid + 1;
            } else {
                return mid;
            }
        }
        return Math.max(0, Math.min(left, points.size() - 2));
    }

    public static List<Query> getSubWorkload(int workloadSize, List<Query> queryList) {

        HashSet<Query> sub = new HashSet<>(queryList);
        List<Query> result = new ArrayList<>();
        for (Query query : sub) {
            result.add(query);
            if (result.size() == workloadSize) {
                break;
            }
        }

        return result;
    }
    public static List<ModelSample> getSubSamples(int size, List<ModelSample> queryList) {
        if (queryList.size() < size) {return queryList;}
        HashSet<ModelSample> sub = new HashSet<>(queryList);
        List<ModelSample> result = new ArrayList<>();
        for (ModelSample query : sub) {
            result.add(query);
            if (result.size() == size) {
                break;
            }
        }

        return result;
    }

    public static void getSubDataset(int size, DataSpace dataset) {
        HashSet<Point> sub = new HashSet<>(dataset.dataset);
        List<Point> result = new ArrayList<>();
        for (Point point : sub) {
            result.add(point);
            if (result.size() == size) {
                break;
            }
        }
        dataset.dataset = result;
    }

    public static int findLeft(List<DataSpace> dataspaces, double queryMin, int dim) {
        int l = 0, r = dataspaces.size() - 1, ans = -1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (dataspaces.get(mid).maxBound.data[dim] >= queryMin) {
                ans = mid;
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return ans;
    }

    public static int findRight(List<DataSpace> dataspaces, double queryMax, int dim) {
        int l = 0, r = dataspaces.size() - 1, ans = -1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (dataspaces.get(mid).minBound.data[dim] <= queryMax) {
                ans = mid;
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return ans;
    }

    public static int upperBound(List<Double> starts, double queryMax) {
        int l = 0, r = starts.size() - 1, ans = -1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (starts.get(mid) <= queryMax) {
                ans = mid;
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return ans;
    }

    public static int lowerBound(List<Double> ends, double queryMin) {
        int l = 0, r = ends.size() - 1, ans = -1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (ends.get(mid) >= queryMin) {
                ans = mid;
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return ans;
    }

    public static int lowerBound(List<Point> dataset, double queryMin, int dim) {
        int l = 0, r = dataset.size() - 1, ans = -1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (dataset.get(mid).data[dim] >= queryMin) {
                ans = mid;
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return ans;
    }

    public static int upperBound(List<Point> dataset, double queryMax, int dim) {
        int l = 0, r = dataset.size() - 1, ans = -1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (dataset.get(mid).data[dim] <= queryMax) {
                ans = mid;
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return ans;
    }

    public static int lowerBound(double[] ends, double queryMin) {
        int l = 0, r = ends.length - 1, ans = -1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (ends[mid] >= queryMin) {
                ans = mid;
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return ans;
    }

    public static int upperBound(double[] starts, double queryMax) {
        int l = 0, r = starts.length - 1, ans = -1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (starts[mid] <= queryMax) {
                ans = mid;
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return ans;
    }
}
