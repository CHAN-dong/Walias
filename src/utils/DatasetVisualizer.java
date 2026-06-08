package utils;

import data_structure.DataSpace;
import data_structure.Point;
import data_structure.Query;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.common.PDRectangle;
import org.apache.pdfbox.pdmodel.graphics.image.JPEGFactory;
import org.apache.pdfbox.pdmodel.graphics.image.PDImageXObject;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.HashSet;

import static utils.FileUtils.readDataset;


public class DatasetVisualizer extends JPanel {
    static HashSet<Query> rectangles;
    static HashSet<Query> partitions;

    static HashSet<Query> insertedWorkload;

    private static DataSpace dataset;
    private int panelSize = 1000;
    private static double[] minValues;
    private static double[] maxValues;

    public DatasetVisualizer(HashSet<Query> workload, DataSpace dataset, HashSet<Query> partitions, HashSet<Query> insertedWorkload) {
        this.rectangles = workload;
        this.minValues = dataset.minBound.data;
        this.maxValues = dataset.maxBound.data;
        this.dataset = dataset;
        this.partitions = partitions;
        this.insertedWorkload = insertedWorkload;
    }

    private static void drawAllElements(Graphics2D g2d, int canvasSize,
                                        double[] minValues, double[] maxValues) {
        double scaleX = canvasSize / (double) (maxValues[0] - minValues[0]);
        double scaleY = canvasSize / (double) (maxValues[1] - minValues[1]);

        int offsetX = (int) ((canvasSize - (maxValues[0] - minValues[0]) * scaleX) / 2);
        int offsetY = (int) ((canvasSize - (maxValues[1] - minValues[1]) * scaleY) / 2);

        g2d.setColor(Color.BLACK);
        g2d.setStroke(new BasicStroke(2f));
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
        for (Query rect : partitions) {
            double x = (rect.pointMin.data[0] - minValues[0]) * scaleX + offsetX;
            double y = (rect.pointMin.data[1] - minValues[1]) * scaleY + offsetY;
            double width = (rect.pointMax.data[0] - rect.pointMin.data[0]) * scaleX;
            double height = (rect.pointMax.data[1] - rect.pointMin.data[1]) * scaleY;
            g2d.draw(new Rectangle2D.Double(x, canvasSize - y - height, width, height));
        }

        g2d.setColor(Color.RED);
        g2d.setStroke(new BasicStroke(1f));
        for (Query rect : rectangles) {
            double x = (int) ((rect.pointMin.data[0] - minValues[0]) * scaleX) + offsetX;
            double y = (int) ((rect.pointMin.data[1] - minValues[1]) * scaleY) + offsetY;
            double width = (int) ((rect.pointMax.data[0] - rect.pointMin.data[0]) * scaleX);
            double height = (int) ((rect.pointMax.data[1] - rect.pointMin.data[1]) * scaleY);
            g2d.draw(new Rectangle2D.Double(x, canvasSize - y - height, width, height));
        }

        g2d.setColor(Color.YELLOW);
        g2d.setStroke(new BasicStroke(1f));
        for (Query rect : insertedWorkload) {
            int x = (int) ((rect.pointMin.data[0] - minValues[0]) * scaleX) + offsetX;
            int y = (int) ((rect.pointMin.data[1] - minValues[1]) * scaleY) + offsetY;
            int width = (int) ((rect.pointMax.data[0] - rect.pointMin.data[0]) * scaleX);
            int height = (int) ((rect.pointMax.data[1] - rect.pointMin.data[1]) * scaleY);
            g2d.drawRect(x, canvasSize - y - height, width, height);
        }


        g2d.setColor(Color.BLACK);
        g2d.setStroke(new BasicStroke(0.5f));
        for (Point point : dataset.dataset) {
            double x = (int) ((point.data[0] - minValues[0]) * scaleX) + offsetX;
            double y = (int) ((point.data[1] - minValues[1]) * scaleY) + offsetY;
            g2d.fill(new Ellipse2D.Double(x - 1, canvasSize - y - 1, 2, 2));
        }

    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;

        g2d.setColor(Color.WHITE);
        g2d.fillRect(0, 0, getWidth(), getHeight());

        drawAllElements(g2d, panelSize, minValues, maxValues);
    }

    public static void exportHighResImage(String filePath, int exportSize) {
        BufferedImage image = new BufferedImage(exportSize, exportSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D exportG2d = image.createGraphics();

        exportG2d.setColor(Color.WHITE);
        exportG2d.fillRect(0, 0, exportSize, exportSize);

        drawAllElements(exportG2d, exportSize, minValues, maxValues);

        exportG2d.dispose();

        try (PDDocument document = new PDDocument()) {
            PDPage page = new PDPage(new PDRectangle(exportSize, exportSize));
            document.addPage(page);

            PDImageXObject pdImage = JPEGFactory.createFromImage(document, image);

            try (PDPageContentStream contentStream = new PDPageContentStream(document, page)) {
                contentStream.drawImage(pdImage, 0, 0, exportSize, exportSize);
            }

            document.save(filePath);
            System.out.println("成功导出 PDF 到: " + filePath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

//    public static void exportHighResImage(String filePath, int exportSize) {
//        BufferedImage image = new BufferedImage(exportSize, exportSize, BufferedImage.TYPE_INT_RGB);
//        Graphics2D exportG2d = image.createGraphics();
//
//        // 设置白色背景
//        exportG2d.setColor(Color.WHITE);
//        exportG2d.fillRect(0, 0, exportSize, exportSize);
//
//        // 调用通用绘制方法（使用导出尺寸）
//        drawAllElements(exportG2d, exportSize, minValues, maxValues);
//
//        // 保存文件
//        try {
//            ImageIO.write(image, "PDF", new File(filePath));
//            System.out.println("成功导出图像到: " + filePath);
//        } catch (IOException e) {
//            e.printStackTrace();
//        } finally {
//            exportG2d.dispose();
//        }
//    }



    public static void drawForPartition(HashSet<Query> partitions, HashSet<Query> workload, DataSpace dataset, HashSet<Query> insertedWorkload) {


        DatasetVisualizer panel = new DatasetVisualizer(workload, dataset, partitions, insertedWorkload);
        panel.setPreferredSize(new Dimension(1100, 1100));

//        JFrame frame = new JFrame("2D Rectangle Visualization");
//        frame.add(panel);
//        frame.pack();
//        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//        frame.setVisible(true);
    }

    public static void main(String[] args) throws IOException {
////        RangeDataset dataset = readDataset("./test.txt");
//        DataSpace dataset = readDataset("D:\\paper_source\\work_8\\10-11\\src\\dataset\\" + "Skew_1M", 100000);
////        Points = Points.subList(0, 1000000);
////        HashSet<Point[]> rectangles = readWorkload("D:\\paper_source\\w
////        ork_6\\dataset\\" + "MIX" + "_Workload_" + "OSM_1M_6Dim" + "_R0.6%_D2", 700);
//        HashSet<Point[]> rectangles = readWorkload("D:\\paper_source\\work_8\\10-11\\src\\dataset\\" + "Workload_UNI", 800);
//        JFrame frame = new JFrame("2D Rectangle Visualization");
//        DatasetVisualizer panel = new DatasetVisualizer(rectangles, dataset, dataset.minBound.data, dataset.maxBound.data);
//        panel.setPreferredSize(new Dimension(1000, 1000));
//        frame.add(panel);
//        frame.pack();
//        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//        frame.setVisible(true);
    }
}
