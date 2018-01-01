package org.gms.ml.display;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import javax.swing.JFrame;
import javax.swing.JPanel;
 
import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;

@SuppressWarnings("serial")
public class DisplayImageSwing extends JPanel {
	
	private static DisplayImageSwing Instance;
	private JFrame f;
	private Image img;
	
	public static void display(double[] pixels, int width, int height) {
		if(Instance == null) {
			Instance = new DisplayImageSwing();
		}
		
		Instance.print(pixels,width,height);
	}
	
	private void print(double[] pixels, int width, int height) {
		img = getImageFromArray(pixels, width, height);
		if(f == null) {
			f = new JFrame("prueba");
			f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			f.add(this);
			f.pack();
			f.setVisible(true);
		}else {
			repaint();
		}
	}
	
	
	@Override
	public void paint(Graphics g) {
		Graphics2D g2 = (Graphics2D) g;
		 g2.drawImage(img, 0, 0, null);
		
	}

	public static void main(String[] args) throws IOException {
		MatFileReader mfr = new MatFileReader(new File("ex3data1.mat"));
		MLDouble XMat = (MLDouble) mfr.getMLArray("X");

		double[][] Xd = XMat.getArray();
		
		display(Xd[3], 20, 20);
		System.out.println(Xd[3].length);
		System.out.println(Arrays.toString(Xd[3]));
	}
	
//	private Image getImageFromArray(double[] pixels, int width, int height) {
//        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_INDEXED);
//        WritableRaster raster = (WritableRaster) image.getData();
//        raster.setPixels(0,0,width,height,pixels);
//        return image;
//    }
	
	private Image getImageFromArray(double[] pixels, int width, int height) {
		BufferedImage b = new BufferedImage(width, height, 3);
		int[] transform = new int[pixels.length];
		for(int x = 0; x < pixels.length; x++) {
	        transform[x] = (int)pixels[x]<<16 | (int)pixels[x] << 8 | (int)pixels[x];
		}
		System.out.println(Arrays.toString(pixels));
		System.out.println(Arrays.toString(transform));
		WritableRaster raster = (WritableRaster) b.getData();
		raster.setPixels(0,0,width,height,transform);
		return b;
	}
}
