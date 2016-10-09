package org.opencv.samples.selfpotrait;

import java.io.File;
import java.io.FileOutputStream;

import android.os.Bundle;
import android.widget.ImageView;
import android.widget.Toast;
import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;

public class ShowResultActivity extends Activity {
	private ImageView imageView;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_show_result);
		
		imageView = (ImageView) findViewById(R.id.imageView1);
		
		if (getIntent().getExtras() != null) {
			String imagePath = getIntent().getExtras().getString("FILENAME");
			File pictureFile = new File(imagePath);
			
			Bitmap bmp = BitmapFactory.decodeFile(imagePath);
			
			// others devices
			int orientation = 0;
	        if(bmp.getHeight() < bmp.getWidth()){
	            orientation = 90;
	        } 

	        Bitmap bMapRotate;
	        if (orientation != 0) {
	            Matrix matrix = new Matrix();
	            matrix.postRotate(orientation);
	            bMapRotate = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(),
	            		bmp.getHeight(), matrix, true);
	        } else {
	            bMapRotate = Bitmap.createScaledBitmap(bmp, bmp.getWidth(),
	            		bmp.getHeight(), true);
	        }
				
			try {
				FileOutputStream fos = new FileOutputStream(pictureFile);
				bMapRotate.compress(Bitmap.CompressFormat.PNG, 90, fos);
				fos.flush();
				fos.close();
				
				Toast.makeText(this, "New Image saved", Toast.LENGTH_LONG).show();
			} catch (Exception error) {
			}
			
			if (bMapRotate != null) {
				imageView.setImageBitmap(bMapRotate);
			}
		}
	}

}
