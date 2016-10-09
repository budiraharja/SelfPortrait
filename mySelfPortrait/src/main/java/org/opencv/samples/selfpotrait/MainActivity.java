package org.opencv.samples.selfpotrait;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Date;
import java.util.Timer;
import java.util.TimerTask;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.Utils;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.samples.selfpotrait.R;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends Activity implements CvCameraViewListener2 {

    private static final String    TAG                 				= "OCVSample::Activity";
    public static final int        JAVA_DETECTOR       				= 0;
    private static final Scalar    FACE_RECT_COLOR     				= new Scalar(0, 255, 0, 255);
    private static final Scalar    PALM_RECT_COLOR     				= new Scalar(255, 0, 0, 255);
    private static final Scalar    FIST_RECT_COLOR     				= new Scalar(0, 0, 255, 255);
    public static final int        NATIVE_DETECTOR     				= 1;
    public static final int		   SCALE			   				= 8;
    public static final int		   SCALE_UP			   				= 2;
    private static final int	   DIFF_X			   				= 1;
    private static final int	   DIFF_Y			   				= 100;
    private static final int 	   CAMERA_TRIGGER_MAX_MILLIS 		= 5000;
    private static final int 	   CAMERA_TRIGGER_INTERVAL_MILLIS 	= 1000;

    private MenuItem               mItemType;
    private MenuItem               mReset;

    private Mat                    mRgba;
    private Mat                    mGray;
    private File                   mCascadeFaceFile;
    private File                   mCascadeFistFile;
    private File                   mCascadePalmFile;
    private CascadeClassifier      mJavaFistDetector;
    private DetectionBasedTracker  mNativeFistDetector;
    private CascadeClassifier      mJavaPalmDetector;
    private DetectionBasedTracker  mNativePalmDetector;
    private CascadeClassifier      mJavaFaceDetector;
    private DetectionBasedTracker  mNativeFaceDetector;

    private int                    mDetectorType       = JAVA_DETECTOR;
    private String[]               mDetectorName;

    private float                  mRelativePalmSize   = 0.2f;
    private int                    mAbsolutePalmSize   = 0;
    
    private Rect rectFace;
    private Rect rectPalm;
    private Rect rectFist;
    private MatOfRect matRectOfFace;
    private MatOfRect matRectOfPalm;
    private MatOfRect matRectOfFist;
    private boolean isPalmDetected = false;
    private int holdFist = 0;
    private Timer timer;
    private CountDownTimer countDownTimer;
    private int timerSecond = 0;
    private boolean isStartTakingPicture = false;
    private boolean isTakePicture = false;
    
    private static boolean isLoaded = false;
    
    private TextView textView;

    private CameraBridgeViewBase   mOpenCvCameraView;
    
    static {
    	if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
    	} else {
    		// Load native library after(!) OpenCV initialization
    		System.loadLibrary("detection_based_tracker");
    	}
    }

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    try {
                        // load cascade files from application resources
                    	// face cascade
                        InputStream isFace = getResources().openRawResource(R.raw.haarcascade_frontalface_alt2);
                        File cascadeFaceDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFaceFile = new File(cascadeFaceDir, "haarcascade_frontalface_alt2.xml");
                        FileOutputStream osFace = new FileOutputStream(mCascadeFaceFile);

                        byte[] bufferFace = new byte[4096];
                        int bytesReadFace;
                        while ((bytesReadFace = isFace.read(bufferFace)) != -1) {
                            osFace.write(bufferFace, 0, bytesReadFace);
                        }
                        isFace.close();
                        osFace.close();

                        mJavaFaceDetector = new CascadeClassifier(mCascadeFaceFile.getAbsolutePath());
                        if (mJavaFaceDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaFaceDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFaceFile.getAbsolutePath());

                        mNativeFaceDetector = new DetectionBasedTracker(mCascadeFaceFile.getAbsolutePath(), 0);
                    	
                        // palm cascade
                        InputStream isPalm = getResources().openRawResource(R.raw.palm);
                        File cascadePalmDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadePalmFile = new File(cascadePalmDir, "palm.xml");
                        FileOutputStream osPalm = new FileOutputStream(mCascadePalmFile);

                        byte[] bufferPalm = new byte[4096];
                        int bytesReadPalm;
                        while ((bytesReadPalm = isPalm.read(bufferPalm)) != -1) {
                            osPalm.write(bufferPalm, 0, bytesReadPalm);
                        }
                        isPalm.close();
                        osPalm.close();

                        mJavaPalmDetector = new CascadeClassifier(mCascadePalmFile.getAbsolutePath());
                        if (mJavaPalmDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaPalmDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadePalmFile.getAbsolutePath());

                        mNativePalmDetector = new DetectionBasedTracker(mCascadePalmFile.getAbsolutePath(), 0);
                        
                    	// fist cascade
                        InputStream is = getResources().openRawResource(R.raw.fist);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFistFile = new File(cascadeDir, "fist.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFistFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaFistDetector = new CascadeClassifier(mCascadeFistFile.getAbsolutePath());
                        if (mJavaFistDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaFistDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFistFile.getAbsolutePath());

                        mNativeFistDetector = new DetectionBasedTracker(mCascadeFistFile.getAbsolutePath(), 0);
                        
                        matRectOfFace = new MatOfRect();
                        matRectOfPalm = new MatOfRect();
                        matRectOfFist = new MatOfRect();
                        
                        cascadeFaceDir.delete();
                        cascadePalmDir.delete();
                        cascadeDir.delete();
                        
                        isLoaded = true;

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public MainActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.hand_detect_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
        
        textView = (TextView) findViewById(R.id.textView1);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        reset();
        isLoaded = false;
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        reset();
        //if(!OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_2, this, mLoaderCallback)) {
        if (!isLoaded) {
        	mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    	//}
    }

    public void onDestroy() {
        super.onDestroy();
        isLoaded = false;
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        
        if (!isStartTakingPicture) {
        	mGray = inputFrame.gray();
        	
	        //scale frame to make process faster
	        Imgproc.resize(mGray, mGray, new Size(mGray.width() / SCALE, mGray.height() / SCALE));
	        
	        //rotate image for portrait orientation
	        if (getResources().getConfiguration().orientation != Configuration.ORIENTATION_PORTRAIT) {
        		Mat tmpGray = new Mat(); 
            	Core.transpose(mGray, tmpGray); 
            	Core.flip(tmpGray, mGray, 0);
            	
            	tmpGray.release();
        	}
	        
	        //Highgui.imwrite(Environment.getExternalStorageDirectory() + "/opencv/test1.jpg", mGray);
	        
	        //detect face
	        mNativeFaceDetector.setMinPalmSize(0);
		    mNativePalmDetector.setMinPalmSize(0);
		    mNativeFistDetector.setMinPalmSize(0);
		    
		    if (mDetectorType == JAVA_DETECTOR) {
		        if (mJavaFaceDetector != null)
		        	mJavaFaceDetector.detectMultiScale(mGray, matRectOfFace, 1.1, 0, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
		                    new Size(1, 1), new Size());
		    }
		    else if (mDetectorType == NATIVE_DETECTOR) {
		        if (mNativeFaceDetector != null)
		        	mNativeFaceDetector.detect(mGray, matRectOfFace);
		    }
		    else {
		        Log.e(TAG, "Detection method is not selected!");
		    }
		    
		    Rect rectFaceClone = null;
		
		    Rect[] faceArray = matRectOfFace.toArray();
		    rectFace = null;
		    for (int i = 0; i < faceArray.length; i++) {
		    //if (faceArray != null && faceArray.length > 0) {
		    	Rect rectFaceTemp = faceArray[i].clone();
		       	//resize rect to original frame size
		    	rectFaceTemp.height *= SCALE;
		    	rectFaceTemp.width *= SCALE;
		    	rectFaceTemp.x *= SCALE;
		    	rectFaceTemp.y *= SCALE;
		       	
		    	//find the biggest face rect
		       	if (rectFace == null || (rectFaceTemp.width > rectFace.width && rectFaceTemp.height > rectFace.height)) {
		       		rectFace = rectFaceTemp.clone();
		       	}
		    //}
		    } 
		    
		    if (rectFace != null) {
		    	//draw rect palm on the right of face's rect
		    	rectFaceClone = rectFace.clone();
	        	rectFaceClone.x -= (rectFace.width + ((int) (rectFace.width / 5)));
		    	
		    	//rotate image for portrait orientation before draw rect
		        if (getResources().getConfiguration().orientation != Configuration.ORIENTATION_PORTRAIT) {
	        		Mat tmpRgba = new Mat(); 
	            	Core.transpose(mRgba, tmpRgba); 
	            	Core.flip(tmpRgba, mRgba, 0);
	            	
	            	tmpRgba.release();
	        	}
		        
	        	Core.rectangle(mRgba, rectFaceClone.tl(), rectFaceClone.br(), FACE_RECT_COLOR, 3);
	        	
	        	//rotate back image for portrait orientation after rect has been drawed
		        if (getResources().getConfiguration().orientation != Configuration.ORIENTATION_PORTRAIT) {
	        		Mat tmpRgba = new Mat(); 
	            	Core.transpose(mRgba, tmpRgba); 
	            	Core.flip(tmpRgba, mRgba, 1);
	            	
	            	tmpRgba.release();
	        	}
		    }
		    
		    if (rectFaceClone != null) {
		    	//detect palm & fist
		    	rectPalm = rectFaceClone.clone();
		    	rectPalm.height /= SCALE;
		    	rectPalm.width /= SCALE;
		    	rectPalm.x /= SCALE;
		    	rectPalm.y /= SCALE;
		    	
		    	if (rectPalm.x >= 0 && rectPalm.y >= 0) {
			    	try {
				       	Mat mGraySubmat = mGray.submat(rectPalm);
				       	Imgproc.resize(mGraySubmat, mGraySubmat, new Size(mGraySubmat.width() * SCALE_UP, mGraySubmat.height() * SCALE_UP));
				       	
				       	//rotate image for portrait orientation
				        //if (getResources().getConfiguration().orientation != Configuration.ORIENTATION_PORTRAIT) {
			        	//	Mat tmpTranspose = new Mat(); 
			            //	Core.transpose(mGraySubmat, tmpTranspose); 
			            //	Core.flip(tmpTranspose, mGraySubmat, 0);
			        	//}
				       	
				       	if (mDetectorType == JAVA_DETECTOR) {
					        if (mJavaPalmDetector != null) {
					        	mJavaPalmDetector.detectMultiScale(mGraySubmat, matRectOfPalm, 1.1, 0, 1, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
					                    new Size(0, 0), new Size());
					        }
					        if (mJavaFistDetector != null) {
					        	mJavaFistDetector.detectMultiScale(mGraySubmat, matRectOfFist, 1.1, 0, 1, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
					                    new Size(0, 0), new Size());
					        }
					    }
					    else if (mDetectorType == NATIVE_DETECTOR) {
					        if (mNativePalmDetector != null) {
					        	mNativePalmDetector.detect(mGraySubmat, matRectOfPalm);
					        }
					        if (mNativeFistDetector != null) {
					        	mNativeFistDetector.detect(mGraySubmat, matRectOfFist);
					        }
					    }
					    else {
					        Log.e(TAG, "Detection method is not selected!");
					    }
				       	
				       	mGraySubmat.release();
				       	
				       	Rect[] palmArray = matRectOfPalm.toArray();
					    if (palmArray != null && palmArray.length > 0) {
					    	isPalmDetected = true;
					    	holdFist = 0;
					    	
					    	Rect rectPalmClone = rectPalm.clone();
					    	rectPalmClone.height *= SCALE;
					    	rectPalmClone.width *= SCALE;
					    	rectPalmClone.x *= SCALE;
					    	rectPalmClone.y *= SCALE;
					    	
					    	//rotate image for portrait orientation before draw rect
					        if (getResources().getConfiguration().orientation != Configuration.ORIENTATION_PORTRAIT) {
				        		Mat tmpRgba = new Mat(); 
				            	Core.transpose(mRgba, tmpRgba); 
				            	Core.flip(tmpRgba, mRgba, 0);
				            	
				            	tmpRgba.release();
				        	}
					        
					        Core.rectangle(mRgba, rectPalmClone.br(), rectPalmClone.tl(), PALM_RECT_COLOR, 5);
				        	
				        	//rotate back image for portrait orientation after rect has been drawed
					        if (getResources().getConfiguration().orientation != Configuration.ORIENTATION_PORTRAIT) {
				        		Mat tmpRgba = new Mat(); 
				            	Core.transpose(mRgba, tmpRgba); 
				            	Core.flip(tmpRgba, mRgba, 1);
				            	
				            	tmpRgba.release();
				        	}
					    }
					    
					    Rect[] fistArray = matRectOfFist.toArray();
					    if (fistArray != null && fistArray.length > 0 && isPalmDetected) {
					    	holdFist++;
					    	if (holdFist > 3) {
					    		Rect rectFistClone = rectPalm.clone();
						    	rectFistClone.height *= SCALE;
						    	rectFistClone.width *= SCALE;
						    	rectFistClone.x *= SCALE;
						    	rectFistClone.y *= SCALE;
						    	
						    	//rotate image for portrait orientation before draw rect
						        if (getResources().getConfiguration().orientation != Configuration.ORIENTATION_PORTRAIT) {
					        		Mat tmpRgba = new Mat(); 
					            	Core.transpose(mRgba, tmpRgba); 
					            	Core.flip(tmpRgba, mRgba, 0);
					            	
					            	tmpRgba.release();
					        	}
						        
						        Core.rectangle(mRgba, rectFistClone.br(), rectFistClone.tl(), FIST_RECT_COLOR, 5);
					        	
					        	//rotate back image for portrait orientation after rect has been drawed
						        if (getResources().getConfiguration().orientation != Configuration.ORIENTATION_PORTRAIT) {
					        		Mat tmpRgba = new Mat(); 
					            	Core.transpose(mRgba, tmpRgba); 
					            	Core.flip(tmpRgba, mRgba, 1);
					            	
					            	tmpRgba.release();
					        	}
						    	
						    	if (countDownTimer == null) {
						    		isStartTakingPicture = true;
									isTakePicture = false;
									
						    		runOnUiThread(new Runnable() {
										@Override
										public void run() {
											if (textView != null) {
								        		textView.setVisibility(View.VISIBLE);
								        		textView.setText(String.valueOf((CAMERA_TRIGGER_MAX_MILLIS / CAMERA_TRIGGER_INTERVAL_MILLIS)));
								        	}
											//Toast.makeText(getApplicationContext(), "Taking Picture in " + (CAMERA_TRIGGER_MAX_MILLIS / CAMERA_TRIGGER_INTERVAL_MILLIS)  + " seconds..", Toast.LENGTH_SHORT).show();
											
											countDownTimer = new MyCounter(CAMERA_TRIGGER_MAX_MILLIS, CAMERA_TRIGGER_INTERVAL_MILLIS);
											countDownTimer.start();
										}
									});
						    	}
					    	}
					    }
			    	} catch (Exception e) {
			    	}
		    	}
		    }
		    
		    
		    mGray.release();
        }
	    
        //mirror front camera
        Core.flip(mRgba, mRgba, 1);
        
        if (isTakePicture) {
        	runOnUiThread(new Runnable() {
				@Override
				public void run() {
					saveImage(mRgba);
				}
			});
        }

        return mRgba;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemType   = menu.add(mDetectorName[mDetectorType]);
        
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemType) {
            mDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[mDetectorType]);
            setDetectorType(mDetectorType);
        } 
        return true;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                Log.i(TAG, "Detection Based Tracker enabled");
                mNativeFistDetector.start();
            } else {
                Log.i(TAG, "Cascade detector enabled");
                mNativeFistDetector.stop();
            }
        }
    }
    
    private void startResultActivity(String fileName) {
    	Intent intent = new Intent(this, ShowResultActivity.class);
		intent.putExtra("FILENAME", fileName);
		this.startActivity(intent);
    }
    
    class MyCounter extends CountDownTimer{
    	 
        public MyCounter(long millisInFuture, long countDownInterval) {
            super(millisInFuture, countDownInterval);
        }
 
        @Override
        public void onFinish() {
        	if (textView != null) {
        		textView.setVisibility(View.INVISIBLE);
        	}
        	isTakePicture = true;
        }
 
        @Override
        public void onTick(long millisUntilFinished) {
        	textView.setText(String.valueOf((int)(millisUntilFinished / 1000)));
        }
    }
    
    private void saveImage(Mat rbga) {
    	isTakePicture = false;
    	if (rbga != null) {
    		Bitmap bmpResult = Bitmap.createBitmap(rbga.cols(), rbga.rows(), Bitmap.Config.ARGB_8888);
    		Utils.matToBitmap(rbga, bmpResult);
    		
    		String path = Environment.getExternalStorageDirectory() + "/opencv";
    		String fileName = "result.png";
    		
    		Toast.makeText(getApplicationContext(), "Picture Taken!!", Toast.LENGTH_SHORT).show();
    		
    		//Highgui.imwrite(fileName, rbga);
    		File filePath = new File(path);
    		
    		if (!filePath.exists()) {
    			filePath.mkdir();
    		}
    		
    		File file = new File(path, fileName);
			try {
				FileOutputStream fOut = new FileOutputStream(file);
				bmpResult.compress(Bitmap.CompressFormat.PNG, 90, fOut);
        	    fOut.flush();
        	    fOut.close();
			} catch (FileNotFoundException e) {
			} catch (IOException e) {
			}
			
            reset();
    		
    		startResultActivity(path + "/" + fileName);
    	}
    }
    
    private void reset() {
    	isStartTakingPicture = false;
    	isTakePicture = false;
    	isPalmDetected = false;
    	holdFist = 0;
    	timerSecond = 0;
    	if (countDownTimer != null) {
    		countDownTimer.cancel();
    		countDownTimer = null;
    	}
    	
    	if (textView != null) {
    		textView.setText("");
    	}
    }
}
