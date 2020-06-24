
#include "ObjectSegmentation.hpp"

#ifdef __linux__
#include <opencv2/features2d/features2d.hpp>
// using namespace std;
// #include <opencv2/gpu/gpu.hpp>
// #include <opencv2/gpu/gpumat.hpp>

#else
#include <opencv2/features2d/features2d.hpp>
#endif

//#define VIS


namespace ImgPipe {
	namespace FilterProcessingPipe {



		// -------------------------------------------------------------------------------------------------

		ObjectSegmentation::ObjectSegmentation() : m_progress(0.0) {

			std::shared_ptr<ImageFilter::ParameterDescription> pParameterDescription
				= std::make_shared<ImageFilter::ParameterDescription>();
			pParameterDescription->parameterName = "InputImageName";
			pParameterDescription->parameterType = "string";
			m_pParametersDescription.push_back(pParameterDescription);

			std::shared_ptr<ImageFilter::ParameterDescription> pParameterDescription1
				= std::make_shared<ImageFilter::ParameterDescription>();
			pParameterDescription1->parameterName = "InputMaskName";
			pParameterDescription1->parameterType = "string";
			m_pParametersDescription.push_back(pParameterDescription1);

			std::shared_ptr<ImageFilter::ParameterDescription> pParameterDescription2
				= std::make_shared<ImageFilter::ParameterDescription>();
			pParameterDescription2->parameterName = "InputMaskCornerName";
			pParameterDescription2->parameterType = "string";
			m_pParametersDescription.push_back(pParameterDescription2);

			std::shared_ptr<ImageFilter::ParameterDescription> pParameterDescription3
				= std::make_shared<ImageFilter::ParameterDescription>();
			pParameterDescription3->parameterName = "OutputMasksName";
			pParameterDescription3->parameterType = "string";
			m_pParametersDescription.push_back(pParameterDescription3);

			m_rng(12345);
			m_counter = 0;

		}


		// -------------------------------------------------------------------------------------------------

		std::string ObjectSegmentation::filterName() {
			return "ObjectSegmentation";
		}

		// -------------------------------------------------------------------------------------------------

		double ObjectSegmentation::progress() {
			return m_progress;
		}

		// -------------------------------------------------------------------------------------------------

		std::vector<std::shared_ptr<ImageFilter::ParameterDescription> >
			ObjectSegmentation::parameterDescriptions() {

			return m_pParametersDescription;
		}

		// -------------------------------------------------------------------------------------------------

		std::vector<boost::any> ObjectSegmentation::parameterValues() {
			std::vector<boost::any> vector;
			vector.push_back(m_inputImageName);
			vector.push_back(m_inputMaskName);
			vector.push_back(m_inputMaskCornerName);
			vector.push_back(m_outputMasksName);
			return vector;
		}

		// -------------------------------------------------------------------------------------------------

		void ObjectSegmentation::setParameter(const std::vector<boost::any>& newValues) {
			try {
				m_inputImageName = boost::any_cast<std::string>(newValues[0]);
				m_inputMaskName = boost::any_cast<std::string>(newValues[1]);
				m_inputMaskCornerName = boost::any_cast<std::string>(newValues[2]);
				m_outputMasksName = boost::any_cast<std::string>(newValues[3]);
			}
			catch (const boost::bad_any_cast &e) {}
		}

		// -------------------------------------------------------------------------------------------------

		std::vector<std::string> ObjectSegmentation::inputDataNames() {
			boost::mutex::scoped_lock l(m_mutex);
			return m_inputDataNames;
		}

		// -------------------------------------------------------------------------------------------------

		int ObjectSegmentation::applyFilter(std::shared_ptr<ImageFilterData>& pData) {

			int isFilterSuccess = 0;
			std::vector<std::string> inputDataNames;

			// assign input vector to output vector
			std::vector<std::pair<std::string, boost::any> > outputVector = pData->second;

			std::shared_ptr<cv::Mat> pImage;
			std::shared_ptr<cv::Mat> pMask;
			std::shared_ptr<cv::Mat> pSegmentation;
			std::shared_ptr<std::vector<cv::Point>> pMaskCorners;
			for (size_t i = 0; i < pData->second.size(); ++i) {

				// get input data names
				inputDataNames.push_back(pData->second[i].first);

				// cast data
				std::shared_ptr<cv::Mat> pImageTmp
					= castInput<std::shared_ptr<cv::Mat> >(pData->second[i], m_inputImageName);
				if (pImageTmp) pImage = pImageTmp;

				std::shared_ptr<cv::Mat> pMaskTmp
					= castInput<std::shared_ptr<cv::Mat> >(pData->second[i], m_inputMaskName);
				if (pMaskTmp) pMask = pMaskTmp;

				std::shared_ptr<std::vector<cv::Point>> pMaskCornersTmp
					= castInput<std::shared_ptr<std::vector<cv::Point>> >(pData->second[i], m_inputMaskCornerName);
				if (pMaskCornersTmp) pMaskCorners = pMaskCornersTmp;

			}

			if (!pImage) return -1;
			if (pImage->channels() == 0 || pImage->rows == 0 || pImage->cols == 0) return -1;
			if (!pMaskCorners || pMaskCorners->size() == 0) return -1;


			// Load the image
			cv::Mat inputImage = pImage->clone();
			cv::Mat src;

			if (pMask && pMask->channels() > 0 && pMask->rows > 0 && pMask->cols > 0) {

				cv::Mat maskedImage(inputImage.size(), CV_8UC3, cv::Scalar(255, 255, 255));
				inputImage.copyTo(maskedImage, *pMask);
				src = maskedImage.clone();
			}
			else {
				src = inputImage.clone();
			}

			// Check if everything was fine
			if (!src.data) return -1;

#ifdef VIS
			cv::imshow("objectseg_mask", *pMask);
			cv::imshow("maskedImage", src);
			cv::waitKey(1);
			// 		if(pMaskCorners)
			// 		  std::cout << "pMaskCorners "<< *pMaskCorners << std::endl;
#endif

			cv::Point2f seedPointBackground;
			if (pMask) {

				if (pMaskCorners && pMaskCorners->size() > 0) {

					// 			std::cout << "pMaskCorners->size() "<<pMaskCorners->size() << std::endl;
					if (pMaskCorners->size() > 3) {
						cv::Point2f corner0 = pMaskCorners->at(0);
						cv::Point2f corner2 = pMaskCorners->at(2);
						cv::Point2f direction = (corner2 - corner0) / cv::norm(corner2 - corner0);
						seedPointBackground = corner0 + direction*(cv::norm(corner2 - corner0) / 15);
					}
					else if (pMaskCorners->size() == 3) {
						cv::Point2f corner0 = pMaskCorners->at(0);
						cv::Point2f corner1 = pMaskCorners->at(1);
						cv::Point2f corner2 = pMaskCorners->at(2);
						cv::Point2f pointBetween1And2 = (corner1 + corner2) / 2;
						cv::Point2f direction = (pointBetween1And2 - corner0) / cv::norm(pointBetween1And2 - corner0);
						seedPointBackground = corner0 + direction*(cv::norm(pointBetween1And2 - corner0) / 15);
					}
					else {
						seedPointBackground.x = 5;
						seedPointBackground.y = 5;
					}
				}

				cv::circle(*pImage, seedPointBackground, 3, CV_RGB(255, 255, 255), -1);
#ifdef VIS
				cv::imshow("image seed ", *pImage);
#endif
			}
			else return -1;

#ifdef VIS
			// Show source image
			cv::imshow("Source Image", src);
			cv::waitKey(1);
#endif

			// 		Change the background from white to black, since that will help later to extract
			// 		better results during the use of Distance Transform
			for (int x = 0; x < src.rows; x++) {
				for (int y = 0; y < src.cols; y++) {
					if (src.at<cv::Vec3b>(x, y)[0] >= 220 && src.at<cv::Vec3b>(x, y)[1] >= 235 && src.at<cv::Vec3b>(x, y)[2] >= 220)
					{
						src.at<cv::Vec3b>(x, y)[0] = 0;
						src.at<cv::Vec3b>(x, y)[1] = 0;
						src.at<cv::Vec3b>(x, y)[2] = 0;

					}
				}
			}
			// 		cv::Mat gray;
			// 		cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY, 1);
			// 		cv::Mat thresholded;
			// 		cv::adaptiveThreshold(gray, thresholded, 255,  cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 7, 8);
			// 		cv::imshow("adaptiveThreshold", thresholded);
			// 		
			// 		cv::Mat bw_inv;
			// 		cv::bitwise_not(thresholded, bw_inv);
			// 		cv::imshow("before closing", bw_inv);
			// 		
			// 		cv::Mat kernelMorph = cv::Mat::ones(4, 4, CV_8UC1);
			// 		cv::morphologyEx(bw_inv, bw_inv, cv::MORPH_CLOSE, kernelMorph);
			// 		cv::imshow("after closing", bw_inv);
			// 		
			// 		cv::floodFill(bw_inv, cv::Point(0,0), cv::Scalar(255));
			// 		
			// 		cv::imshow("after floodfill", bw_inv);
			// 		
			// 		cv::Mat cannyThresh;
			// 		cv::Canny(bw_inv, cannyThresh, 200, 230);
			// 		cv::imshow("Edges", cannyThresh);
			// 
			// 		std::vector<std::vector<cv::Point>> contours1;
			// 		
			// 		cv::findContours( cannyThresh, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
			// 		cv::Mat drawing = cv::Mat::zeros(cannyThresh.size(), CV_8UC3);
			// 		
			// 		for (int i = 0; i< contours1.size(); i++)
			// 		{
			// 		    if(contours1[i].size() >= 8)
			// 			cv::drawContours( drawing, contours1, i, cv::Scalar( 255,255,255), 1 );
			// 		}
			// 		cv::imshow("drawing detected contours", drawing);
			// 		
			// 		std::vector<cv::Point> convexHullPoints;
			// 		std::vector<cv::Point> pts;
			// 		for ( size_t i = 0; i< contours1.size(); i++)
			// 		    for ( size_t j = 0; j< contours1[i].size(); j++)
			// 			pts.push_back(contours1[i][j]);
			// 		cv::convexHull( pts, convexHullPoints );
			// 		
			// 		cv::polylines( drawing, convexHullPoints, true, cv::Scalar(0,0,255), 2 );
			// 		cv::imshow("drawing hulls", drawing);


			// 		// sharpen image using "unsharp mask" algorithm
			// 		cv::Mat img = src.clone();
			// 		cv::Mat blurred; double sigma = 1, threshold = 5, amount = ;
			// 		cv::GaussianBlur(img, blurred, cv::Size(), sigma, sigma);
			// 		cv::Mat lowContrastMask = abs(img - blurred) < threshold;
			// 		cv::Mat sharpened = img*(1+amount) + blurred*(-amount);
			// 		img.copyTo(sharpened, lowContrastMask);
			// 		cv::imshow( "sharpened", sharpened );


					// Create a kernel that we will use for accuting/sharpening our image
			cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
				1, 1, 1,
				1, -8, 1,
				1, 1, 1); // an approximation of second derivative, a quite strong kernel
			// do the laplacian filtering as it is
			// well, we need to convert everything in something more deeper then CV_8U
			// because the kernel has some negative values,
			// and we can expect in general to have a Laplacian image with negative values
			// BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
			// so the possible negative number will be truncated
			cv::Mat imgLaplacian;
			cv::Mat sharp = src; // copy source image to another temporary one
			cv::filter2D(sharp, imgLaplacian, CV_32F, kernel);
			src.convertTo(sharp, CV_32F);
			cv::Mat imgResult = sharp - imgLaplacian;
			// convert back to 8bits gray scale
			imgResult.convertTo(imgResult, CV_8UC3);
			imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
			// imshow( "Laplace Filtered Image", imgLaplacian );
			cv::imshow("New Sharped Image", imgResult);
			src = imgResult; // copy back




			// Create binary image from source image
			cv::Mat bw;
			cv::cvtColor(src, bw, cv::COLOR_BGR2GRAY, 0);
			//cv::adaptiveThreshold(bw, bw, 200, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 15, 2);
			//cv::threshold(bw, bw, 40, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
#ifdef VIS
			cv::imshow("Binary Image", bw);
			cv::waitKey(1);
#endif

			// Perform the distance transform algorithm
			cv::Mat dist;
			cv::distanceTransform(bw, dist, cv::DIST_L2, 3);
			// Normalize the distance image for range = {0.0, 1.0}
			// so we can visualize and threshold it
			cv::normalize(dist, dist, 0.0, 1.0, cv::NORM_MINMAX);
#ifdef VIS
			cv::imshow("Distance Transform Image", dist);
#endif
			// Threshold to obtain the peaks
			// This will be the markers for the foreground objects
			cv::threshold(dist, dist, 0.32, 1.0, cv::THRESH_BINARY);
			// Dilate a bit the dist image
			cv::Mat kernel1 = cv::Mat::ones(6, 6, CV_8UC1);
			cv::dilate(dist, dist, kernel1);
			// 		cv::Mat kernel2 = cv::Mat::ones(4, 4, CV_8UC1);
			// 		cv::erode(dist, dist, kernel2);
#ifdef VIS
			cv::imshow("Peaks", dist);
#endif

			// Create the CV_8U version of the distance image
			// It is needed for findContours()
			cv::Mat dist_8u;
			dist.convertTo(dist_8u, CV_8U);

			// Find total markers
			std::vector<std::vector<cv::Point> > contours;
			cv::findContours(dist_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

			// Create the marker image for the watershed algorithm
			cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32SC1);

			// Draw the foreground markers
			for (size_t i = 0; i < contours.size(); i++)
				cv::drawContours(markers, contours, static_cast<int>(i), cv::Scalar::all(static_cast<int>(i) + 1), -1);

			// Draw the background marker
			cv::circle(markers, /*cv::Point(5,5)*/seedPointBackground, 3, CV_RGB(255, 255, 255), -1);
#ifdef VIS
			cv::imshow("Markers", markers * 10000);
#endif

			cv::Mat srcCopy = src.clone();
			// Perform the watershed algorithm
			cv::watershed(srcCopy, markers);
			cv::Mat mark = cv::Mat::zeros(markers.size(), CV_8UC1);
			markers.convertTo(mark, CV_8UC1);
			bitwise_not(mark, mark);

#ifdef VIS
			cv::imshow("Markers_v2", mark); // uncomment this if you want to see how the mark image looks like at that point
#endif

		// Generate random colors
			std::vector<cv::Vec3b> colors;
			for (size_t i = 0; i < contours.size(); i++)
			{
				int b = cv::theRNG().uniform(0, 255);
				int g = cv::theRNG().uniform(0, 255);
				int r = cv::theRNG().uniform(0, 255);
				colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
			}
			// Create the result image
			cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);

			// 		cv::Mat labels; 
			// 		cv::Mat centroids;
			// 		cv::Mat stats;
			// 		cv::connectedComponentsWithStats(markers, labels, stats, centroids, 8, CV_32S);

			// 		cv::Mat allMaskImage = cv::Mat::zeros(markers.size(), CV_8UC1);
			// 		if(m_allMaskSummed.rows == 0 && m_allMaskSummed.cols == 0) {
			// 		    m_allMaskSummed = cv::Mat::zeros(markers.size(), CV_8UC1);
			// 		}
			// 		if(m_allMaskImageThresholded.rows == 0 && m_allMaskImageThresholded.cols == 0)
			// 			m_allMaskImageThresholded = cv::Mat::zeros(markers.size(), CV_8UC1);

			m_counter++;
			std::map<int, int> mapForIndex;
			// Fill labeled objects with random colors
			for (int i = 0; i < markers.rows; i++)
			{

				for (int j = 0; j < markers.cols; j++)
				{
					int index = markers.at<int>(i, j);
					if (index > 0 && index <= static_cast<int>(contours.size())) {
						dst.at<cv::Vec3b>(i, j) = colors[index - 1];
						mapForIndex.insert(std::make_pair(index, index));
						// 			    m_allMaskSummed.at<unsigned char>(i,j) = m_allMaskSummed.at<unsigned char>(i,j) + 1;
						// 			    m_allMaskImageThresholded.at<unsigned char>(i,j) = m_allMaskSummed.at<unsigned char>(i,j) + 1;
						// 
						// 		    
						// 			    if(m_allMaskImageThresholded.at<unsigned char>(i,j) > m_counter*0.5) 
						// 				  m_allMaskImageThresholded.at<unsigned char>(i,j) = 1;
						// 			    else 
						// 			      m_allMaskImageThresholded.at<unsigned char>(i,j) = 0;

					}
					else {
						dst.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
						//mask.at<unsigned char>(i,j) = 0;

					}
				}
				//masks.push_back(mask);
			}
			// 		if(m_allMaskSummed.rows > 0 && m_allMaskSummed.cols > 0)
			// 		    cv::imshow("allMaskImage", m_allMaskSummed);
			// 		
			// 	      cv::threshold(m_allMaskImageThresholded, m_allMaskImageThresholded, m_counter*2, 0, cv::THRESH_BINARY);

			// 		std::vector<std::vector<cv::Point> > contoursfinal;		
			// 		cv::findContours(m_allMaskImageThresholded, contoursfinal, cv::RETR_EXTERNAL/*RETR_EXTERNAL*/, cv::CHAIN_APPROX_SIMPLE);
			// 		for (size_t i = 0; i < contoursfinal.size(); i++)
			// 		    cv::drawContours(m_allMaskImageThresholded, contoursfinal, static_cast<int>(i), cv::Scalar::all(static_cast<int>(i)+1), -1);

			// 		cv::imshow("allMaskImageThresholded", m_allMaskImageThresholded);


					// init mask vector
			std::vector<cv::Mat> masks; // vector with one mask for each object
			for (int i = 0; i < mapForIndex.size(); i++) {
				cv::Mat mask = cv::Mat::zeros(markers.size(), CV_8UC1);
				masks.push_back(mask);
			}

			// set each region in individual mask 
			for (int i = 0; i < markers.rows; i++)
			{

				for (int j = 0; j < markers.cols; j++)
				{
					int index = markers.at<int>(i, j);
					if (index > 0 && index <= static_cast<int>(contours.size())) {
						masks[index - 1].at<unsigned char>(i, j) = 255;
					}
				}
			}


			// 		cv::Mat dst_gray;
			// 		cv::cvtColor(dst, dst_gray, cv::COLOR_RGB2GRAY);
			// // 		cv::Mat dstCopy;
			// // 		dst_gray.convertTo(dstCopy, CV_8U);
			// 		cv::imshow("dst_gray", dst_gray);
			// 
			// 		std::vector<std::vector<cv::Point> > contoursfinal;		
			// 		cv::findContours(dst_gray, contoursfinal, cv::RETR_EXTERNAL/*RETR_EXTERNAL*/, cv::CHAIN_APPROX_SIMPLE);
			// 		for (size_t i = 0; i < contoursfinal.size(); i++)
			// 		    cv::drawContours(dst_gray, contoursfinal, static_cast<int>(i), cv::Scalar::all(static_cast<int>(i)+1), -1);
			// 		
			// 		cv::imshow("iamgecontourFinal", dst_gray);



			//#ifdef VIS
					// Visualize the final image
			cv::imshow("Final Result ObjectSegmentation", dst);
			cv::waitKey(1);
			//#endif

					// visualize masks
			// 						for (int j = 0; j < masks.size(); j++){
			// 						    cv::imshow("mask", masks[j]);
			// 						    cv::waitKey(500);
			// 						}


			pData->second.push_back(std::make_pair(m_outputMasksName,
				std::make_shared<std::vector<cv::Mat> >(masks)));

			m_inputDataNames = inputDataNames;
			return isFilterSuccess;
		}

		// -------------------------------------------------------------------------------------------------

	} // namespace 

} // namespace