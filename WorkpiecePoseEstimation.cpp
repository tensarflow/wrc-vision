/*
  HEADER_DUMMY
*/
#include "WorkpiecePoseEstimation.hpp"
#include "../include/ObjectStruct.hpp"

#ifdef __linux__
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
// using namespace std;
// #include <opencv2/gpu/gpu.hpp>
// #include <opencv2/gpu/gpumat.hpp>
#include <string>

#else
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core/eigen.hpp>

#endif

#define DEBUG
#define INFO

namespace ImgPipe {
	namespace FilterProcessingPipe {

		// -------------------------------------------------------------------------------------------------

		WorkpiecePoseEstimation::WorkpiecePoseEstimation() : m_progress(0.0) {

			std::shared_ptr<ImageFilter::ParameterDescription> pParameterDescription
				= std::make_shared<ImageFilter::ParameterDescription>();
			pParameterDescription->parameterName = "InputImageName";
			pParameterDescription->parameterType = "string";
			m_pParametersDescription.push_back(pParameterDescription);

			std::shared_ptr<ImageFilter::ParameterDescription> pParameterDescription0
				= std::make_shared<ImageFilter::ParameterDescription>();
			pParameterDescription0->parameterName = "InputMaskName";
			pParameterDescription0->parameterType = "string";
			m_pParametersDescription.push_back(pParameterDescription0);

			std::shared_ptr<ImageFilter::ParameterDescription> pParameterDescription1
				= std::make_shared<ImageFilter::ParameterDescription>();
			pParameterDescription1->parameterName = "InputMasksName";
			pParameterDescription1->parameterType = "string";
			m_pParametersDescription.push_back(pParameterDescription1);

			std::shared_ptr<ImageFilter::ParameterDescription> pParameterDescription2
				= std::make_shared<ImageFilter::ParameterDescription>();
			pParameterDescription2->parameterName = "ArucoSizeMM";
			pParameterDescription2->parameterType = "int";
			m_pParametersDescription.push_back(pParameterDescription2);

			std::shared_ptr<ImageFilter::ParameterDescription> pParameterDescription3
				= std::make_shared<ImageFilter::ParameterDescription>();
			pParameterDescription3->parameterName = "ArucoReferenceID";
			pParameterDescription3->parameterType = "int";
			m_pParametersDescription.push_back(pParameterDescription3);

			std::shared_ptr<ImageFilter::ParameterDescription> pParameterDescription4
				= std::make_shared<ImageFilter::ParameterDescription>();
			pParameterDescription4->parameterName = "OutputObjectDataName";
			pParameterDescription4->parameterType = "string";
			m_pParametersDescription.push_back(pParameterDescription4);
			
			std::shared_ptr<ImageFilter::ParameterDescription> pParameterDescription5
				= std::make_shared<ImageFilter::ParameterDescription>();
			pParameterDescription5->parameterName = "InputDistortionCoefficientsName";
			pParameterDescription5->parameterType = "string";
			m_pParametersDescription.push_back(pParameterDescription5);
			
			std::shared_ptr<ImageFilter::ParameterDescription> pParameterDescription6
				= std::make_shared<ImageFilter::ParameterDescription>();
			pParameterDescription6->parameterName = "InputCalibrationMatrixName";
			pParameterDescription6->parameterType = "string";
			m_pParametersDescription.push_back(pParameterDescription6);
			
			
		}


		// -------------------------------------------------------------------------------------------------

		std::string WorkpiecePoseEstimation::filterName() {
			return "WorkpiecePoseEstimation";
		}

		// -------------------------------------------------------------------------------------------------

		double WorkpiecePoseEstimation::progress() {
			return m_progress;
		}

		// -------------------------------------------------------------------------------------------------

		std::vector<std::shared_ptr<ImageFilter::ParameterDescription> >
			WorkpiecePoseEstimation::parameterDescriptions() {

			return m_pParametersDescription;
		}

		// -------------------------------------------------------------------------------------------------

		std::vector<boost::any> WorkpiecePoseEstimation::parameterValues() {
			std::vector<boost::any> vector;
			vector.push_back(m_inputImageName);
			vector.push_back(m_inputMaskName);
			vector.push_back(m_inputMasksName);
			vector.push_back(m_arucoSizeMM);
			vector.push_back(m_arucoReferenceID);
			vector.push_back(m_outputObjectDataName);
			vector.push_back(m_inputDistortionCoefficientsName);
			vector.push_back(m_inputCalibrationMatrixName);
			return vector;
		}

		// -------------------------------------------------------------------------------------------------

		void WorkpiecePoseEstimation::setParameter(const std::vector<boost::any>& newValues) {
			try {
				m_inputImageName = boost::any_cast<std::string>(newValues[0]);
				m_inputMaskName = boost::any_cast<std::string>(newValues[1]);
				m_inputMasksName = boost::any_cast<std::string>(newValues[2]);
				m_arucoSizeMM = boost::any_cast<int>(newValues[3]);
				m_arucoReferenceID = boost::any_cast<int>(newValues[4]);
				m_outputObjectDataName = boost::any_cast<std::string>(newValues[5]);
				m_inputDistortionCoefficientsName = boost::any_cast<std::string>(newValues[6]);
				m_inputCalibrationMatrixName = boost::any_cast<std::string>(newValues[7]);
			}
			catch (const boost::bad_any_cast &e) {}
		}

		// -------------------------------------------------------------------------------------------------

		std::vector<std::string> WorkpiecePoseEstimation::inputDataNames() {
			boost::mutex::scoped_lock l(m_mutex);
			return m_inputDataNames;
		}

		// -------------------------------------------------------------------------------------------------

		void WorkpiecePoseEstimation::imshowScaled2(std::string label, cv::Mat image){
			cv::Mat smallImage;
			cv::Size imageSizeHalf(image.cols/2, image.rows/2);
			cv::resize(image, smallImage, imageSizeHalf);
			cv::imshow(label, smallImage);
		}
		
		int WorkpiecePoseEstimation::applyFilter(std::shared_ptr<ImageFilterData>& pData) {

			int isFilterSuccess = 0;
			std::vector<std::string> inputDataNames;

			// Cast the PointClouds
			std::shared_ptr<cv::Mat> pInputImage;
			std::shared_ptr<cv::Mat> pInputMask; // this is the mask defined by the aruco markers
			std::shared_ptr<std::vector<cv::Mat> > pInputMasks; // these are the masks of every single object detected in ObjectSegmentation
			std::shared_ptr<Eigen::VectorXd> pInputDistortionCoefficients;
			std::shared_ptr<Eigen::Matrix3d> pInputCalibrationMatrix;

			// Get data from pData
			for (size_t i = 0; i < pData->second.size(); ++i)
			{
				std::shared_ptr<cv::Mat> pInputImageTmp
					= castInput<std::shared_ptr<cv::Mat> >(
						pData->second[i], m_inputImageName);
				if (pInputImageTmp) {
					pInputImage = pInputImageTmp;
				}

				std::shared_ptr<cv::Mat> pInputMaskTmp
					= castInput<std::shared_ptr<cv::Mat> >(
						pData->second[i], m_inputMaskName);
				if (pInputMaskTmp) {
					pInputMask = pInputMaskTmp;
				}

				std::shared_ptr<std::vector<cv::Mat> > pInputMasksTmp
					= castInput<std::shared_ptr<std::vector<cv::Mat> > >(
						pData->second[i], m_inputMasksName);
				if (pInputMasksTmp) {
					pInputMasks = pInputMasksTmp;
				}
				
				std::shared_ptr<Eigen::VectorXd> pInputDistortionCoefficientsTmp
					= castInput<std::shared_ptr<Eigen::VectorXd> >(
						pData->second[i], m_inputDistortionCoefficientsName);
				if (pInputDistortionCoefficientsTmp) {
					pInputDistortionCoefficients = pInputDistortionCoefficientsTmp;
				}
				
				std::shared_ptr<Eigen::Matrix3d> pInputCalibrationMatrixTmp
					= castInput<std::shared_ptr<Eigen::Matrix3d> >(
						pData->second[i], m_inputCalibrationMatrixName);
				if (pInputCalibrationMatrixTmp) {
					pInputCalibrationMatrix = pInputCalibrationMatrixTmp;
				}

				// get input data names (for GUI)
				inputDataNames.push_back(pData->second[i].first);
			}

			if (!pInputImage) return -1;
			if (!pInputMasks) return -1;
			if (pInputMasks->size() == 0) return -1;
			if (pInputImage->channels() < 3) return -1;
			if (pInputImage->rows == 0 || pInputImage->cols == 0) return -1;
			if (pInputDistortionCoefficients == NULL){ std::cout << "Couldn't read Distortion Coeffs!" << std::endl; return -1;}
			if (pInputCalibrationMatrix == NULL){ std::cout << "Couldn't read Calibration Coeffs!" << std::endl; return -1;}
			
			// Load the image
			cv::Mat srcImg = pInputImage->clone();
			cv::Mat src;
			if (pInputMask && pInputMask->channels() > 0 && pInputMask->rows > 0 && pInputMask->cols > 0) {

				cv::Mat maskedImage(pInputImage->size(), CV_8UC3, cv::Scalar(255, 255, 255));
				pInputImage->copyTo(maskedImage, *pInputMask);
				src = maskedImage;
#ifdef DEBUG
				imshowScaled2("maskedImage", maskedImage);
				cv::waitKey(1);
#endif
			}
			else {
				src = *pInputImage;
			}

			int markersX = 12;
			int markersY = 12;
			float markerLength = 0.035;
			float markerSeparation = 0.006;

			// Detect Aruco poses
			std::vector<int> markerIds;
			std::vector<std::vector<cv::Point2f>> markerCorners;
			cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_1000);
			cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(markersX, markersY, markerLength, markerSeparation, dictionary);
			cv::aruco::detectMarkers(*pInputImage, board->dictionary, markerCorners, markerIds);

			cv::Mat arucoImage = pInputImage->clone();
			
			

			// Generate camera parameters
			cv::Mat cameraMatrix;
			cv::Mat distCoeffs;
			cv::eigen2cv(*pInputCalibrationMatrix, cameraMatrix);
			cv::eigen2cv(*pInputDistortionCoefficients, distCoeffs);
			
			cv::Vec3d rvec, tvec;
			if (markerIds.size() > 0) {
				cv::aruco::drawDetectedMarkers(arucoImage, markerCorners, markerIds);

				int valid = cv::aruco::estimatePoseBoard(markerCorners, markerIds, board, cameraMatrix, distCoeffs, rvec, tvec);

				if (valid)
					cv::aruco::drawAxis(arucoImage, cameraMatrix, distCoeffs, rvec, tvec, 0.1);

#ifdef DEBUG
				imshowScaled2("Arucoboard Pose Estimation", arucoImage);
				cv::waitKey(1);
#endif

			}
			else {
				std::cout << "No IDs detected!" << std::endl;
			}

			// get referenceMarker data
			int markerPositionInVector = 0;
			for (size_t i = 0; i < markerIds.size(); ++i) {

				if (markerIds[i] == m_arucoReferenceID) {
					markerPositionInVector = static_cast<int>(i);
				}
			}

			// wich arucos are found? false if not found. (Create bool vector where we save all detected markers as true in that spesific index=id)
			std::vector<bool> idVectorFull;
			for (int i = 0; i < markersX*markersY; ++i) {
				std::vector<int>::iterator it;
				it = std::find(markerIds.begin(), markerIds.end(), i);
				if (it != markerIds.end()) {
					idVectorFull.push_back(true);
				}
				else {
					idVectorFull.push_back(false);
				}
			}

			// get all corners of detected aruco markers
			std::vector<std::vector<cv::Point3f>> boardMarkerPoints;

			for (size_t i = 0; i < board->objPoints.size(); i++) {
				if (idVectorFull[i])
					boardMarkerPoints.push_back(board->objPoints[i]);
			}

			std::vector<cv::Point2f> boardDetectedMarkerPoints;
			for (size_t i = 0; i < markerIds.size(); i++) {
				boardDetectedMarkerPoints.push_back(cv::Point2f(board->objPoints[markerIds[i]][0].x, board->objPoints[markerIds[i]][0].y));
				boardDetectedMarkerPoints.push_back(cv::Point2f(board->objPoints[markerIds[i]][1].x, board->objPoints[markerIds[i]][1].y));
				boardDetectedMarkerPoints.push_back(cv::Point2f(board->objPoints[markerIds[i]][2].x, board->objPoints[markerIds[i]][2].y));
				boardDetectedMarkerPoints.push_back(cv::Point2f(board->objPoints[markerIds[i]][3].x, board->objPoints[markerIds[i]][3].y));
			}

			std::vector<cv::Point2f> imageDetectedMarkerPoints;
			for (size_t i = 0; i < markerIds.size(); i++) {
				imageDetectedMarkerPoints.push_back(markerCorners[i][0]);
				imageDetectedMarkerPoints.push_back(markerCorners[i][1]);
				imageDetectedMarkerPoints.push_back(markerCorners[i][2]);
				imageDetectedMarkerPoints.push_back(markerCorners[i][3]);
			}

			cv::Mat homographyTransform = cv::findHomography(imageDetectedMarkerPoints, boardDetectedMarkerPoints, CV_RANSAC);

			// Calculate position of camera
			cv::Mat rvecT;
			cv::Rodrigues(rvec, rvecT);

			cv::Mat transformTmp = (cv::Mat_<double>(4, 4) << rvecT.at<double>(0, 0), rvecT.at<double>(0, 1), rvecT.at<double>(0, 2), tvec[0], 
						                          rvecT.at<double>(1, 0), rvecT.at<double>(1, 1), rvecT.at<double>(1, 2), tvec[1], 
									  rvecT.at<double>(2, 0), rvecT.at<double>(2, 1), rvecT.at<double>(2, 2), tvec[2], 
									  0, 0, 0, 1);
			
			cv::Mat transformCamToWorld;
			cv::invert(transformTmp, transformCamToWorld);

			cv::Point2f camPos = cv::Point2f(transformCamToWorld.at<double>(0, 3), transformCamToWorld.at<double>(1, 3));
			std::vector<cv::Point2f> camPosImg;
			std::vector<cv::Point2f> camPosBrd;
			camPosBrd.push_back(camPos);
			cv::Mat homographyTransformInverse = cv::findHomography(boardDetectedMarkerPoints, imageDetectedMarkerPoints, CV_RANSAC);
			cv::perspectiveTransform(camPosBrd, camPosImg, homographyTransformInverse);
			cv::circle(srcImg, camPosImg[0], 5, cv::Scalar(255, 0, 0), -1);

			std::vector<ObjectStruct> objects;

			cv::Mat drawcontour(pInputImage->size(), CV_8UC1, cv::Scalar(0, 0, 0));

			cv::Mat masks(pInputImage->size(), CV_8UC1, cv::Scalar(255, 255, 255));

			std::vector <std::vector <cv::Point> > contours;
			std::vector <std::vector <cv::Point> > singleContour;

			// Get contour of single mask
			for (size_t i = 0; i < pInputMasks->size(); ++i) {
				cv::Mat black;
				masks.copyTo(black, (*pInputMasks)[i]);

				cv::findContours(black, singleContour, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
				contours.push_back(singleContour[0]);
			}
			if (contours.size() == 0) {
				std::cout << "No contours detected!" << std::endl;
				return -1;
			}

			cv::cvtColor(drawcontour, drawcontour, CV_GRAY2RGB);
			drawcontour.convertTo(drawcontour, CV_8UC3);

			cv::RotatedRect minRect;
			cv::RotatedRect minEllipse;
			cv::Mat drawing(pInputImage->size(), CV_8UC3, cv::Scalar(0, 0, 0));
			

			for (size_t i = 0; i < contours.size(); ++i) {
			  
				// Find the rotated rectangles and ellipses for each contour
				minEllipse = cv::fitEllipse(cv::Mat(contours[i]));
				minRect = cv::minAreaRect(cv::Mat(contours[i]));

				// Draw contours + rotated rects + ellipses

				cv::Scalar color = cv::Scalar(m_rng.uniform(0, 255), m_rng.uniform(0, 255), m_rng.uniform(0, 255));
				// contour
				cv::drawContours(drawing, contours, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
				// ellipse
				cv::ellipse(drawing, minEllipse, color, 2, 8);
				// rotated rectangle
				cv::Point2f rect_points[4];
				minRect.points(rect_points);
				for (int j = 0; j < 4; j++)
					cv::line(drawing, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);

				// corner points
				std::vector<cv::Point2f> cornerPoints;
				cornerPoints.resize(4);
				std::vector<cv::Point2f> middlePoints;
				middlePoints.resize(4);

				float angle;
				cv::Point2f center;

				// compute coordinates and dimensions of objects
				center = minRect.center;
				cv::circle(srcImg, center, 3, CV_RGB(255, 0, 0), -1);
				float height = minRect.size.height;
				float width = minRect.size.width;
				angle = minRect.angle*CV_PI / 180.0;
				float b = (float)cos(angle)*0.5f;
				float a = (float)sin(angle)*0.5f;

				cornerPoints[0].x = center.x - a*height - b*width;
				cornerPoints[0].y = center.y + b*height - a*width;
				cornerPoints[1].x = center.x + a*height - b*width;
				cornerPoints[1].y = center.y - b*height - a*width;
				cornerPoints[2].x = 2 * center.x - cornerPoints[0].x;
				cornerPoints[2].y = 2 * center.y - cornerPoints[0].y;
				cornerPoints[3].x = 2 * center.x - cornerPoints[1].x;
				cornerPoints[3].y = 2 * center.y - cornerPoints[1].y;

				// middle points
				middlePoints[0] = (cornerPoints[1] + cornerPoints[0]) / 2;
				middlePoints[1] = (cornerPoints[2] + cornerPoints[1]) / 2;
				middlePoints[2] = (cornerPoints[3] + cornerPoints[2]) / 2;
				middlePoints[3] = (cornerPoints[0] + cornerPoints[3]) / 2;

				for (int j = 0; j < cornerPoints.size(); j++) {
					cv::circle(srcImg, cornerPoints[j], 3, CV_RGB(255, 0, 0), -1);
					cv::circle(srcImg, middlePoints[j], 3, CV_RGB(255, 0, 0), -1);
				}

				// Get the origin
				std::vector<cv::Point2f> referenceMarkerCorners = markerCorners[markerPositionInVector];
				cv::Point2f origin = referenceMarkerCorners[3]; // this value (0 to 3) in the vector depends on the id
				cv::circle(srcImg, origin, 3, CV_RGB(255, 0, 0), -1);
				cv::putText(srcImg, "Origin", cv::Point(origin.x, origin.y + 13), cv::FONT_HERSHEY_COMPLEX_SMALL, .8, cv::Scalar(0, 0, 255));

				// Calculate origin on board
				std::vector<cv::Point2f> originOnImageVector;
				originOnImageVector.push_back(origin);
				std::vector<cv::Point2f> originOnBoardVector;
				cv::perspectiveTransform(originOnImageVector, originOnBoardVector, homographyTransform);

				// Calculate middle points on board
				std::vector<cv::Point2f> middlePointsOnBoard;
				cv::perspectiveTransform(middlePoints, middlePointsOnBoard, homographyTransform);
				std::vector<cv::Point2f>  middlePointsOnBoardRegardingAruco;
				for (size_t i = 0; i < middlePointsOnBoard.size(); ++i) {
					cv::Point2f point = middlePointsOnBoard[i] - originOnBoardVector[0];
					middlePointsOnBoardRegardingAruco.push_back(point);
				}

				// Calculate corner points on board
				std::vector<cv::Point2f> cornerPointsOnBoard;
				cv::perspectiveTransform(cornerPoints, cornerPointsOnBoard, homographyTransform);
				std::vector<cv::Point2f>  cornerPointsOnBoardRegardingAruco;
				for (size_t i = 0; i < cornerPointsOnBoard.size(); ++i) {
					cv::Point2f point = cornerPointsOnBoard[i] - originOnBoardVector[0];
					cornerPointsOnBoardRegardingAruco.push_back(point);
				}

				// Calculate center on board
				std::vector<cv::Point2f> centerOnBoard;
				std::vector<cv::Point2f> centerVector;
				centerVector.push_back(center);
				cv::perspectiveTransform(centerVector, centerOnBoard, homographyTransform);
				cv::Point2f centerOnBoardRegardingAruco = centerOnBoard[0] - originOnBoardVector[0];

				// calculate angle
				cv::Point2f vecAngle = middlePointsOnBoardRegardingAruco[1] - middlePointsOnBoardRegardingAruco[3];
				float angleOnBoardRegardingAruco = cv::fastAtan2(vecAngle.y, -vecAngle.x);

				// calculate dimensions of object on board
				if (middlePointsOnBoardRegardingAruco.size() < 4) return 0;
				float objectHeight = cv::norm(middlePointsOnBoardRegardingAruco[0] - middlePointsOnBoardRegardingAruco[2]);
				float objectWidth = cv::norm(middlePointsOnBoardRegardingAruco[1] - middlePointsOnBoardRegardingAruco[3]);

				// Set width and height correct (width is always smaller) (for image)
				if (minRect.size.height < minRect.size.width)
				{
					cv::Size2f temp;
					temp.width = minRect.size.height;
					temp.height = minRect.size.width;
					minRect.size = temp;
				}

				// Set width and height correct (width is always smaller) (for board)
				if (objectHeight < objectWidth)
				{
					float temp;
					temp = objectHeight;
					objectWidth = objectHeight;
					objectHeight = temp;
				}

				// save all information in object struct
				ObjectStruct object;
				object.angleOnImage = angle;
				object.angleOnBoard = angleOnBoardRegardingAruco;
				object.centerOnImage = center;
				object.originOnImage = origin;
				object.centerOnBoard = centerOnBoardRegardingAruco;
				object.cornerPointsOnImage = cornerPoints;
				object.cornerPointsOnBoard = cornerPointsOnBoardRegardingAruco;
				object.middlePointsOnBoard = middlePointsOnBoardRegardingAruco;
				object.middlePointsOnImage = middlePoints;
				object.heightOnImage = minRect.size.height;
				object.widthOnImage = minRect.size.width;
				object.heightOnBoard = objectHeight;
				object.widthOnBoard = objectWidth;
				object.mask = (*pInputMasks)[i];
				objects.push_back(object);

#ifdef DEBUG
				std::cout << "---------Width on board: " << objectWidth << std::endl;
#endif

				/// Calculate actual (corrected) center point of object
				if ((minRect.size.width / minRect.size.height) < 0.65)
				{
					// Detect nearest middle point to camera
					cv::Point2f nearestMiddlePoint = objects.at(i).middlePointsOnImage[0];
					for (size_t j = 1; j < objects.at(i).middlePointsOnImage.size(); j++)
					{
						if (cv::norm(camPosImg[0] - nearestMiddlePoint) > cv::norm(camPosImg[0] - objects.at(i).middlePointsOnImage[j]))
						{
							nearestMiddlePoint = objects.at(i).middlePointsOnImage[j];
						}
					}
					cv::circle(srcImg, nearestMiddlePoint, 3, CV_RGB(0, 255, 0), -1);
					cv::putText(srcImg, "2nd", nearestMiddlePoint + cv::Point2f(0, -5), cv::FONT_HERSHEY_SIMPLEX, .3, cv::Scalar(0, 0, 255));

					// Detect first nearest corner point to camera
					int indexFir = 0;
					cv::Point2f nearestCornerPointFir = objects.at(i).cornerPointsOnImage[0];
					for (size_t j = 1; j < objects.at(i).cornerPointsOnImage.size(); j++)
					{
						if (cv::norm(camPosImg[0] - nearestCornerPointFir) > cv::norm(camPosImg[0] - objects.at(i).cornerPointsOnImage[j]))
						{
							nearestCornerPointFir = objects.at(i).cornerPointsOnImage[j];
							indexFir = j;
						}
					}
					cv::circle(srcImg, nearestCornerPointFir, 3, CV_RGB(0, 255, 0), -1);
					cv::putText(srcImg, "1st", nearestCornerPointFir + cv::Point2f(0, -5), cv::FONT_HERSHEY_SIMPLEX, .3, cv::Scalar(0, 0, 255));

					// Detect second nearest corner point to camera
					cv::Point2f nearestCornerPointSec;
					if (indexFir != 0)
						nearestCornerPointSec = objects.at(i).cornerPointsOnImage[0];
					else
						nearestCornerPointSec = objects.at(i).cornerPointsOnImage[1];

					for (size_t j = 0; j < objects.at(i).cornerPointsOnImage.size(); j++)
					{
						if ((cv::norm(camPosImg[0] - nearestCornerPointSec) > cv::norm(camPosImg[0] - objects.at(i).cornerPointsOnImage[j])) && (j != indexFir))
						{
							nearestCornerPointSec = objects.at(i).cornerPointsOnImage[j];
						}
					}
					cv::circle(srcImg, nearestCornerPointSec, 3, CV_RGB(0, 255, 0), -1);
					cv::putText(srcImg, "3rd", nearestCornerPointSec + cv::Point2f(0, -5), cv::FONT_HERSHEY_SIMPLEX, .3, cv::Scalar(0, 0, 255));

					cv::circle(srcImg, camPosImg[0], 5, CV_RGB(0, 255, 0), -1);

					// calculate actual center point
					cv::Point2f actualCenterPoint;
					if (nearestCornerPointSec.x < nearestCornerPointFir.x)
						actualCenterPoint = nearestMiddlePoint + cv::Point2f(-((nearestCornerPointSec - nearestCornerPointFir) / 2).y, ((nearestCornerPointSec - nearestCornerPointFir) / 2).x);
					else
						actualCenterPoint = nearestMiddlePoint + cv::Point2f(((nearestCornerPointSec - nearestCornerPointFir) / 2).y, -((nearestCornerPointSec - nearestCornerPointFir) / 2).x);

					cv::circle(srcImg, actualCenterPoint, 5, CV_RGB(0, 255, 255), 1);
					cv::putText(srcImg, "actual center", actualCenterPoint + cv::Point2f(-20, -10), cv::FONT_HERSHEY_SIMPLEX, .3, CV_RGB(0, 255, 255));

					// Correct centerpoint of object
					objects.at(i).correctedCenterOnImage = actualCenterPoint;
				}
			}
#ifdef DEBUG
			std::cout << "-------- next image" << std::endl;
			imshowScaled2("drawing", drawing);
			cv::waitKey(1);
#endif
#ifdef INFO
			imshowScaled2("Relevant Points", srcImg);
#endif

			pData->second.push_back(std::make_pair(m_outputObjectDataName,
				std::make_shared<std::vector<ObjectStruct>>(objects)));

			isFilterSuccess = 1;

			m_inputDataNames = inputDataNames;
			return isFilterSuccess;
		}

		// -------------------------------------------------------------------------------------------------

	} // namespace 

} // namespace 