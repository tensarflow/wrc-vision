#include "WorkplaceCoordinateSystems.hpp"


#include <string> 

#define DEBUG

#if 0

namespace {
	
//ony for robot Aruco constalation relevant 
static std::vector<cv::Point3f> Create3DArucoMarkerCorners(float arucoSize, float distBetweenMarkers)
{
	// This function creates the 3D points of your chessboard in its own coordinate system
	std::vector<cv::Point3f> corners;
	corners = {
		cv::Point3f(0, 0, 0),
		cv::Point3f(arucoSize, 0, 0),
		cv::Point3f(arucoSize, arucoSize, 0),
		cv::Point3f(0, arucoSize, 0),

		cv::Point3f(arucoSize + distBetweenMarkers, 0, 0),
		cv::Point3f(2 * arucoSize + distBetweenMarkers, 0, 0),
		cv::Point3f(2 * arucoSize + distBetweenMarkers, arucoSize, 0),
		cv::Point3f(arucoSize + distBetweenMarkers, arucoSize, 0) };
	return corners;
}

struct ArucoCornersandID {
	ArucoCornersandID(std::vector<cv::Point2f> _corners, int _id) : corners(_corners), id(_id) {}
	std::vector<cv::Point2f> corners;
	int id;
};

struct by_ID {
	bool operator()(ArucoCornersandID const &a, ArucoCornersandID const &b) {
		return a.id < b.id;
	}
};

} // namespace

#endif

namespace ImgPipe {
	namespace FilterProcessingPipe {

		// -------------------------------------------------------------------------------------------------

		WorkplaceCoordinateSystems::WorkplaceCoordinateSystems() : m_progress(0.0) {

			std::shared_ptr<ImageFilter::ParameterDescription> pParameterDescription
				= std::make_shared<ImageFilter::ParameterDescription>();
			pParameterDescription->parameterName = "InputImageName";
			pParameterDescription->parameterType = "string";
			m_pParametersDescription.push_back(pParameterDescription);
			
			std::shared_ptr<ImageFilter::ParameterDescription> pParameterDescription1
				= std::make_shared<ImageFilter::ParameterDescription>();
			pParameterDescription1->parameterName = "InputDistortionCoefficientsName";
			pParameterDescription1->parameterType = "string";
			m_pParametersDescription.push_back(pParameterDescription1);
			
			std::shared_ptr<ImageFilter::ParameterDescription> pParameterDescription2
				= std::make_shared<ImageFilter::ParameterDescription>();
			pParameterDescription2->parameterName = "InputCalibrationMatrixName";
			pParameterDescription2->parameterType = "string";
			m_pParametersDescription.push_back(pParameterDescription2);
			
			std::shared_ptr<ImageFilter::ParameterDescription> pParameterDescription3
				= std::make_shared<ImageFilter::ParameterDescription>();
			pParameterDescription3->parameterName = "InputCoordFrameNames";
			pParameterDescription3->parameterType = "string";
			m_pParametersDescription.push_back(pParameterDescription3);

			std::shared_ptr<ImageFilter::ParameterDescription> pParameterDescription4
				= std::make_shared<ImageFilter::ParameterDescription>();
			pParameterDescription4->parameterName = "InputFrameROIs";
			pParameterDescription4->parameterType = "string";
			m_pParametersDescription.push_back(pParameterDescription4);
		}


		// -------------------------------------------------------------------------------------------------

		std::string WorkplaceCoordinateSystems::filterName() {
			return "WorkplaceCoordinateSystems";
		}

		// -------------------------------------------------------------------------------------------------

		double WorkplaceCoordinateSystems::progress() {
			return m_progress;
		}

		// -------------------------------------------------------------------------------------------------

		std::vector<std::shared_ptr<ImageFilter::ParameterDescription> >
			WorkplaceCoordinateSystems::parameterDescriptions() {

			return m_pParametersDescription;
		}

		// -------------------------------------------------------------------------------------------------

		std::vector<boost::any> WorkplaceCoordinateSystems::parameterValues() {
			std::vector<boost::any> vector;
			vector.push_back(m_inputImageName);
			vector.push_back(m_inputDistortionCoefficientsName);
			vector.push_back(m_inputCalibrationMatrixName);
			
			std::stringstream ss;
			ss.str("");
			size_t i = 0;
			for (std::string id : m_inputCoordFrameNames) {
				ss << id;
				i++;
				if (i < m_inputCoordFrameNames.size()) {
					ss << ",";
				}
			}
			vector.push_back(ss.str());
			
			ss.str("");
			i = 0;
			for (auto e : m_inputFrameROIs) {
				cv::Rect roi = e.second;
				ss << roi.x << "," << roi.y << "," << (roi.x + roi.width) << "," << (roi.y + roi.height);
				if (i < m_inputFrameROIs.size()) {
					ss << ",";
				}
			}
			vector.push_back(ss.str());

			return vector;
		}

		// -------------------------------------------------------------------------------------------------

		void WorkplaceCoordinateSystems::setParameter(const std::vector<boost::any>& newValues) {
			try {
				m_inputImageName = boost::any_cast<std::string>(newValues[0]);
				m_inputDistortionCoefficientsName = boost::any_cast<std::string>(newValues[1]);
				m_inputCalibrationMatrixName = boost::any_cast<std::string>(newValues[2]);

				std::string input = boost::any_cast<std::string>(newValues[3]);
				m_inputCoordFrameNames.clear();
				size_t pos = input.find(","), lpos = 0;
				while (pos != std::string::npos) {
					m_inputCoordFrameNames.push_back(input.substr(lpos, pos - lpos));
					lpos = pos + 1;
					pos = input.find(",", lpos);
				}
				
				input = boost::any_cast<std::string>(newValues[4]);
				m_inputFrameROIs.clear();
				size_t idx = 0;
				size_t n = 0;
				cv::Rect roi;
				pos = input.find(",");
				lpos = 0;
				while (pos != std::string::npos) {
					std::string val = input.substr(lpos, pos - lpos);
					std::cout << val << std::endl;
					if (n == 0) { // left
						roi.x = std::stoi(val);
					} else if (n == 1) { // top
						roi.y = std::stoi(val);
					} else if (n == 2) { // right
						roi.width = std::stoi(val) - roi.x - 1;
					} else if (n == 3) { // bottom
						roi.height = std::stoi(val) - roi.y - 1;
					}
					
					if (n == 3) {
						m_inputFrameROIs[m_inputCoordFrameNames[idx]] = roi;
						std::cout << m_inputCoordFrameNames[idx] << " " << roi << std::endl;
						idx++;
					}
					
					n = (n + 1) % 4;
					
					lpos = pos + 1;
					pos = input.find(",", lpos);
				}
			} catch (const boost::bad_any_cast &e) {
				std::cout << "WorkplaceCoordinateSystems::setParameter failed!" << std::endl;
			}
		}

		// -------------------------------------------------------------------------------------------------

		std::vector<std::string> WorkplaceCoordinateSystems::inputDataNames() {
			boost::mutex::scoped_lock l(m_mutex);
			return m_inputDataNames;
		}

		// -------------------------------------------------------------------------------------------------

		int WorkplaceCoordinateSystems::applyFilter(std::shared_ptr<ImageFilterData>& pData) {
			
			std::cout << "WorkplaceCoordinateSystems::applyFilter" << std::endl;

			int isFilterSuccess = 0;
			std::vector<std::string> inputDataNames;

			std::shared_ptr<cv::Mat> pImage;
			std::shared_ptr<Eigen::VectorXd> pInputDistortionCoefficients;
			std::shared_ptr<Eigen::Matrix3d> pInputCalibrationMatrix;
			for (size_t i = 0; i < pData->second.size(); ++i) {

				// get input data names (for GUI)
				inputDataNames.push_back(pData->second[i].first);

				// cast data
				std::shared_ptr<cv::Mat> pImageTmp
					= castInput<std::shared_ptr<cv::Mat> >(pData->second[i], m_inputImageName);

				if (pImageTmp) {
					pImage = pImageTmp;
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

			}
			// Check Data
			if (!pImage) { std::cout << "Couldn't load image!" << std::endl; return -1; }
			if (pImage->channels() == 0 || pImage->rows == 0 || pImage->cols == 0) { std::cout << "Couldn't load image!" << std::endl; return -1; }
			if (pInputDistortionCoefficients == NULL){ std::cout << "Couldn't read Distortion Coeffs!" << std::endl; return -1;}
			if (pInputCalibrationMatrix == NULL){ std::cout << "Couldn't read Calibration Coeffs!" << std::endl; return -1;}
			
			cv::Mat img = *pImage;
			//cv::imwrite("source.png", img);
			
			// Show source image
			//cv::imshow("Source Image", *pImage);
			//cv::waitKey(1);
			cv::Mat drawImg = pImage->clone();
			
			cv::Mat maskedImg(img.size(), img.type());

			/// Variables used below
			// Generate camera parameters
			cv::Mat cameraMatrix;
			cv::Mat distCoeffs;
			cv::eigen2cv(*pInputCalibrationMatrix, cameraMatrix);
			cv::eigen2cv(*pInputDistortionCoefficients, distCoeffs);
			
			// Transformations
			cv::Vec3d rvecBoardPlacementMat, tvecBoardPlacementMat;
			cv::Vec3d rvecBoardTaskBoard, tvecBoardTaskBoard;
			cv::Vec3d rvecBoardWorld, tvecBoardWorld;
			
			cv::Rect roi;
			
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// Aruco Board World
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			///Create and detect board
			// Create board
#if 0
			int markersXWorld = 11; // check
			int markersYWorld = 5;  // check
			float markerLengthWorld = 0.040; // check
			float markerSeparationWorld = 0.005; // check
			cv::Ptr<cv::aruco::Dictionary> dictionaryWorld = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_100);
			cv::Ptr<cv::aruco::GridBoard> boardWorld
			      = cv::aruco::GridBoard::create(markersXWorld, markersYWorld, markerLengthWorld, markerSeparationWorld, dictionaryWorld);

			// Detect board
			std::vector<int> IDsBoardWorld;
			std::vector<std::vector<cv::Point2f>> markerCornersBoardWorld;
			maskedImg.setTo(255);
			roi = m_inputFrameROIs["WorkplaceCoordinateSystems_Reference"];
			img(roi).copyTo(maskedImg(roi));
			cv::aruco::detectMarkers(maskedImg, boardWorld->dictionary, markerCornersBoardWorld, IDsBoardWorld);
			
			cv::rectangle(drawImg, roi, CV_RGB(0x00, 0xFF, 0xFF), 3);

			if (IDsBoardWorld.size() > 3) {
				// Draw detected markers of board
				cv::aruco::drawDetectedMarkers(drawImg, markerCornersBoardWorld, IDsBoardWorld);

				// Estimate pose board
				int valid = cv::aruco::estimatePoseBoard(markerCornersBoardWorld, IDsBoardWorld, boardWorld, cameraMatrix, distCoeffs, rvecBoardWorld, tvecBoardWorld);

				if (valid) {
					cv::aruco::drawAxis(drawImg, cameraMatrix, distCoeffs, rvecBoardWorld, tvecBoardWorld, 0.1);
				}
			}
#else
			maskedImg.setTo(255);
			roi = m_inputFrameROIs["WorkplaceCoordinateSystems_Reference"];
			img(roi).copyTo(maskedImg(roi));
			
			cv::Ptr<cv::aruco::Dictionary> dictionaryWorldSingleMarker = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_50);

			// Detect single markers
			std::vector<int> worldIDsSingleMarker;
			std::vector< std::vector<cv::Point2f> > markerCornersSingleMarker;
			cv::aruco::detectMarkers(maskedImg, dictionaryWorldSingleMarker, markerCornersSingleMarker, worldIDsSingleMarker);
			
			// Estimate pose single marker
			std::vector< cv::Vec3d > rvecsWorldMarker, tvecsWorldMarker;	
			cv::aruco::estimatePoseSingleMarkers(markerCornersSingleMarker, 0.060, cameraMatrix, distCoeffs, rvecsWorldMarker, tvecsWorldMarker);
			
			// Get indices of the wanted markers (0, 1, 3, 4, ...)
			
			auto itrWorld = std::find(worldIDsSingleMarker.begin(), worldIDsSingleMarker.end(), 5);
			if (itrWorld != worldIDsSingleMarker.end()) {
				size_t indexMarkerId = std::distance(worldIDsSingleMarker.begin(), itrWorld);	
				rvecBoardWorld = rvecsWorldMarker[indexMarkerId];
				tvecBoardWorld = tvecsWorldMarker[indexMarkerId];

				cv::aruco::drawAxis(drawImg, cameraMatrix, distCoeffs, rvecsWorldMarker[indexMarkerId], tvecsWorldMarker[indexMarkerId], 1.);
			}
#endif
			
			for (size_t i = 0; i < m_inputCoordFrameNames.size(); i++) {
				std::string cfID = m_inputCoordFrameNames[i];
				cv::Rect roi = m_inputFrameROIs[cfID];
				
				// set masked image
				maskedImg.setTo(255);
				img(roi).copyTo(maskedImg(roi));
				cv::rectangle(drawImg, roi, CV_RGB(0xFF, 0x00, 0xFF), 1);
				
				if (cfID.compare("WorkplaceCoordinateSystems_BoardInReference") == 0) {
					
					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// Aruco Board Task Board
					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					///Create and detect board
					// Create board
					int markersXTaskBoard = 10; // check
					int markersYTaskBoard = 10;  // check
					float markerLengthTaskBoard = 0.030; // check
					float markerSeparationTaskBoard = 0.009; // check
					cv::Ptr<cv::aruco::Dictionary> dictionaryTaskBoard = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100);
					cv::Ptr<cv::aruco::GridBoard> boardTaskBoard
						  = cv::aruco::GridBoard::create(markersXTaskBoard, markersYTaskBoard, markerLengthTaskBoard, markerSeparationTaskBoard, dictionaryTaskBoard);

					// Detect board
					std::vector<int> IDsBoardTaskBoard;
					std::vector<std::vector<cv::Point2f>> markerCornersBoardTaskBoard;
					cv::aruco::detectMarkers(maskedImg, boardTaskBoard->dictionary, markerCornersBoardTaskBoard, IDsBoardTaskBoard);

					if (IDsBoardTaskBoard.size() > 3) {
						// Draw detected markers of board
						cv::aruco::drawDetectedMarkers(drawImg, markerCornersBoardTaskBoard, IDsBoardTaskBoard);

						// Estimate pose board
						int valid = cv::aruco::estimatePoseBoard(markerCornersBoardTaskBoard, IDsBoardTaskBoard, boardTaskBoard, cameraMatrix, distCoeffs, rvecBoardTaskBoard, tvecBoardTaskBoard);

						if (valid) {
							// Calculate transformation
							cv::Mat taskBoardInWorld = ImgPipe::Library::TransformCoordinateFrames(rvecBoardTaskBoard, tvecBoardTaskBoard, rvecBoardWorld, tvecBoardWorld);
							
							// Publish data to image pipeline
							pData->second.push_back(std::make_pair("WorkplaceCoordinateSystems_BoardInReference",
								std::make_shared<cv::Mat>(ImgPipe::Library::FixXYPlane(taskBoardInWorld))));
							
							cv::aruco::drawAxis(drawImg, cameraMatrix, distCoeffs, rvecBoardTaskBoard, tvecBoardTaskBoard, 0.1);
						}
					}
					
				} else if (cfID.compare("WorkplaceCoordinateSystems_PlacementMatInReference") == 0) {
					
					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// Aruco Board  Placement Mat
					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					///Create and detect board
					// Create board
					int markersXPlacementMat = 12;
					int markersYPlacementMat = 12;
					float markerLengthPlacementMat = 0.035;
					float markerSeparationPlacementMat = 0.006/*0.005*/;
					cv::Ptr<cv::aruco::Dictionary> dictionaryBoardPlacementMat = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_1000);
					cv::Ptr<cv::aruco::GridBoard> boardPlacementMat 
						  = cv::aruco::GridBoard::create(markersXPlacementMat, markersYPlacementMat, markerLengthPlacementMat, markerSeparationPlacementMat, dictionaryBoardPlacementMat);


					// Detect board
					std::vector<int> IDsBoardPlacementMat;
					std::vector<std::vector<cv::Point2f>> markerCornersBoardPlacementMat;
					cv::aruco::detectMarkers(maskedImg, boardPlacementMat->dictionary, markerCornersBoardPlacementMat, IDsBoardPlacementMat);

					if (IDsBoardPlacementMat.size() > 3) {
						// Draw detected markers of board
						cv::aruco::drawDetectedMarkers(drawImg, markerCornersBoardPlacementMat, IDsBoardPlacementMat);	

						// Estimate pose board
						int valid = cv::aruco::estimatePoseBoard(markerCornersBoardPlacementMat, IDsBoardPlacementMat, boardPlacementMat, cameraMatrix, distCoeffs, rvecBoardPlacementMat, tvecBoardPlacementMat);

						if (valid) {
							// Calculate Transformation
							cv::Mat placementMatInWorld = ImgPipe::Library::TransformCoordinateFrames(rvecBoardPlacementMat, tvecBoardPlacementMat, rvecBoardWorld, tvecBoardWorld);
							
							// Publish data to image pipeline
							pData->second.push_back(std::make_pair("WorkplaceCoordinateSystems_PlacementMatInReference",
								std::make_shared<cv::Mat>(ImgPipe::Library::FixXYPlane(placementMatInWorld))));
							
							cv::aruco::drawAxis(drawImg, cameraMatrix, distCoeffs, rvecBoardPlacementMat, tvecBoardPlacementMat, 0.1);
						}
					}
					
				} else if (cfID.compare("WorkplaceCoordinateSystems_Box1InWorld") == 0) {
					
					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// Aruco Board  box 1
					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					///Create and detect board
					// Create board
					int markersXBox1 = 8; // todo
					int markersYBox1 = 6; // todo
					float markerLengthBox1 = 0.042; // todo
					float markerSeparationBox1 = 0.012; // todo
					cv::Ptr<cv::aruco::Dictionary> dictionaryBoardBox1 = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50); // todo
					cv::Ptr<cv::aruco::GridBoard> boardBox1 
						  = cv::aruco::GridBoard::create(markersXBox1, markersYBox1, markerLengthBox1, markerSeparationBox1, dictionaryBoardBox1);

					// Detect board
					std::vector<int> IDsBoardBox1;
					std::vector<std::vector<cv::Point2f>> markerCornersBoardBox1;
					cv::aruco::detectMarkers(maskedImg, boardBox1->dictionary, markerCornersBoardBox1, IDsBoardBox1);

					if (IDsBoardBox1.size() > 3) {
						// Draw detected markers of board
						cv::aruco::drawDetectedMarkers(drawImg, markerCornersBoardBox1, IDsBoardBox1);	

						// Estimate pose board
						cv::Vec3d rvecBoardBox1, tvecBoardBox1;
						int valid = cv::aruco::estimatePoseBoard(markerCornersBoardBox1, IDsBoardBox1, boardBox1, cameraMatrix, distCoeffs, rvecBoardBox1, tvecBoardBox1);

						if (valid) {
							cv::Mat box1InWorld = ImgPipe::Library::TransformCoordinateFrames(rvecBoardBox1, tvecBoardBox1, rvecBoardWorld, tvecBoardWorld);
							
							pData->second.push_back(std::make_pair("WorkplaceCoordinateSystems_Box1InWorld",
								std::make_shared<cv::Mat>(ImgPipe::Library::FixXYPlane(box1InWorld))));
							
							cv::aruco::drawAxis(drawImg, cameraMatrix, distCoeffs, rvecBoardBox1, tvecBoardBox1, 0.1);
						}
					}					
					
				} else if (cfID.compare("WorkplaceCoordinateSystems_Box2InWorld") == 0) {

					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// Aruco Board box 2
					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					///Create and detect board
					// Create board
					int markersXBox2 = 8; // check
					int markersYBox2 = 6;  // check
					float markerLengthBox2 = 0.042; // check
					float markerSeparationBox2 = 0.012; // check
					cv::Ptr<cv::aruco::Dictionary> dictionaryBox2 = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);
					cv::Ptr<cv::aruco::GridBoard> boardBox2
						  = cv::aruco::GridBoard::create(markersXBox2, markersYBox2, markerLengthBox2, markerSeparationBox2, dictionaryBox2);

					// Detect board
					std::vector<int> IDsBoardBox2;
					std::vector<std::vector<cv::Point2f>> markerCornersBoardBox2;
					cv::aruco::detectMarkers(maskedImg, boardBox2->dictionary, markerCornersBoardBox2, IDsBoardBox2);
					if (IDsBoardBox2.size() > 3) {
						// Draw detected markers of board
						cv::aruco::drawDetectedMarkers(drawImg, markerCornersBoardBox2, IDsBoardBox2);

						// Estimate pose board
						cv::Vec3d rvecBoardBox2, tvecBoardBox2;
						int valid = cv::aruco::estimatePoseBoard(markerCornersBoardBox2, IDsBoardBox2, boardBox2, cameraMatrix, distCoeffs, rvecBoardBox2, tvecBoardBox2);

						if (valid) {
							// Calculate transformation
							cv::Mat box2InWorld = ImgPipe::Library::TransformCoordinateFrames(rvecBoardBox2, tvecBoardBox2, rvecBoardWorld, tvecBoardWorld);
							
							pData->second.push_back(std::make_pair("WorkplaceCoordinateSystems_Box2InWorld",
								std::make_shared<cv::Mat>(ImgPipe::Library::FixXYPlane(box2InWorld))));
							
							cv::aruco::drawAxis(drawImg, cameraMatrix, distCoeffs, rvecBoardBox2, tvecBoardBox2, 0.1);
						}
					}

				} else if (cfID.compare("WorkplaceCoordinateSystems_Rack1InWorld") == 0) {
					///Create and detect markers
					// Create marker for reference and for placement mat
					cv::Ptr<cv::aruco::Dictionary> dictionarySingleMarker = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_50);

					// Detect single markers
					std::vector<int> IDsSingleMarker;
					std::vector< std::vector<cv::Point2f> > markerCornersSingleMarker;
					cv::aruco::detectMarkers(maskedImg, dictionarySingleMarker, markerCornersSingleMarker, IDsSingleMarker);
					
					// Get indices of the wanted markers (0, 1, 3, 4, ...)
					std::vector<int>::iterator itr;
					
					int markerId0 = 1;
					auto itr0 = std::find(IDsSingleMarker.begin(), IDsSingleMarker.end(), markerId0);
					
					int markerId1 = 2;
					auto itr1 = std::find(IDsSingleMarker.begin(), IDsSingleMarker.end(), markerId1);
					
					// Draw detected markers
					cv::aruco::drawDetectedMarkers(drawImg, markerCornersSingleMarker, IDsSingleMarker);
#if 0				
					if (itr0 != IDsSingleMarker.end() && itr1 != IDsSingleMarker.end()) {
						auto corners3D = Create3DArucoMarkerCorners(0.06f, 0.862f);

						std::vector<cv::Point2f> arucoCorners_temp;
						std::vector<ArucoCornersandID> arucoCornersandID;

						for (int i = 0; i < arucoCorners.size(); i++) {
							arucoCornersandID.emplace_back(ArucoCornersandID(markerCornersSingleMarker[i], IDsSingleMarker[i]));
						}

						std::sort(arucoCornersandID.begin(), arucoCornersandID.end(), by_ID());

						for (int i = 0; i < arucoCornersandID.size(); i++) {
							for (int j = 0; j < arucoCornersandID[i].corners.size(); j++) {
								arucoCorners_temp.push_back(arucoCornersandID[i].corners[j]);
							}
						}
						
						cv::Vec3d rotVec, transVec;
						
						if (cv::solvePnP(corners3D, arucoCorners_temp, cameraMatrix, distCoeffs, rotVec, transVec)) {
							cv::aruco::drawAxis(drawImg, cameraMatrix, distCoeffs, rotVec, transVec, 1.);
						}
						
					} else {
#endif
						// Estimate pose single marker
						std::vector< cv::Vec3d > rvecsSingleMarker, tvecsSingleMarker;	
						cv::aruco::estimatePoseSingleMarkers(markerCornersSingleMarker, 0.060, cameraMatrix, distCoeffs, rvecsSingleMarker, tvecsSingleMarker);

						cv::Vec3d rvecMarker0, tvecMarker0;
						cv::Vec3d rvecMarker1, tvecMarker1;
						
						if (itr0 != IDsSingleMarker.end()) {
							size_t indexMarkerId0 = std::distance(IDsSingleMarker.begin(), itr0);
							rvecMarker0 = rvecsSingleMarker[indexMarkerId0];
							tvecMarker0 = tvecsSingleMarker[indexMarkerId0];
							cv::Mat marker0InWorld = ImgPipe::Library::TransformCoordinateFrames(rvecMarker0, tvecMarker0, rvecBoardWorld, tvecBoardWorld);
							pData->second.push_back(std::make_pair("WorkplaceCoordinateSystems_Marker1InWorld",
								std::make_shared<cv::Mat>(ImgPipe::Library::FixXYPlane(marker0InWorld))));
							
							pData->second.push_back(std::make_pair("WorkplaceCoordinateSystems_Rack1InWorld",
								std::make_shared<cv::Mat>(ImgPipe::Library::FixXYPlane(marker0InWorld))));

							cv::aruco::drawAxis(drawImg, cameraMatrix, distCoeffs, rvecsSingleMarker[indexMarkerId0], tvecsSingleMarker[indexMarkerId0], 1.);
						}

						
						if (itr1 != IDsSingleMarker.end()) {
							size_t indexMarkerId1 = std::distance(IDsSingleMarker.begin(), itr1);
							rvecMarker1 = rvecsSingleMarker[indexMarkerId1];
							tvecMarker1 = tvecsSingleMarker[indexMarkerId1];
							cv::Mat marker1InWorld = ImgPipe::Library::TransformCoordinateFrames(rvecMarker1, tvecMarker1, rvecBoardWorld, tvecBoardWorld);
							pData->second.push_back(std::make_pair("WorkplaceCoordinateSystems_Marker2InWorld",
								std::make_shared<cv::Mat>(ImgPipe::Library::FixXYPlane(marker1InWorld))));

							cv::aruco::drawAxis(drawImg, cameraMatrix, distCoeffs, rvecsSingleMarker[indexMarkerId1], tvecsSingleMarker[indexMarkerId1], 1.);
						}
					//}
				} else if (cfID.compare("WorkplaceCoordinateSystems_Rack2InWorld") == 0) {
					///Create and detect markers
					// Create marker for reference and for placement mat
					cv::Ptr<cv::aruco::Dictionary> dictionarySingleMarker = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_50);

					// Detect single markers
					std::vector<int> IDsSingleMarker;
					std::vector< std::vector<cv::Point2f> > markerCornersSingleMarker;
					cv::aruco::detectMarkers(maskedImg, dictionarySingleMarker, markerCornersSingleMarker, IDsSingleMarker);
					
					// Estimate pose single marker
					std::vector< cv::Vec3d > rvecsSingleMarker, tvecsSingleMarker;	
					cv::aruco::estimatePoseSingleMarkers(markerCornersSingleMarker, 0.060, cameraMatrix, distCoeffs, rvecsSingleMarker, tvecsSingleMarker);

					cv::Vec3d rvecMarker2, tvecMarker2;
					cv::Vec3d rvecMarker3, tvecMarker3;
					
					// Get indices of the wanted markers (0, 1, 3, 4, ...)
					std::vector<int>::iterator itr;

					int markerId2 = 3;
					itr = std::find(IDsSingleMarker.begin(), IDsSingleMarker.end(), markerId2);
					if (itr != IDsSingleMarker.end()) {
						size_t indexMarkerId2 = std::distance(IDsSingleMarker.begin(), itr);
						rvecMarker2 = rvecsSingleMarker[indexMarkerId2];
						tvecMarker2 = tvecsSingleMarker[indexMarkerId2];
						cv::Mat marker2InWorld = ImgPipe::Library::TransformCoordinateFrames(rvecMarker2, tvecMarker2, rvecBoardWorld, tvecBoardWorld);
						pData->second.push_back(std::make_pair("WorkplaceCoordinateSystems_Marker3InWorld",
							std::make_shared<cv::Mat>(ImgPipe::Library::FixXYPlane(marker2InWorld))));
						
						pData->second.push_back(std::make_pair("WorkplaceCoordinateSystems_Rack2InWorld",
							std::make_shared<cv::Mat>(ImgPipe::Library::FixXYPlane(marker2InWorld))));

						cv::aruco::drawAxis(drawImg, cameraMatrix, distCoeffs, rvecsSingleMarker[indexMarkerId2], tvecsSingleMarker[indexMarkerId2], 1.);
					}

					int markerId3 = 4;
					itr = std::find(IDsSingleMarker.begin(), IDsSingleMarker.end(), markerId3);
					if (itr != IDsSingleMarker.end()) {
						size_t indexMarkerId3 = std::distance(IDsSingleMarker.begin(), itr);	
						rvecMarker3 = rvecsSingleMarker[indexMarkerId3];
						tvecMarker3 = tvecsSingleMarker[indexMarkerId3];
						cv::Mat marker3InWorld = ImgPipe::Library::TransformCoordinateFrames(rvecMarker3, tvecMarker3, rvecBoardWorld, tvecBoardWorld);
						pData->second.push_back(std::make_pair("WorkplaceCoordinateSystems_Marker4InWorld",
							std::make_shared<cv::Mat>(ImgPipe::Library::FixXYPlane(marker3InWorld))));

						cv::aruco::drawAxis(drawImg, cameraMatrix, distCoeffs, rvecsSingleMarker[indexMarkerId3], tvecsSingleMarker[indexMarkerId3], 1.);
					}
				}
			}

			
			cv::resize(drawImg, drawImg, cv::Size(drawImg.cols/2, drawImg.rows/2));
			cv::imshow("Board Pose Estimation", drawImg);
			cv::waitKey(500);
					
					
					
					

// 			///Create and detect markers
// 			// Create marker for reference and for placement mat
// 			cv::Ptr<cv::aruco::Dictionary> dictionarySingleMarker = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
// 
// 			// Uncomment this if you want to print the markers
// 			/*cv::Mat markerImageReference;
// 			cv::Mat markerImagePlacementMat;
// 			cv::aruco::drawMarker(dictionarySingleMarker, 23, 200, markerImageReference, 1);
// 			cv::aruco::drawMarker(dictionarySingleMarker, 24, 200, markerImagePlacementMat, 1);
// 			cv::imwrite("markerImageReference.jpg", markerImageReference);
// 			cv::imwrite("markerImagePlacementMat.jpg", markerImagePlacementMat);*/
// 
// 			// Detect single markers
// 			std::vector< int > IDsSingleMarker;
// 			std::vector< std::vector<cv::Point2f> > markerCornersSingleMarker;
// 			cv::aruco::detectMarkers(*pImage, dictionarySingleMarker, markerCornersSingleMarker, IDsSingleMarker);
// 			std::vector< cv::Vec3d > rvecsSingleMarker, tvecsSingleMarker;
// 
// 
// 			int indexReferenceMarker;
// 			int indexPlacementMatMarker;
// 			if (IDsSingleMarker.size() > 0)
// 			{
// 				// Draw detected markers
// 				cv::aruco::drawDetectedMarkers(drawImg, markerCornersSingleMarker, IDsSingleMarker);
// 				cv::imshow("Single Marker Detection", drawImg);
// 				cv::waitKey(1);
// 
// 				// Get index of the wanted marker (23 -> reference; 24 -> placement mat)
// 				std::vector<int>::iterator itr;
// 				int referenceMarkerID = m_inputReferenceMarkerIDName;
// 				int placementMatMarkerID = m_inputPlacementMatMarkerIDName;
// 				itr = std::find(IDsSingleMarker.begin(), IDsSingleMarker.end(), referenceMarkerID);
// 				indexReferenceMarker = std::distance(IDsSingleMarker.begin(), itr);
// 				itr = std::find(IDsSingleMarker.begin(), IDsSingleMarker.end(), placementMatMarkerID);
// 				indexPlacementMatMarker = std::distance(IDsSingleMarker.begin(), itr);
// 
// 				// Estimate pose single marker
// 				cv::aruco::estimatePoseSingleMarkers(markerCornersSingleMarker, 0.045, cameraMatrix, distCoeffs, rvecsSingleMarker, tvecsSingleMarker);
// 				cv::aruco::drawAxis(drawImg, cameraMatrix, distCoeffs, rvecsSingleMarker[indexReferenceMarker], tvecsSingleMarker[indexReferenceMarker], 0.0225);
// 				cv::aruco::drawAxis(drawImg, cameraMatrix, distCoeffs, rvecsSingleMarker[indexPlacementMatMarker], tvecsSingleMarker[indexPlacementMatMarker], 0.0225);
// 				cv::imshow("Single Marker Pose Estimation", drawImg);
// 				cv::waitKey(1);
// 			}
// 			else
// 			{
// 				std::cout << "Could not find single markers: " << std::endl;
// 				return 1;
// 			}
			
			/*
			std::cout << "tvecBoardWorld " << std::endl << tvecBoardWorld << std::endl;
			std::cout << "tvecBoardPlacementMat " << std::endl << tvecBoardPlacementMat << std::endl;
			std::cout << "tvecBoardWorld " << std::endl << tvecBoardWorld << std::endl;*/
			
			
// 			std::cout << "placementMatToWorld " << std::endl << placementMatToWorld << std::endl;
// 			std::cout << "taskBoardToWorld " << std::endl << taskBoardToWorld << std::endl;
// 			cv::waitKey(1000);



			m_inputDataNames = inputDataNames;
			return isFilterSuccess;
		}

		// -------------------------------------------------------------------------------------------------

	} // namespace 

} // namespace 