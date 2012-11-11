#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include "util.h"

//visualizing features in the net, showing error rates
class NeuralNetworkDisplayUnit{

	private:
	std::string windowName;

	public:

	cv::Mat VectorXd_to_cv_Mat(const VectorXd &x, uint dimX, uint dimY){
			assert(dimX*dimY == x.size());
			cv::Mat img(dimX,dimY, CV_8UC1);

			for(uint i=0;i<dimX;i++){
				for(uint j=0;j<dimY;j++){
					img.at<uchar>(i,j) = x(j+i*dimY);
				}
			}
			return img;
		}
		NeuralNetworkDisplayUnit(){ 
			windowName = "NeuralNetOutput";
		}

	void visualize_unit(uint l_hidden_layer, uint j_hidden_unit){

	}

	void visualize_sample(VectorXd &x, uint dimX, uint dimY){
		assert(dimX*dimY == x.size());
		using namespace cv;

		Mat img = VectorXd_to_cv_Mat(x, dimX, dimY);

		namedWindow(this->windowName.c_str(), CV_WINDOW_AUTOSIZE);
		PRINT("show sample");
		imshow( this->windowName.c_str(), img);

		waitKey(0);
	}
	void visualize_sample_matrix(std::vector<VectorXd> &x, uint dimX, uint dimY){

	}
	void error_score_on_data_set(){

	}

};
