#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include "loader.h"
#include "util.h"

namespace loader{
//code from http://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c

	class MnistLoader: public Loader{


		int reverseInt (int i) 
		{
				unsigned char c1, c2, c3, c4;

				c1 = i & 255;
				c2 = (i >> 8) & 255;
				c3 = (i >> 16) & 255;
				c4 = (i >> 24) & 255;

				return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
		}
		void read_mnist(std::string fname, std::vector<VectorXd> &samples, bool labels)
		{
				std::ifstream file(fname.c_str());
				if (file.is_open())
				{
						int magic_number;
						int number_of_images=0;
						int n_rows=0;
						int n_cols=0;
						file.seekg(0, std::ios::beg);
						file.read(reinterpret_cast<char*>(&magic_number),sizeof(int)); 
						magic_number= reverseInt(magic_number);
						file.read((char*)&number_of_images,sizeof(number_of_images));
						number_of_images= reverseInt(number_of_images);

						if(!labels){
							file.read((char*)&n_rows,sizeof(n_rows));
							n_rows= reverseInt(n_rows);
							file.read((char*)&n_cols,sizeof(n_cols));
							n_cols= reverseInt(n_cols);
						}
						for(int i=0;i<number_of_images;++i)
						{
								if(!labels){
								VectorXd v(n_cols*n_rows);
								samples.push_back(v);
								for(int r=0;r<n_rows;++r)
								{
										for(int c=0;c<n_cols;++c)
										{
												unsigned char temp=0;
												file.read((char*)&temp,sizeof(temp));
												samples.at(i)(c+r*n_cols) = temp;
										}
								}
								}else{
									VectorXd v(1);
									samples.push_back(v);
									unsigned char temp=0;
									file.read((char*)&temp,sizeof(temp));
									samples.at(i)(0) = temp;
								}
						}
				}else{
					HALT("Could not open file");
				}
		}

		void get_training_data(std::vector<VectorXd> &samples){
			read_mnist("../data/train-images-idx3-ubyte", samples, false);
		}
		void get_training_labels(std::vector<VectorXd> &labels){
			read_mnist("../data/train-labels-idx1-ubyte", labels, true);
		}
		void get_test_data(std::vector<VectorXd> &samples){
			read_mnist("../data/t10k-images-idx3-ubyte", samples, false);
		}
		void get_test_labels(std::vector<VectorXd> &labels){
			read_mnist("../data/t10k-labels-idx1-ubyte", labels, true);
		}
	};



};
