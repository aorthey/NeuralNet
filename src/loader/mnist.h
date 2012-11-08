#pragma once
#include <string>
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
		void read_mnist(string fname, std::vector<VectorXd> &samples)
		{
				ifstream file (fname);
				if (file.is_open())
				{
						int magic_number=0;
						int number_of_images=0;
						int n_rows=0;
						int n_cols=0;
						file.read((char*)&magic_number,sizeof(magic_number)); 
						magic_number= reverseInt(magic_number);
						file.read((char*)&number_of_images,sizeof(number_of_images));
						number_of_images= reverseInt(number_of_images);
						file.read((char*)&n_rows,sizeof(n_rows));
						n_rows= reverseInt(n_rows);
						file.read((char*)&n_cols,sizeof(n_cols));
						n_cols= reverseInt(n_cols);
						for(int i=0;i<number_of_images;++i)
						{
								samples.at(i) = VectorXd(n_rows*n_cols);
								for(int r=0;r<n_rows;++r)
								{
										for(int c=0;c<n_cols;++c)
										{
												unsigned char temp=0;
												file.read((char*)&temp,sizeof(temp));
												samples.at(i)(c+r*n_cols) = temp;
										}
								}
						}
						return cur;
				}else{
					HALT("Could not open file");
				}
		}

		void get_training_data(std::vector<VectorXd> &samples){
			read_mnist("pt10k-images-idx3-ubyte.gz", samples);
		}
		VectorXd get_training_labels(){
			HALT("Not yet implemented");

		}
		VectorXd get_test_data(){
			//pt10k-images-idx3-ubyte.gz
			HALT("Not yet implemented");
		}
		VectorXd get_test_labels(){
			HALT("Not yet implemented");
		}
	};



};
